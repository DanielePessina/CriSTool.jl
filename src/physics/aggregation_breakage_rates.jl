## Aggregation

@inline _aggregation_kernel(::ConstantAggregation, kernel_scale, shape_factor, length_one, length_two) = kernel_scale
@inline _aggregation_kernel(::SizeSumAggregation, kernel_scale, shape_factor, length_one, length_two) =
    kernel_scale * (length_one + length_two)
@inline _aggregation_kernel(::VolumeSumAggregation, kernel_scale, shape_factor, length_one, length_two) =
    kernel_scale * shape_factor * (length_one^3 + length_two^3)
@inline _aggregation_kernel(::AverageAggregation, kernel_scale, shape_factor, length_one, length_two) =
    kernel_scale * (length_one + length_two) / 2

"""Built-in aggregation kernels evaluated by the fixed-pivot mesh operator."""
const FixedPivotAggregationLaw = Union{ConstantAggregation, SizeSumAggregation,
                                       VolumeSumAggregation, AverageAggregation}

"""
    FixedPivotAggregationTable(cell_centre, cell_face)

Mesh-only geometry of the fixed-pivot aggregation operator. Entry `k` belongs
to the `k`-th unordered cell pair `(lower_index, upper_index)` with
`lower_index ≤ upper_index`, enumerated with `upper_index` running fastest.
The product volume `Lᵢ^3 + Lⱼ^3` is split between the neighbouring pivots
`aggregation_deposit_index[k]` and `aggregation_deposit_index[k] + 1` with
the volume-linear weights `aggregation_lower_weight[k]` and
`aggregation_upper_weight[k]`:

- a product at or below the first pivot goes wholly to the first cell;
- a product between the last pivot and the upper mesh face stays in the
  last cell;
- a product above the upper mesh face leaves the finite domain (both weights
  are zero).

The table depends only on the mesh, so a solve builds it once and every RHS
call reuses it.
"""
struct FixedPivotAggregationTable
    aggregation_deposit_index::Vector{Int}
    aggregation_lower_weight::Vector{Float64}
    aggregation_upper_weight::Vector{Float64}
end

function FixedPivotAggregationTable(cell_centre::AbstractVector, cell_face::AbstractVector)
    cell_count = length(cell_centre)
    cell_count >= 2 ||
        throw(ArgumentError("fixed-pivot aggregation needs at least two mesh cells."))
    pair_count = cell_count * (cell_count + 1) ÷ 2
    deposit_index = Vector{Int}(undef, pair_count)
    lower_weight = Vector{Float64}(undef, pair_count)
    upper_weight = Vector{Float64}(undef, pair_count)
    pivot_volume = [centre_length^3 for centre_length in cell_centre]
    first_volume = first(pivot_volume)
    last_volume = last(pivot_volume)
    maximum_volume = last(cell_face)^3

    pair_index = 0
    for lower_index in 1:cell_count
        # Product volumes increase with the upper partner, so the lower pivot
        # only moves forward along a row.
        pivot_index = 1
        for upper_index in lower_index:cell_count
            pair_index += 1
            product_volume = pivot_volume[lower_index] + pivot_volume[upper_index]
            while pivot_index < cell_count && pivot_volume[pivot_index + 1] <= product_volume
                pivot_index += 1
            end
            if product_volume <= first_volume
                deposit_index[pair_index] = 1
                lower_weight[pair_index] = 1.0
                upper_weight[pair_index] = 0.0
            elseif product_volume > maximum_volume
                deposit_index[pair_index] = 1
                lower_weight[pair_index] = 0.0
                upper_weight[pair_index] = 0.0
            elseif product_volume >= last_volume
                deposit_index[pair_index] = cell_count - 1
                lower_weight[pair_index] = 0.0
                upper_weight[pair_index] = 1.0
            else
                pivot_upper_weight = (product_volume - pivot_volume[pivot_index]) /
                                     (pivot_volume[pivot_index + 1] - pivot_volume[pivot_index])
                deposit_index[pair_index] = pivot_index
                lower_weight[pair_index] = 1 - pivot_upper_weight
                upper_weight[pair_index] = pivot_upper_weight
            end
        end
    end
    return FixedPivotAggregationTable(deposit_index, lower_weight, upper_weight)
end

FixedPivotAggregationTable(solver::AbstractDiscretisedSolver) =
    FixedPivotAggregationTable(solver.cell_centre, solver.cell_face)

"""
    _mesh_aggregation_table(aggregationfunction, solver)

Mesh geometry an aggregation law needs on every RHS call: a
[`FixedPivotAggregationTable`](@ref) for the built-in kernels, `nothing` for
`NoAggregation` and for custom laws (which supply `aggregationrate`).
"""
_mesh_aggregation_table(aggregationfunction::AbstractAggregationFunction, solver) = nothing
_mesh_aggregation_table(aggregationfunction::FixedPivotAggregationLaw,
                        solver::AbstractDiscretisedSolver) = FixedPivotAggregationTable(solver)

"""
    _add_aggregation_rate!(rate, aggregationfunction, parameters, problem, state, time,
                           aggregation_table)

Add the aggregation number-density rate to the mesh vector `rate` in place.
`NoAggregation` adds nothing, a custom law adds its `aggregationrate`, and the
built-in kernels run the allocation-free fixed-pivot operator on the
precomputed [`FixedPivotAggregationTable`](@ref).
"""
@inline _add_aggregation_rate!(rate, ::NoAggregation, parameters, problem, state, time,
                               ::Nothing) = rate

function _add_aggregation_rate!(rate, aggregationfunction::AbstractAggregationFunction,
                                parameters, problem, state, time, ::Nothing)
    rate .+= aggregationrate(aggregationfunction, parameters, problem, state, time)
    return rate
end

# Binary Smoluchowski aggregation on the discretised length grid. Each cell is
# represented by its centre and contains `numberdensity[i] * cell_dL`
# particles per suspension volume. Aggregation adds crystal volumes, so a
# product with volume `Lᵢ^3 + Lⱼ^3` is deposited by volume-linear fixed-pivot
# interpolation between neighbouring cell centres.
#
# The pair loop uses unordered collisions: unequal-size pairs occur once and
# equal-size pairs carry the usual factor `1/2`. This gives the correct
# particle number loss and conserves the represented crystal volume for
# products between the first and last pivots. Products above the upper mesh
# face are treated as outflow from the finite domain; products between the
# last pivot and that face are retained in the last cell.
function _add_aggregation_rate!(rate, aggregationfunction::FixedPivotAggregationLaw,
                                parameters, problem, state, time,
                                aggregation_table::FixedPivotAggregationTable)
    numberdensity = crystal_state(problem, state)
    length_mesh = problem.solver.cell_centre
    cell_dL = problem.solver.cell_dL
    shape_factor = problem.model.system.volume_shape_factor
    kernel_scale = exp10(_named_params(aggregationfunction, parameters).
                         log10_aggregation_coefficient)
    deposit_index = aggregation_table.aggregation_deposit_index
    lower_weight = aggregation_table.aggregation_lower_weight
    upper_weight = aggregation_table.aggregation_upper_weight
    cell_count = length(numberdensity)
    length(rate) == cell_count == length(length_mesh) ||
        throw(ArgumentError("aggregation rate, population and mesh must have the same length."))
    length(deposit_index) == cell_count * (cell_count + 1) ÷ 2 ||
        throw(ArgumentError("aggregation table does not match the mesh."))

    pair_index = 0
    @inbounds for lower_index in 1:cell_count
        lower_length = length_mesh[lower_index]
        # Particles per volume in the lower cell. A pair's number-density rate
        # is its event rate divided by cell_dL, i.e. K * lower_number * nⱼ.
        lower_number = numberdensity[lower_index] * cell_dL
        lower_loss = zero(lower_number * kernel_scale)
        for upper_index in lower_index:cell_count
            pair_index += 1
            collision_kernel = _aggregation_kernel(aggregationfunction, kernel_scale,
                                                   shape_factor, lower_length,
                                                   length_mesh[upper_index])
            pair_rate = collision_kernel * lower_number * numberdensity[upper_index]
            lower_loss += pair_rate
            if upper_index == lower_index
                # Equal-size pairs: half the events, each losing two particles.
                pair_rate = pair_rate / 2
            else
                rate[upper_index] -= pair_rate
            end
            target_index = deposit_index[pair_index]
            rate[target_index] += lower_weight[pair_index] * pair_rate
            rate[target_index + 1] += upper_weight[pair_index] * pair_rate
        end
        rate[lower_index] -= lower_loss
    end
    return rate
end

"""
    aggregationrate(::NoAggregation, parameters::AbstractVector,
                    problem::CrystallisationProblem, state, t) -> Real

Return zero aggregation rate (placeholder for no aggregation).

# Returns
- 0.0
"""
function aggregationrate(::NoAggregation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return 0.0
end

"""
    aggregationrate(aggregationfunction, parameters::AbstractVector,
                    problem::CrystallisationProblem, state, t) -> Vector

Number-density aggregation rate on the solver mesh for the built-in kernels
`ConstantAggregation`, `SizeSumAggregation`, `VolumeSumAggregation` and
`AverageAggregation`: the binary Smoluchowski operator with volume-conserving
fixed-pivot deposition of the products.

# Arguments
- `parameters`: Vector [log10_aggregation_coefficient], the base-10 logarithm
  of the SI aggregation coefficient. Its dimensional unit depends on the
  selected kernel family.
- `problem`: Crystallisation problem with the discretised solver mesh.
- `state`: Current solver state, including the mesh number-density population.
- `t`: Time in seconds.

# Returns
- Vector of aggregation rates at each cell
"""
function aggregationrate(aggregationfunction::FixedPivotAggregationLaw, parameters::T,
                         prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    numberdensity = crystal_state(prob, state)
    kernel_scale = exp10(_named_params(aggregationfunction, parameters).
                         log10_aggregation_coefficient)
    rate_type = promote_type(eltype(numberdensity), typeof(kernel_scale))
    aggregation_rate = zeros(rate_type, length(numberdensity))
    return _add_aggregation_rate!(aggregation_rate, aggregationfunction, parameters, prob,
                                  state, t, FixedPivotAggregationTable(prob.solver))
end

## Breakage

@inline _breakage_frequency(breakagefunction::EmpiricalBreakage, parameters, crystal_length) =
    parameters.breakage_coefficient *
    crystal_length^(3 * parameters.breakage_size_exponent)
@inline _breakage_frequency(breakagefunction::UniformBreakage, parameters, crystal_length) =
    exp(parameters.ln_breakage_coefficient) *
    (crystal_length / breakagefunction.reference_length)^(
        3 * parameters.breakage_size_exponent)

"""Built-in breakage laws with binary daughters uniform in crystal volume."""
const UniformVolumeBreakageLaw = Union{EmpiricalBreakage, UniformBreakage}

"""
    _add_breakage_rate!(rate, breakagefunction, parameters, problem, state, time)

Add the breakage number-density rate to the mesh vector `rate` in place.
`NoBreakage` adds nothing, a custom law adds its `breakagerate`, and the
built-in laws run the allocation-free uniform-in-volume daughter operator.
"""
@inline _add_breakage_rate!(rate, ::NoBreakage, parameters, problem, state, time) = rate

function _add_breakage_rate!(rate, breakagefunction::AbstractBreakageFunction,
                             parameters, problem, state, time)
    rate .+= breakagerate(breakagefunction, parameters, problem, state, time)
    return rate
end

# Binary breakage with a daughter distribution uniform in crystal volume. A
# parent of representative volume `Vⱼ = Lⱼ^3` produces two daughters; the
# fraction placed in a length cell is the exact volume overlap `2 ΔV / Vⱼ`.
# The corresponding continuous daughter density is `b(L | Lⱼ) = 6L^2 / Lⱼ^3`
# for `0 ≤ L ≤ Lⱼ`.
#
# A parent's centre lies inside its own cell, so its daughters fill every
# smaller cell completely and its own cell up to the centre. With face lengths
# `f` and parent rates `rⱼ = Γ(Lⱼ) nⱼ` the birth in cell `i` is
# `(f_{i+1}^3 - f_i^3) Σ_{j>i} 2rⱼ/Vⱼ + (Lᵢ^3 - f_i^3) 2rᵢ/Vᵢ`, which one sweep
# from the largest cell downwards evaluates in O(N) instead of O(N²).
#
# The empirical and logarithmic parameterisations share this daughter law but
# use their historical breakage-frequency parameter conventions.
function _add_breakage_rate!(rate, breakagefunction::UniformVolumeBreakageLaw,
                             parameters, problem, state, time)
    named_parameters = _named_params(breakagefunction, parameters)
    numberdensity = crystal_state(problem, state)
    length_mesh = problem.solver.cell_centre
    cell_faces = problem.solver.cell_face
    length(rate) == length(numberdensity) == length(length_mesh) ||
        throw(ArgumentError("breakage rate, population and mesh must have the same length."))

    larger_parent_daughter_density = zero(eltype(rate))
    @inbounds for cell_index in reverse(eachindex(numberdensity))
        parent_length = length_mesh[cell_index]
        parent_volume = parent_length^3
        lower_face_volume = cell_faces[cell_index]^3
        upper_face_volume = cell_faces[cell_index + 1]^3
        parent_rate = numberdensity[cell_index] *
                      _breakage_frequency(breakagefunction, named_parameters, parent_length)
        own_daughter_density = 2 * parent_rate / parent_volume
        rate[cell_index] += (upper_face_volume - lower_face_volume) *
                            larger_parent_daughter_density +
                            (parent_volume - lower_face_volume) * own_daughter_density -
                            parent_rate
        larger_parent_daughter_density += own_daughter_density
    end
    return rate
end

"""
    breakagerate(::NoBreakage, parameters::AbstractVector,
                 problem::CrystallisationProblem, state, t) -> Real

Return zero breakage rate (placeholder for no breakage).

# Returns
- 0.0
"""
function breakagerate(::NoBreakage, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return 0.0
end

"""
    breakagerate(breakagefunction, parameters::AbstractVector,
                 problem::CrystallisationProblem, state, t) -> Vector

Number-density breakage rate (daughter birth minus parent death) on the solver
mesh for `EmpiricalBreakage` and `UniformBreakage`, whose binary daughters are
uniform in crystal volume.

# Arguments
- `parameters`: `EmpiricalBreakage` takes [breakage_coefficient,
  breakage_size_exponent]; `UniformBreakage` takes [ln_breakage_coefficient,
  breakage_size_exponent].
- `problem`: Crystallisation problem with the discretised solver mesh.
- `state`: Current solver state, including the mesh number-density population.
- `t`: Time in seconds.

# Returns
- Vector of breakage rates at each cell
"""
function breakagerate(breakagefunction::UniformVolumeBreakageLaw, parameters::T,
                      prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    numberdensity = crystal_state(prob, state)
    rate_type = promote_type(eltype(numberdensity), eltype(parameters))
    breakage_rate = zeros(rate_type, length(numberdensity))
    return _add_breakage_rate!(breakage_rate, breakagefunction, parameters, prob, state, t)
end
