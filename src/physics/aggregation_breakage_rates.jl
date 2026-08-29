## Aggregation

@inline _aggregation_kernel(::aggr_scalar, kernel_scale, shape_factor, length_one, length_two) = kernel_scale
@inline _aggregation_kernel(::aggr_linear, kernel_scale, shape_factor, length_one, length_two) =
    kernel_scale * (length_one + length_two)
@inline _aggregation_kernel(::aggr_linearvol, kernel_scale, shape_factor, length_one, length_two) =
    kernel_scale * shape_factor * (length_one^3 + length_two^3)
@inline _aggregation_kernel(::aggr_avg, kernel_scale, shape_factor, length_one, length_two) =
    kernel_scale * (length_one + length_two) / 2

@inline function _aggregation_pivot_index(length_mesh, product_volume)
    lower_index = firstindex(length_mesh)
    upper_index = lastindex(length_mesh)
    while lower_index <= upper_index
        middle_index = (lower_index + upper_index) >>> 1
        if length_mesh[middle_index]^3 <= product_volume
            lower_index = middle_index + 1
        else
            upper_index = middle_index - 1
        end
    end
    return upper_index
end

function _deposit_aggregation_birth!(aggregation_rate,
                                     product_volume,
                                     event_rate,
                                     length_mesh,
                                     maximum_volume,
                                     cell_dL,
                                     lower_index)
    first_volume = first(length_mesh)^3
    last_volume = last(length_mesh)^3
    if product_volume <= first_volume
        aggregation_rate[begin] += event_rate / cell_dL
    elseif product_volume > maximum_volume
        return nothing
    elseif product_volume >= last_volume
        aggregation_rate[end] += event_rate / cell_dL
    else
        lower_volume = length_mesh[lower_index]^3
        upper_volume = length_mesh[lower_index + 1]^3
        upper_weight = (product_volume - lower_volume) /
                       (upper_volume - lower_volume)
        lower_weight = 1 - upper_weight
        aggregation_rate[lower_index] += lower_weight * event_rate / cell_dL
        aggregation_rate[lower_index + 1] += upper_weight * event_rate / cell_dL
    end
    return nothing
end

"""
    _aggregation_rate_discrete(aggregationfunction, parameters, problem, state)

Evaluate the binary Smoluchowski aggregation operator on the discretised
length grid. Each cell is represented by its centre and contains
`numberdensity[i] * cell_dL` particles per suspension volume. Aggregation
adds crystal volumes, so a product with volume `Lᵢ^3 + Lⱼ^3` is deposited by
volume-linear fixed-pivot interpolation between neighbouring cell centres.

The pair loop uses unordered collisions: unequal-size pairs occur once and
equal-size pairs carry the usual factor `1/2`. This gives the correct particle
number loss and conserves the represented crystal volume for products between
the first and last pivots. Products above the upper mesh face are treated as
outflow from the finite domain; products between the last pivot and that face
are retained in the last cell.
"""
function _aggregation_rate_discrete(aggregationfunction,
                                    parameters,
                                    problem,
                                    state)
    numberdensity = crystal_state(problem, state)
    length_mesh = problem.solver.cell_centre
    maximum_volume = last(problem.solver.cell_face)^3
    cell_dL = problem.solver.cell_dL
    kernel_scale = exp10(_named_params(aggregationfunction, parameters).logβ)
    rate_type = promote_type(eltype(numberdensity), typeof(kernel_scale))
    aggregation_rate = zeros(rate_type, length(numberdensity))

    @inbounds for lower_index in eachindex(numberdensity)
        lower_number = numberdensity[lower_index] * cell_dL
        lower_length = length_mesh[lower_index]
        pivot_index = _aggregation_pivot_index(length_mesh, 2 * lower_length^3)
        for upper_index in lower_index:length(numberdensity)
            upper_number = numberdensity[upper_index] * cell_dL
            collision_kernel = _aggregation_kernel(aggregationfunction,
                                                    kernel_scale,
                                                    problem.kv,
                                                    lower_length,
                                                    length_mesh[upper_index])
            event_rate = collision_kernel * lower_number * upper_number /
                         (lower_index == upper_index ? 2 : 1)

            if lower_index == upper_index
                aggregation_rate[lower_index] -= 2 * event_rate / cell_dL
            else
                aggregation_rate[lower_index] -= event_rate / cell_dL
                aggregation_rate[upper_index] -= event_rate / cell_dL
            end

            product_volume = lower_length^3 + length_mesh[upper_index]^3
            while pivot_index < lastindex(length_mesh) &&
                  length_mesh[pivot_index + 1]^3 <= product_volume
                pivot_index += 1
            end
            _deposit_aggregation_birth!(aggregation_rate,
                                        product_volume,
                                        event_rate,
                                        length_mesh,
                                        maximum_volume,
                                        cell_dL,
                                        pivot_index)
        end
    end
    return aggregation_rate
end

"""
    aggregationrate(::noaggregation, parameters::AbstractVector, mesh::AbstractVector,
                    numberdensity::AbstractVector) -> Real

Return zero aggregation rate (placeholder for no aggregation).

# Returns
- 0.0
"""
function aggregationrate(::noaggregation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return 0.0
end

"""
    aggregationrate(::aggr_scalar, parameters::AbstractVector, fullmesh::AbstractVector,
                    fullnumberdensity::AbstractVector) -> Vector

Calculate size-independent (scalar) aggregation rate.

# Arguments
- `parameters`: Vector [log10(β)] where β is aggregation kernel constant
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of aggregation rates at each cell
"""
function aggregationrate(af::aggr_scalar, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return _aggregation_rate_discrete(af, parameters, prob, state)
end

"""
    aggregationrate(::aggr_linear, parameters::AbstractVector, fullmesh::AbstractVector,
                    fullnumberdensity::AbstractVector) -> Vector

Calculate linear size-dependent aggregation rate (kernel proportional to sum of sizes).

# Arguments
- `parameters`: Vector [log10(β)] where β is aggregation kernel constant
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of aggregation rates at each cell
"""
function aggregationrate(af::aggr_linear, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return _aggregation_rate_discrete(af, parameters, prob, state)
end

"""
    aggregationrate(::aggr_linearvol, parameters::AbstractVector, fullmesh::AbstractVector,
                    fullnumberdensity::AbstractVector) -> Vector

Calculate linear volume-dependent aggregation rate (kernel proportional to sum of volumes).

# Arguments
- `parameters`: Vector [log10(β)] where β is aggregation kernel constant
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of aggregation rates at each cell
"""
function aggregationrate(af::aggr_linearvol, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return _aggregation_rate_discrete(af, parameters, prob, state)
end

function aggregationrate(af::aggr_avg, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return _aggregation_rate_discrete(af, parameters, prob, state)
end

@inline _breakage_frequency(breakagefunction::breakage_empirical, parameters, crystal_length) =
    parameters.b * crystal_length^(3 * parameters.n)
@inline _breakage_frequency(breakagefunction::breakage_uniform, parameters, crystal_length) =
    exp(parameters.logb) * (CRISTOOL_MICROMETER_SCALE * crystal_length)^(3 * parameters.n)

"""
    _breakage_rate_uniform_volume(breakagefunction, parameters, problem, state)

Evaluate binary breakage with a daughter distribution uniform in crystal
volume. A parent of representative volume `Vⱼ = Lⱼ^3` produces two daughters;
the fraction placed in a length cell is the exact volume overlap
`2 * ΔV / Vⱼ`. The corresponding continuous daughter density is
`b(L | Lⱼ) = 6L^2 / Lⱼ^3` for `0 ≤ L ≤ Lⱼ`.

The empirical and logarithmic parameterisations share this daughter law but
use their historical breakage-frequency parameter conventions.
"""
function _breakage_rate_uniform_volume(breakagefunction,
                                       parameters,
                                       problem,
                                       state)
    named_parameters = _named_params(breakagefunction, parameters)
    numberdensity = crystal_state(problem, state)
    length_mesh = problem.solver.cell_centre
    cell_faces = problem.solver.cell_face
    cell_dL = problem.solver.cell_dL
    rate_scale = breakagefunction isa breakage_empirical ? named_parameters.b :
                 named_parameters.logb
    rate_type = promote_type(eltype(numberdensity), typeof(rate_scale))
    breakage_rate = zeros(rate_type, length(numberdensity))

    @inbounds for parent_index in eachindex(numberdensity)
        parent_length = length_mesh[parent_index]
        parent_volume = parent_length^3
        parent_frequency = _breakage_frequency(breakagefunction,
                                               named_parameters,
                                               parent_length)
        parent_birth_rate = numberdensity[parent_index] * parent_frequency

        for child_index in eachindex(numberdensity)
            child_lower = cell_faces[child_index]
            child_upper = min(cell_faces[child_index + 1], parent_length)
            volume_overlap = child_upper^3 - child_lower^3
            if volume_overlap > 0
                daughter_count = 2 * volume_overlap / parent_volume
                breakage_rate[child_index] += parent_birth_rate * daughter_count
            end
            child_lower >= parent_length && break
        end
        breakage_rate[parent_index] -= numberdensity[parent_index] * parent_frequency
    end
    return breakage_rate
end

"""
    breakagerate(::nobreakage, parameters::AbstractVector, fullmesh::AbstractVector,
                 fullnumberdensity::AbstractVector) -> Real

Return zero breakage rate (placeholder for no breakage).

# Returns
- 0.0
"""
function breakagerate(::nobreakage, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return 0.0
end

"""
    breakagerate(::breakage_empirical, parameters::AbstractVector, mesh::AbstractVector,
                 numberdensity::AbstractVector) -> Real

Calculate empirical breakage rate.

# Arguments
- `parameters`: Vector [breakage constant, breakage exponent]
- `mesh`: Cell center positions
- `numberdensity`: Number density at each cell

# Returns
- Breakage rate contribution
"""
function breakagerate(bf::breakage_empirical, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return _breakage_rate_uniform_volume(bf, parameters, prob, state)
end

"""
    breakagerate(::breakage_uniform, parameters::AbstractVector, fullmesh::AbstractVector,
                 fullnumberdensity::AbstractVector) -> Vector

Calculate uniform breakage rate (daughter fragments uniformly distributed).

# Arguments
- `parameters`: Vector [log(breakage constant), breakage exponent]
- `fullmesh`: Cell center positions
- `fullnumberdensity`: Number density at each cell

# Returns
- Vector of breakage rates at each cell
"""
function breakagerate(bf::breakage_uniform, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    return _breakage_rate_uniform_volume(bf, parameters, prob, state)
end
