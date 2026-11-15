"""
Post-processing utilities for crystallisation simulations. Computes moments, quantiles and weighted densities from solver outputs.
"""

"""
    momentcalculator(mesh, numberdensity, moment)

Calculate the n-th moment of a number density distribution.
Optimized to reduce allocations by using explicit loops.
"""
function momentcalculator(mesh::AbstractVector, numberdensity::AbstractVector,
                          moment::Int64)
    step_size = Base.step(mesh)
    result = zero(eltype(numberdensity))
    @inbounds for i in eachindex(numberdensity, mesh)
        result += step_size * numberdensity[i] * mesh[i]^moment
    end
    return result
end

"""
    momentcalculator(mesh::AbstractVector, numberdensity::AbstractMatrix, moment::Int64) -> Vector

Calculate the n-th moment of number density distributions for multiple time points.

# Arguments
- `mesh::AbstractVector`: Particle size grid (uniform spacing required)
- `numberdensity::AbstractMatrix`: Number density matrix of size (N_grid, N_time)
- `moment::Int64`: Order of the moment to calculate (0, 1, 2, 3, or 4)

# Returns
- `Vector`: Vector of moment values for each time point
"""
function momentcalculator(mesh::AbstractVector, numberdensity::AbstractMatrix,
                          moment::Int64)
    step_size = Base.step(mesh)
    ncols = size(numberdensity, 2)
    result = Vector{eltype(numberdensity)}(undef, ncols)
    @inbounds for j in 1:ncols
        col_sum = zero(eltype(numberdensity))
        for i in eachindex(mesh)
            col_sum += step_size * numberdensity[i, j] * mesh[i]^moment
        end
        result[j] = col_sum
    end
    return result
end

"""
    volumeweighteddensity(mesh, numberdensity, shape_factor)

Compute volume-weighted density distribution.
"""
function volumeweighteddensity(mesh::AbstractVector, numberdensity::AbstractVecOrMat, shape_factor)
    # Integer literal powers lower to multiplications; a Float64 exponent calls `pow`
    # once per cell and saved time.
    shape_factor .* (numberdensity .* (mesh .^ 3))
end

"""
    surfaceweighteddensity(mesh, numberdensity, shape_factor)

Compute surface-weighted density distribution.
"""
function surfaceweighteddensity(mesh::AbstractVector, numberdensity::AbstractVector, shape_factor)
    shape_factor .* (numberdensity .* mesh .^ 2)
end

"""
    quantilecalculator(cellcentre, voldensity, quantile)

Calculate the particle size at a specified quantile from the volume density distribution.
Optimized to reduce allocations by computing cumulative sum in-place.
"""
function quantilecalculator(cellcentre::AbstractVector{T}, voldensity::AbstractVector{K},
                            quantile::W) where {T <: Real, K <: Real, W <: Real}
    # Early return if voldensity is approximately zero
    if all(x -> abs(x) < eps(K), voldensity)
        return zero(K)
    end

    step_size = cellcentre[2] - cellcentre[1]

    # Compute total integral first (avoid allocating full cumsum vector)
    total = zero(K)
    @inbounds for i in eachindex(voldensity)
        total += voldensity[i] * step_size
    end

    if total ≈ 0
        return zero(K)
    end

    # Find quantile by iterating through cumulative sum
    target = quantile * total
    cumulative = zero(K)
    @inbounds for i in eachindex(voldensity)
        prev_cumulative = cumulative
        cumulative += voldensity[i] * step_size

        if cumulative >= target
            if i == 1
                return zero(K)
            else
                # Linear interpolation
                x0, x1 = prev_cumulative / total, cumulative / total
                y0, y1 = cellcentre[i-1], cellcentre[i]
                return y0 + (quantile - x0) * (y1 - y0) / (x1 - x0)
            end
        end
    end

    return cellcentre[end]
end

"""
    quantilecalculator(cellcentre::AbstractVector{T}, voldensitymatrix::AbstractMatrix{K}, quantiles::W) -> Vector where {T<:Real, K<:Real, W<:Real}

Calculate particle size at a specified quantile for multiple time points.

# Arguments
- `cellcentre::AbstractVector{T}`: Particle size grid cell centres
- `voldensitymatrix::AbstractMatrix{K}`: Volume density matrix of size (N_grid, N_time)
- `quantiles::W`: Quantile value (0-1, e.g., 0.5 for median)

# Returns
- `Vector`: Particle sizes (in m) at the specified quantile for each time point
"""
function quantilecalculator(cellcentre::AbstractVector{T},
                            voldensitymatrix::AbstractMatrix{K},
                            quantiles::W) where {T <: Real, K <: Real, W <: Real}
    quant_matrix = map(eachcol(voldensitymatrix)) do col
        quantilecalculator(cellcentre, col, quantiles)
    end
    return quant_matrix
end

"""
    ReconstructionFailure(value, message)

Status of a saved trajectory that cannot be reconstructed into physical size
metrics or a quadrature population: the offending `value` and the reason.
Reconstruction returns it instead of throwing. Direct simulation raises it as
`DomainError(value, message)`; prepared inference records it as a failed
candidate.
"""
struct ReconstructionFailure{Value <: Real}
    value::Value
    message::String
end

"""The `DomainError` that reports `failure` to callers outside prepared inference."""
_reconstruction_domain_error(failure::ReconstructionFailure) =
    DomainError(failure.value, failure.message)

"""The first `ReconstructionFailure` among `statuses`, or `nothing`."""
_first_reconstruction_failure() = nothing
_first_reconstruction_failure(failure::ReconstructionFailure, remaining_statuses...) = failure
_first_reconstruction_failure(reconstructed, remaining_statuses...) =
    _first_reconstruction_failure(remaining_statuses...)

"""
    _reconstruct_from(build, statuses...)

`build(statuses...)` when no status is a `ReconstructionFailure`; otherwise the
first failure, in argument order.
"""
_reconstruct_from(build, statuses...) =
    _reconstruct_from(build, _first_reconstruction_failure(statuses...), statuses)
_reconstruct_from(build, ::Nothing, statuses::Tuple) = build(statuses...)
_reconstruct_from(build, failure::ReconstructionFailure, statuses::Tuple) = failure

"""A reconstructed value, or throw a reconstruction failure as its `DomainError`."""
_reconstructed_or_throw(reconstructed) = reconstructed
_reconstructed_or_throw(failure::ReconstructionFailure) =
    throw(_reconstruction_domain_error(failure))

"""Whether a particle count is numerically empty, so its size ratios are zero."""
_is_empty_moment_population(particle_count) =
    particle_count >= zero(particle_count) && particle_count <= CRISTOOL_MOMENT_DENSITY_FLOOR

"""`ReconstructionFailure` when the size ratio `numerator / denominator` is undefined, else `nothing`."""
function _moment_size_ratio_failure(numerator, denominator, particle_count)
    isfinite(particle_count) ||
        return ReconstructionFailure(particle_count,
            "Particle count must be finite when computing a size metric.")
    _is_empty_moment_population(particle_count) && return nothing
    isfinite(numerator) && numerator >= zero(numerator) ||
        return ReconstructionFailure(numerator,
            "A nonempty population requires a finite nonnegative moment numerator.")
    isfinite(denominator) && denominator > zero(denominator) ||
        return ReconstructionFailure(denominator,
            "A nonempty population requires a finite positive moment denominator.")
    return nothing
end

"""Size ratio `numerator / denominator` of a checked state, zero for a numerically empty population."""
_moment_size_ratio(numerator, denominator, particle_count) =
    _is_empty_moment_population(particle_count) ?
        zero(promote_type(typeof(numerator), typeof(denominator))) : numerator / denominator

"""
    _momentsizes(mesh, numberdensity)

Moment-derived size metrics (d10, d32, d43, moment2) of a mesh number-density
matrix, computed once when a mesh solution is assembled, or the
`ReconstructionFailure` of an undefined ratio.
"""
function _momentsizes(mesh::AbstractVector, numberdensity::AbstractMatrix)
    step_size = Base.step(mesh)
    n_columns = size(numberdensity, 2)
    density_type = eltype(numberdensity)
    moment_matrix = Matrix{density_type}(undef, 5, n_columns)

    # The mesh is shared by every saved time.  Hoist its powers once, then
    # traverse each contiguous density column once to accumulate μ₀…μ₄.
    mesh_squared = Vector{eltype(mesh)}(undef, length(mesh))
    mesh_cubed = similar(mesh_squared)
    mesh_fourth = similar(mesh_squared)
    @inbounds for (mesh_slot, mesh_index) in enumerate(eachindex(mesh))
        mesh_value = mesh[mesh_index]
        mesh_squared[mesh_slot] = mesh_value^2
        mesh_cubed[mesh_slot] = mesh_value^3
        mesh_fourth[mesh_slot] = mesh_value^4
    end

    @inbounds for column_index in 1:n_columns
        mu0 = zero(density_type)
        mu1 = zero(density_type)
        mu2 = zero(density_type)
        mu3 = zero(density_type)
        mu4 = zero(density_type)
        mesh_slot = 1
        for mesh_index in eachindex(mesh)
            weighted_density = step_size * numberdensity[mesh_index, column_index]
            mesh_value = mesh[mesh_index]
            mu0 += weighted_density
            mu1 += weighted_density * mesh_value
            mu2 += weighted_density * mesh_squared[mesh_slot]
            mu3 += weighted_density * mesh_cubed[mesh_slot]
            mu4 += weighted_density * mesh_fourth[mesh_slot]
            mesh_slot += 1
        end
        moment_matrix[1, column_index] = mu0
        moment_matrix[2, column_index] = mu1
        moment_matrix[3, column_index] = mu2
        moment_matrix[4, column_index] = mu3
        moment_matrix[5, column_index] = mu4
    end
    return _raw_moment_size_metrics(moment_matrix)
end

"""
    _moment_ratio_trajectory(moment_matrix, numerator_row, denominator_row)

Saved size ratios `µ[numerator_row] / µ[denominator_row]`, `NaN` when the
numerator moment is not tracked, or the first `ReconstructionFailure` in time.
"""
function _moment_ratio_trajectory(moment_matrix::AbstractMatrix, numerator_row::Int,
                                  denominator_row::Int)
    tracked_moments, saved_count = size(moment_matrix)
    tracked_moments >= numerator_row ||
        return fill(convert(eltype(moment_matrix), NaN), saved_count)
    for saved_index in 1:saved_count
        ratio_failure = _moment_size_ratio_failure(moment_matrix[numerator_row, saved_index],
                                                   moment_matrix[denominator_row, saved_index],
                                                   moment_matrix[1, saved_index])
        ratio_failure === nothing || return ratio_failure
    end
    return [_moment_size_ratio(moment_matrix[numerator_row, saved_index],
                               moment_matrix[denominator_row, saved_index],
                               moment_matrix[1, saved_index]) for saved_index in 1:saved_count]
end

"""
    _raw_moment_size_metrics(moment_matrix)

Moment-ratio size metrics from a raw-moment matrix `(nmoments, nt)` whose row
`k + 1` holds µk. A ratio whose numerator moment is not tracked is `NaN`. An
undefined ratio returns its `ReconstructionFailure` (d10, then d32, then d43).
"""
_raw_moment_size_metrics(moment_matrix::AbstractMatrix) =
    _reconstruct_from(_moment_ratio_trajectory(moment_matrix, 2, 1),
                      _moment_ratio_trajectory(moment_matrix, 4, 3),
                      _moment_ratio_trajectory(moment_matrix, 5, 4)) do d10, d32, d43
        (; d10, d32, d43, moment2 = collect(view(moment_matrix, 3, :)))
    end

"""
Assemble a `CrystallisationSolution` from an ODE solution and its population;
size metrics that failed reconstruction return their `ReconstructionFailure`.
"""
_crystallisation_solution(problem::DiscretisedProblem, ode_solution, population,
                          size_failure::ReconstructionFailure, final_state,
                          reactor_state::NamedTuple) = size_failure

function _crystallisation_solution(problem::DiscretisedProblem, ode_solution, population,
                                   size_moments::NamedTuple, final_state, reactor_state::NamedTuple)
    solvent_trajectories = _solvent_solution_state(problem, ode_solution)
    return CrystallisationSolution(collect(ode_solution.t), solvent_trajectories.concentration,
                                   population, size_moments.d10, size_moments.d32,
                                   size_moments.d43, size_moments.moment2,
                                   solvent_trajectories, reactor_state, final_state,
                                   ode_solution.stats, ode_solution.retcode)
end

"""
    state_vars(solution) -> NamedTuple

Named solvent-state and reactor trajectories (including `concentration`), plus
the population arrays of the solver: `numberdensity`/`voldensity` (mesh ×
time) for mesh solvers, `moments` for MoM, and `moments`/`nodes`/`weights`
for QMOM and DQMOM.
"""
state_vars(solution::CrystallisationSolution) =
    merge(solution.solvent_state, solution.reactor_state, _population_vars(solution.population))

_population_vars(population::MeshPopulation) =
    (; numberdensity = population.numberdensity, voldensity = population.voldensity)
_population_vars(population::MomentPopulation) = (; moments = population.moments)
_population_vars(population::QuadraturePopulation) =
    (; moments = population.moments, nodes = population.nodes, weights = population.weights)

"""
    size_metrics(solution) -> NamedTuple

Particle-size trajectories: `d10`, `d32`, `d43`, `moment2` for every solver,
plus the volume quantiles `d10q`/`d50q`/`d90q` for mesh solvers.
"""
size_metrics(solution::CrystallisationSolution) =
    merge((; d10 = solution.d10, d32 = solution.d32, d43 = solution.d43, moment2 = solution.moment2),
          _population_size_metrics(solution.population))

_population_size_metrics(population::MeshPopulation) =
    (; d10q = population.d10q, d50q = population.d50q, d90q = population.d90q)
_population_size_metrics(population::AbstractPopulationTrajectory) = (;)

"""
    observable_values(sol::AbstractSolution, name::Symbol) -> AbstractArray

Return the simulated trajectory for a named observable. Built-in state and
size metrics are available automatically; users can extend this method for a
custom solution observable.
"""
function observable_values(sol::AbstractSolution, name::Symbol)
    named_values = merge(state_vars(sol), reactor_vars(sol), size_metrics(sol))
    hasproperty(named_values, name) ||
        throw(ArgumentError("No simulated observable named :$name for $(typeof(sol))."))
    return getproperty(named_values, name)
end

"""
    sol[name::Symbol]

Named trajectory of a solution, the same as `observable_values(sol, name)`:
`sol[:concentration]`, `sol[:d43]`, `sol[:d50q]`, `sol[:moments]` or a
configured solvent-state name.
"""
Base.getindex(sol::AbstractSolution, name::Symbol) = observable_values(sol, name)

"""
    successful_retcode(sol::AbstractSolution) -> Bool

Whether the solve behind `sol` succeeded, judged from `sol.retcode` as for any
SciML solution.
"""
SciMLBase.successful_retcode(sol::AbstractSolution) = SciMLBase.successful_retcode(sol.retcode)
