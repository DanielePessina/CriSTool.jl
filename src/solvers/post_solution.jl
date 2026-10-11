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
    shape_factor .* (numberdensity .* (mesh .^ 3.0))
end

"""
    surfaceweighteddensity(mesh, numberdensity, shape_factor)

Compute surface-weighted density distribution.
"""
function surfaceweighteddensity(mesh::AbstractVector, numberdensity::AbstractVector, shape_factor)
    shape_factor .* (numberdensity .* mesh .^ 2.0)
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

"""Size ratio `numerator / denominator`, zero for a numerically empty population."""
function _safe_moment_size_ratio(numerator, denominator, particle_count)
    isfinite(particle_count) ||
        throw(DomainError(particle_count,
                          "Particle count must be finite when computing a size metric."))
    if particle_count >= zero(particle_count) &&
       particle_count <= CRISTOOL_MOMENT_DENSITY_FLOOR
        return zero(promote_type(typeof(numerator), typeof(denominator)))
    end
    isfinite(numerator) && numerator >= zero(numerator) ||
        throw(DomainError(numerator,
                          "A nonempty population requires a finite nonnegative moment numerator."))
    isfinite(denominator) && denominator > zero(denominator) ||
        throw(DomainError(denominator,
                          "A nonempty population requires a finite positive moment denominator."))
    return numerator / denominator
end

"""
    _momentsizes(mesh, numberdensity)

Moment-derived size metrics (d10, d32, d43, moment2) of a mesh number-density
matrix, computed once when a mesh solution is assembled.
"""
function _momentsizes(mesh::AbstractVector, numberdensity::AbstractMatrix)
    step_size = Base.step(mesh)
    n_columns = size(numberdensity, 2)
    d10 = Vector{eltype(numberdensity)}(undef, n_columns)
    d32 = similar(d10)
    d43 = similar(d10)
    moment2 = similar(d10)

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

    density_type = eltype(numberdensity)
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

        moment2[column_index] = mu2
        d10[column_index] = _safe_moment_size_ratio(mu1, mu0, mu0)
        d32[column_index] = _safe_moment_size_ratio(mu3, mu2, mu0)
        d43[column_index] = _safe_moment_size_ratio(mu4, mu3, mu0)
    end
    return (; d10, d32, d43, moment2)
end

"""
    _raw_moment_size_metrics(moment_matrix)

Moment-ratio size metrics from a raw-moment matrix `(nmoments, nt)` whose row
`k + 1` holds µk. A ratio whose numerator moment is not tracked is `NaN`.
"""
function _raw_moment_size_metrics(moment_matrix::AbstractMatrix)
    tracked_moments, saved_count = size(moment_matrix)
    untracked_ratio = fill(convert(eltype(moment_matrix), NaN), saved_count)
    moment_ratio(numerator_row, denominator_row) = tracked_moments >= numerator_row ?
        [_safe_moment_size_ratio(moment_matrix[numerator_row, saved_index],
                                 moment_matrix[denominator_row, saved_index],
                                 moment_matrix[1, saved_index]) for saved_index in 1:saved_count] :
        copy(untracked_ratio)
    return (; d10 = moment_ratio(2, 1), d32 = moment_ratio(4, 3), d43 = moment_ratio(5, 4),
              moment2 = collect(view(moment_matrix, 3, :)))
end

"""Assemble a `CrystallisationSolution` from an ODE solution and its population."""
function _crystallisation_solution(problem::CrystallisationProblem, ode_solution, population,
                                   size_moments::NamedTuple, final_state, reactor_state::NamedTuple)
    solvent_trajectories = _solvent_solution_state(problem, ode_solution)
    return CrystallisationSolution(collect(ode_solution.t), solvent_trajectories.concentration,
                                   population, size_moments.d10, size_moments.d32,
                                   size_moments.d43, size_moments.moment2,
                                   solvent_trajectories, reactor_state, final_state,
                                   ode_solution.stats,
                                   SciMLBase.successful_retcode(ode_solution.retcode))
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
