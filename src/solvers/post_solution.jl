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

"""
    _momentsizes(mesh, numberdensity)

Compute the four moment-derived size metrics (d10, d32, d43, moment2) from a number-density
matrix and mesh. Used by FV solvers at solution-construction time so that the resulting
`CrystallisationFVSolution` carries these sizes as fields — the same way a
`CrystallisationMoMSolution` does. Callers should not need this directly.
"""
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

function getmomentsizes(::CrystallisationProblem,
                        solution::CrystallisationFVSolution)
    return (d10 = solution.d10, d32 = solution.d32, d43 = solution.d43, moment2 = solution.moment2)
end
"""
    getmomentsizes(problem::CrystallisationProblem, solution::CrystallisationMoMSolution) -> NamedTuple

Extract characteristic crystal sizes from a Method of Moments solution.

# Arguments
- `problem::CrystallisationProblem`: The crystallisation problem (unused for MoM but kept for dispatch)
- `solution::CrystallisationMoMSolution`: MoM simulation solution

# Returns
- `NamedTuple{(:d10, :d32, :d43, :moment2)}`: Named tuple with size metrics directly from solution
"""
function getmomentsizes(problem::CrystallisationProblem,
                        solution::CrystallisationMoMSolution)
    return (d10 = solution.d10, d32 = solution.d32, d43 = solution.d43, moment2 = solution.moment2)
end

"""
    bisectionfinder(f::Function, a::Number, b::Number;
                   tol::AbstractFloat=1e-10, maxiter::Integer=200)

    Find the quantile with a bisection search of the cumulative density function.
"""

"""
    get_characteristic_size(sol::AbstractSolution)

Returns the main characteristic particle size from a simulation solution.
- For `CrystallisationMoMSolution`, this is the final `d43`.
- For `CrystallisationFVSolution`, this is the final `d50q`.
"""
get_characteristic_size(sol::CrystallisationMoMSolution) = sol.d43[end]
get_characteristic_size(sol::CrystallisationFVSolution) = sol.d50q[end]

"""
    time(sol::AbstractSolution) -> AbstractVector

Time grid of a solution.
"""
time(sol::AbstractSolution) = sol.time


"""
    _size_trajectory(sol::AbstractSolution) -> AbstractVector

Characteristic size trajectory: `d43` for MoM, `d50q` for discretised
solutions (solver-type dispatch).
"""
_size_trajectory(sol::CrystallisationMoMSolution) = sol.d43
_size_trajectory(sol::CrystallisationFVSolution) = sol.d50q

"""
    state_vars(sol::CrystallisationMoMSolution) -> NamedTuple

Named solvent-state variables of a MoM solution, including `concentration`
and any additional coupled solvent variables.
"""
state_vars(sol::CrystallisationMoMSolution) = merge(sol.solvent_state, reactor_vars(sol))

"""
    state_vars(sol::CrystallisationFVSolution) -> NamedTuple

Named solvent-state variables of a discretised solution, plus `numberdensity`
 (mesh × time) and `voldensity` (mesh × time).
"""
state_vars(sol::CrystallisationFVSolution) = merge(sol.solvent_state, reactor_vars(sol),
                                                   (; numberdensity = sol.numberdensity,
                                                      voldensity = sol.voldensity))

"""
    size_metrics(sol::CrystallisationMoMSolution) -> NamedTuple

Particle-size metrics of a MoM solution: `d10`, `d32`, `d43`, `moment2`.
"""
size_metrics(sol::CrystallisationMoMSolution) =
    (; d10 = sol.d10, d32 = sol.d32, d43 = sol.d43, moment2 = sol.moment2)

"""
    size_metrics(sol::CrystallisationFVSolution) -> NamedTuple

Particle-size metrics of a discretised solution: quantiles `d10q`/`d50q`/`d90q`
plus `d10`/`d32`/`d43`.
"""
size_metrics(sol::CrystallisationFVSolution) =
    (; d10q = sol.d10q, d50q = sol.d50q, d90q = sol.d90q,
      d10 = sol.d10, d32 = sol.d32, d43 = sol.d43)

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
