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
    volumeweighteddensity(mesh, numberdensity, kv)

Compute volume-weighted density distribution.
"""
function volumeweighteddensity(mesh::AbstractVector, numberdensity::AbstractVecOrMat, kv)
    kv .* (numberdensity .* (mesh .^ 3.0))
end

"""
    surfaceweighteddensity(mesh, numberdensity, kv)

Compute surface-weighted density distribution.
"""
function surfaceweighteddensity(mesh::AbstractVector, numberdensity::AbstractVector, kv)
    kv .* (numberdensity .* mesh .^ 2.0)
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
        return cellcentre[1] * 1e6
    end

    step_size = cellcentre[2] - cellcentre[1]

    # Compute total integral first (avoid allocating full cumsum vector)
    total = zero(K)
    @inbounds for i in eachindex(voldensity)
        total += voldensity[i] * step_size
    end

    if total ≈ 0
        return cellcentre[1] * 1e6
    end

    # Find quantile by iterating through cumulative sum
    target = quantile * total
    cumulative = zero(K)
    @inbounds for i in eachindex(voldensity)
        prev_cumulative = cumulative
        cumulative += voldensity[i] * step_size

        if cumulative >= target
            if i == 1
                return cellcentre[1] * 1e6
            else
                # Linear interpolation
                x0, x1 = prev_cumulative / total, cumulative / total
                y0, y1 = cellcentre[i-1], cellcentre[i]
                return (y0 + (quantile - x0) * (y1 - y0) / (x1 - x0)) * 1e6
            end
        end
    end

    return cellcentre[end] * 1e6
end

"""
    quantilecalculator(cellcentre::AbstractVector{T}, voldensitymatrix::AbstractMatrix{K}, quantiles::W) -> Vector where {T<:Real, K<:Real, W<:Real}

Calculate particle size at a specified quantile for multiple time points.

# Arguments
- `cellcentre::AbstractVector{T}`: Particle size grid cell centres
- `voldensitymatrix::AbstractMatrix{K}`: Volume density matrix of size (N_grid, N_time)
- `quantiles::W`: Quantile value (0-1, e.g., 0.5 for median)

# Returns
- `Vector`: Particle sizes (in μm) at the specified quantile for each time point
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

Compute the four moment-derived size metrics (d10, d32, d43, μ₂) from a number-density
matrix and mesh. Used by FV solvers at solution-construction time so that the resulting
`CrystallisationFVSolution` carries these sizes as fields — the same way a
`CrystallisationMoMSolution` does. Callers should not need this directly.
"""
function _momentsizes(mesh::AbstractVector, numberdensity::AbstractMatrix)
    mu0 = momentcalculator(mesh, numberdensity, 0)
    mu1 = momentcalculator(mesh, numberdensity, 1)
    mu2 = momentcalculator(mesh, numberdensity, 2)
    mu3 = momentcalculator(mesh, numberdensity, 3)
    mu4 = momentcalculator(mesh, numberdensity, 4)
    return (d10 = 1e6 .* (mu1 ./ (mu0 .+ 1e-8)),
            d32 = 1e6 .* (mu3 ./ (mu2 .+ 1e-8)),
            d43 = 1e6 .* (mu4 ./ (mu3 .+ 1e-8)),
            mu2 = 1e6 .* mu2)
end

function getmomentsizes(::CrystallisationProblem,
                        solution::CrystallisationFVSolution)
    return (d10 = solution.d10, d32 = solution.d32, d43 = solution.d43, mu2 = solution.mu2)
end
"""
    getmomentsizes(problem::CrystallisationProblem, solution::CrystallisationMoMSolution) -> NamedTuple

Extract characteristic crystal sizes from a Method of Moments solution.

# Arguments
- `problem::CrystallisationProblem`: The crystallisation problem (unused for MoM but kept for dispatch)
- `solution::CrystallisationMoMSolution`: MoM simulation solution

# Returns
- `NamedTuple{(:d10, :d32, :d43, :mu2)}`: Named tuple with size metrics directly from solution
"""
function getmomentsizes(problem::CrystallisationProblem,
                        solution::CrystallisationMoMSolution)
    return (d10 = solution.d10, d32 = solution.d32, d43 = solution.d43, mu2 = solution.mu2)
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
