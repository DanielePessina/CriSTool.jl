"""
Utilities for converting MCMC chain objects to matrices compatible with CriSTool.
Provides standardized interfaces for MCMCChains.jl and Distributions.jl integration.
"""

"""
    chains_to_matrix(chain::MCMCChains.Chains; burnin::Int=0, params=nothing) -> Matrix{Float64}

Convert an MCMCChains.Chains object to a matrix suitable for ensemble simulations.

# Arguments
- `chain`: MCMCChains.Chains object from Turing.jl or similar
- `burnin`: Number of initial samples to discard (default: 0)
- `params`: Parameter names to extract (default: all non-internal parameters)

# Returns
- `Matrix{Float64}` with shape (n_params, n_samples) where samples from all chains are concatenated

# Example
```julia
chain = sample(model, NUTS(), MCMCThreads(), 1000, 4)
samples = chains_to_matrix(chain, burnin=200)
run_ensemble(samples, measurements, ...)
```
"""
function chains_to_matrix(chain::MCMCChains.Chains;
                          burnin::Int = 0,
                          params::Union{Nothing, Vector{Symbol}} = nothing)
    # Validate burnin
    n_iterations = size(chain, 1)
    if burnin < 0
        throw(ArgumentError("burnin must be non-negative, got $burnin"))
    end
    if burnin >= n_iterations
        throw(ArgumentError("burnin ($burnin) must be less than chain length ($n_iterations)"))
    end

    # Get parameter names (exclude internal parameters like :lp)
    if params === nothing
        all_params = names(chain, :parameters)
        params = [p for p in all_params if !startswith(string(p), "lp")]
    end

    if isempty(params)
        throw(ArgumentError("No parameters found in chain after filtering"))
    end

    # Extract samples after burnin, concatenating all chains
    # Shape: (iterations - burnin) × n_params × n_chains
    chain_subset = chain[(burnin + 1):end, params, :]

    # Convert to 3D array
    arr = Array(chain_subset)

    # Handle cases where Array() drops singleton dimensions
    # MCMCChains can return 2D or 1D arrays when params=1 or chains=1
    if ndims(arr) == 2
        # Could be (iterations, chains) with 1 param OR (iterations, params) with 1 chain
        # We need to figure out which dimension was dropped
        n_iters = size(arr, 1)
        dim2_size = size(arr, 2)

        # Check if we have 1 parameter or 1 chain
        if length(params) == 1 && dim2_size == size(chain_subset, 3)
            # Array is (iterations, chains) - missing param dimension
            arr = reshape(arr, n_iters, 1, dim2_size)
        else
            # Array is (iterations, params) - missing chain dimension
            arr = reshape(arr, n_iters, dim2_size, 1)
        end
    elseif ndims(arr) == 1
        # Array is (iterations,) with 1 param and 1 chain
        n_iters = size(arr, 1)
        arr = reshape(arr, n_iters, 1, 1)
    end

    # Now arr is guaranteed to be 3D: (iterations, params, chains)
    n_iters = size(arr, 1)
    n_params = size(arr, 2)
    n_chains = size(arr, 3)
    n_samples = n_iters * n_chains

    # Reshape: concatenate chains, then transpose to (params, samples)
    # First reshape to (iterations * chains, params)
    reshaped = reshape(permutedims(arr, (1, 3, 2)), n_samples, n_params)

    # Transpose to get (params, samples)
    return Matrix{Float64}(permutedims(reshaped))
end

"""
    distribution_to_matrix(dist::Distribution, n_samples::Int; rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Sample from a distribution and return as a matrix compatible with ensemble simulations.

# Arguments
- `dist`: Any Distributions.jl distribution (univariate or multivariate)
- `n_samples`: Number of samples to draw
- `rng`: Random number generator (default: Random.GLOBAL_RNG)

# Returns
- `Matrix{Float64}` with shape (n_params, n_samples)

# Example
```julia
prior = product_distribution([Normal(0,1), Uniform(-1,1)])
samples = distribution_to_matrix(prior, 1000)
```
"""
function distribution_to_matrix(dist::Distributions.Distribution, n_samples::Int;
                                rng::AbstractRNG = Random.GLOBAL_RNG)
    if n_samples <= 0
        throw(ArgumentError("n_samples must be positive, got $n_samples"))
    end

    samples = rand(rng, dist, n_samples)

    # Ensure output is always (n_params, n_samples)
    if samples isa AbstractVector
        # Univariate distribution: samples is a vector of length n_samples
        return reshape(convert(Vector{Float64}, samples), 1, :)
    else
        # Multivariate distribution: samples is (n_params, n_samples) or similar
        return Matrix{Float64}(samples)
    end
end

"""
    create_product_prior(distributions::Vector{<:UnivariateDistribution}) -> Product

Create a multivariate prior from independent univariate distributions.
This is the recommended replacement for the deprecated `Factored` type.

# Arguments
- `distributions`: Vector of univariate distributions

# Returns
- A `Product` distribution from Distributions.jl

# Example
```julia
prior = create_product_prior([
    TriangularDist(15, 65, 30),
    TriangularDist(0.15, 2.5, 1.0),
    TriangularDist(1e-4, 5, 1.0),
    TriangularDist(1.0, 3.5, 2.0)
])

# Equivalent to:
# prior = product_distribution([...])
```
"""
function create_product_prior(distributions::Vector{<:Distributions.UnivariateDistribution})
    if isempty(distributions)
        throw(ArgumentError("distributions vector cannot be empty"))
    end
    return Distributions.product_distribution(distributions)
end

"""
    prior_to_matrix(prior, n_samples::Int; rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Convert any prior type (Factored, Product, or vector of distributions) to a sample matrix.
Handles backward compatibility with the deprecated `Factored` type.

# Arguments
- `prior`: Prior distribution (Factored, Product, Distribution, or Vector of UnivariateDistribution)
- `n_samples`: Number of samples to draw

# Returns
- `Matrix{Float64}` with shape (n_params, n_samples)
"""
function prior_to_matrix(prior::Distributions.Distribution, n_samples::Int;
                         rng::AbstractRNG = Random.GLOBAL_RNG)
    return distribution_to_matrix(prior, n_samples; rng = rng)
end

"""
    prior_to_matrix(priors::Vector{<:Distributions.UnivariateDistribution}, n_samples::Int; rng=Random.GLOBAL_RNG) -> Matrix{Float64}

Convert a vector of univariate distributions to a sample matrix.

# Arguments
- `priors::Vector{<:Distributions.UnivariateDistribution}`: Vector of independent prior distributions
- `n_samples::Int`: Number of samples to draw
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator

# Returns
- `Matrix{Float64}` with shape (n_params, n_samples)

Creates a product distribution from the vector and samples from it.
"""
function prior_to_matrix(priors::Vector{<:Distributions.UnivariateDistribution},
                         n_samples::Int;
                         rng::AbstractRNG = Random.GLOBAL_RNG)
    prod_dist = create_product_prior(priors)
    return distribution_to_matrix(prod_dist, n_samples; rng = rng)
end
