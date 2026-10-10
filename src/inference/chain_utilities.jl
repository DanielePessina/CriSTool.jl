"""
Conversion of MCMCChains.jl chains to the parameter-by-sample matrices used by
`ParameterSamples`.
"""

"""
    chains_to_matrix(chain::MCMCChains.Chains; burnin::Int=0, params=nothing) -> Matrix{Float64}

Convert an MCMCChains.Chains object to a parameter-by-sample matrix.

# Arguments
- `chain`: MCMCChains.Chains object from Turing.jl or similar
- `burnin`: Number of initial samples to discard (default: 0)
- `params`: Parameter names to extract (default: all non-internal parameters)

# Returns
- `Matrix{Float64}` with shape (n_params, n_samples) where samples from all chains are concatenated

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
