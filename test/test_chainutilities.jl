"""
Unit tests for chain utilities in CriSTool.

Tests the chains_to_matrix function and related utilities for converting
MCMC chains to parameter matrices compatible with ensemble simulations.
"""

using Test
using CriSTool
using MCMCChains
using Distributions
using Random
using JLD2
using LinearAlgebra
using Turing

@testset "Chain Utilities" begin

    # Get the path to the test chain file
    test_chain_path = joinpath(@__DIR__, "test_chain.jld2")

    @testset "chains_to_matrix - Normal case" begin
        # Use hand-labelled values so this checks parameter/iteration/chain
        # ordering, not only the returned dimensions.
        n_iters = 3
        n_params = 2
        n_chains = 2
        data = reshape(collect(1.0:12.0), n_iters, n_params, n_chains)
        param_names = [:first_parameter, :second_parameter]

        chain = Chains(data, param_names)

        # Test conversion
        result = chains_to_matrix(chain, burnin = 0)

        @test size(result) == (n_params, n_iters * n_chains)
        @test result == [1.0 2.0 3.0 7.0 8.0 9.0;
                         4.0 5.0 6.0 10.0 11.0 12.0]
        @test chains_to_matrix(chain; params = [:second_parameter]) ==
              [4.0 5.0 6.0 10.0 11.0 12.0]
    end

    @testset "chains_to_matrix - Single parameter" begin
        # Test with 1 parameter, 2 chains
        rng = MersenneTwister(42)
        n_iters = 100
        n_params = 1
        n_chains = 2

        data = rand(rng, n_iters, n_params, n_chains)
        param_names = [:θ]

        chain = Chains(data, param_names)

        # This should work after the fix
        result = chains_to_matrix(chain, burnin = 0)

        @test size(result) == (n_params, n_iters * n_chains)
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "chains_to_matrix - Single chain" begin
        # Test with 4 params, 1 chain
        rng = MersenneTwister(42)
        n_iters = 100
        n_params = 4
        n_chains = 1

        data = rand(rng, n_iters, n_params, n_chains)
        param_names = [:Aⱼ, :γ, :Ag, :g]

        chain = Chains(data, param_names)

        # This should work after the fix
        result = chains_to_matrix(chain, burnin = 0)

        @test size(result) == (n_params, n_iters * n_chains)
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "chains_to_matrix - Single iteration after burnin" begin
        # Test with extreme burnin (only 1 iteration remaining)
        rng = MersenneTwister(42)
        n_iters = 100
        n_params = 4
        n_chains = 2

        data = rand(rng, n_iters, n_params, n_chains)
        param_names = [:Aⱼ, :γ, :Ag, :g]

        chain = Chains(data, param_names)

        # Burnin = n_iters - 1, leaving only 1 iteration
        burnin = n_iters - 1
        result = chains_to_matrix(chain, burnin = burnin)

        @test size(result) == (n_params, (n_iters - burnin) * n_chains)
        @test size(result, 2) == n_chains  # Only 1 iteration per chain
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "chains_to_matrix - Burnin handling" begin
        # Test various burnin values
        rng = MersenneTwister(42)
        n_iters = 100
        n_params = 4
        n_chains = 2

        data = rand(rng, n_iters, n_params, n_chains)
        param_names = [:Aⱼ, :γ, :Ag, :g]

        chain = Chains(data, param_names)

        # Test with different burnin values
        for burnin in [0, 10, 50, 99]
            result = chains_to_matrix(chain, burnin = burnin)
            expected_samples = (n_iters - burnin) * n_chains

            @test size(result) == (n_params, expected_samples)
            @test eltype(result) == Float64
        end
    end

    @testset "chains_to_matrix - Parameter filtering" begin
        # Test filtering of internal parameters (like :lp)
        n_iters = 3
        n_chains = 2

        # Create data with internal parameters
        data = reshape(collect(1.0:30.0), n_iters, 5, n_chains)
        param_names = [:Aⱼ, :γ, :Ag, :g, :lp]

        chain = Chains(data, param_names,
                      Dict(:internals => [:lp]))

        # Test that :lp is filtered out
        result = chains_to_matrix(chain, burnin = 0)

        # Should only have 4 parameters (lp filtered out), in the original
        # parameter order and with both chains concatenated.
        @test result == [1.0 2.0 3.0 16.0 17.0 18.0;
                         4.0 5.0 6.0 19.0 20.0 21.0;
                         7.0 8.0 9.0 22.0 23.0 24.0;
                         10.0 11.0 12.0 25.0 26.0 27.0]
        @test size(result, 2) == n_iters * n_chains
    end

    @testset "chains_to_matrix - Edge case: 1 param, 1 chain, 1 iteration" begin
        # Most extreme edge case
        rng = MersenneTwister(42)
        n_iters = 10
        n_params = 1
        n_chains = 1

        data = rand(rng, n_iters, n_params, n_chains)
        param_names = [:θ]

        chain = Chains(data, param_names)

        # Test with burnin leaving only 1 iteration
        result = chains_to_matrix(chain, burnin = n_iters - 1)

        @test size(result) == (1, 1)  # 1 param, 1 iteration
        @test eltype(result) == Float64
        @test isfinite(result[1])
    end

    @testset "chains_to_matrix - Error handling" begin
        # Test error handling for invalid inputs
        rng = MersenneTwister(42)
        n_iters = 100
        n_params = 4
        n_chains = 2

        data = rand(rng, n_iters, n_params, n_chains)
        param_names = [:Aⱼ, :γ, :Ag, :g]

        chain = Chains(data, param_names)

        # Test negative burnin
        @test_throws ArgumentError chains_to_matrix(chain, burnin = -1)

        # Test burnin >= chain length
        @test_throws ArgumentError chains_to_matrix(chain, burnin = n_iters)
        @test_throws ArgumentError chains_to_matrix(chain, burnin = n_iters + 10)
    end

    @testset "distribution_to_matrix - Univariate distribution" begin
        # Test with univariate distribution
        dist = Normal(0, 1)
        n_samples = 1000

        result = distribution_to_matrix(dist, n_samples)

        @test size(result) == (1, n_samples)
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "distribution_to_matrix - Multivariate distribution" begin
        # Test with multivariate distribution
        dist = MvNormal(zeros(4), I)
        n_samples = 1000

        result = distribution_to_matrix(dist, n_samples)

        @test size(result) == (4, n_samples)
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "distribution_to_matrix - Product distribution" begin
        # Test with product of univariate distributions
        dists = [Normal(0, 1), Uniform(-1, 1), TriangularDist(0, 2, 1)]
        prod_dist = product_distribution(dists)
        n_samples = 1000

        result = distribution_to_matrix(prod_dist, n_samples)

        @test size(result) == (3, n_samples)
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "create_product_prior" begin
        # Test creation of product priors
        priors = [
            TriangularDist(15, 65, 30),
            TriangularDist(0.15, 2.5, 1.0),
            TriangularDist(1e-4, 5, 1.0),
            TriangularDist(1.0, 3.5, 2.0)
        ]

        prod_prior = create_product_prior(priors)

        @test prod_prior isa Distributions.Product
        @test length(prod_prior) == 4
    end

    @testset "create_product_prior - Empty vector error" begin
        # Test error on empty vector - need to type it correctly
        @test_throws ArgumentError create_product_prior(UnivariateDistribution[])
    end

    @testset "prior_to_matrix - Distribution" begin
        # Test prior_to_matrix with Distribution
        prior = product_distribution([
            Normal(0, 1),
            Uniform(-1, 1),
            TriangularDist(0, 2, 1)
        ])
        n_samples = 1000

        result = prior_to_matrix(prior, n_samples)

        @test size(result) == (3, n_samples)
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "prior_to_matrix - Vector of distributions" begin
        # Test prior_to_matrix with vector of distributions
        priors = [
            TriangularDist(15, 65, 30),
            TriangularDist(0.15, 2.5, 1.0),
            TriangularDist(1e-4, 5, 1.0),
            TriangularDist(1.0, 3.5, 2.0)
        ]
        n_samples = 1000

        result = prior_to_matrix(priors, n_samples)

        @test size(result) == (4, n_samples)
        @test eltype(result) == Float64
        @test all(isfinite.(result))
    end

    @testset "distribution_to_matrix - Error handling" begin
        # Test error handling for invalid inputs
        dist = Normal(0, 1)

        @test_throws ArgumentError distribution_to_matrix(dist, 0)
        @test_throws ArgumentError distribution_to_matrix(dist, -100)
    end

    @testset "chains_to_matrix - Real MCMC chain from file" begin
        # Test with a real MCMC chain generated from the diagnostic script
        # This tests the actual use case from UQ-Methods-Comparison.jl

        if !isfile(test_chain_path)
            @test_skip "Test chain file not found at $test_chain_path"
        else
            # Load the real chain
            data = load(test_chain_path)
            chain = data["chain"]
            optimal_params = data["optimalparameters"]

            # Test basic conversion with no burnin
            result = chains_to_matrix(chain, burnin = 0)

            n_iterations = size(chain, 1)
            n_chains = size(chain, 3)
            expected_samples = n_iterations * n_chains

            @test size(result, 1) == 4  # 4 parameters: Aⱼ, γ, Ag, g
            @test size(result, 2) == expected_samples
            @test eltype(result) == Float64
            @test all(isfinite.(result))

            # Test with burnin (same as in UQ-Methods-Comparison)
            burnin = 20
            result_burned = chains_to_matrix(chain, burnin = burnin)

            expected_burned = (n_iterations - burnin) * n_chains
            @test size(result_burned, 1) == 4
            @test size(result_burned, 2) == expected_burned
            @test eltype(result_burned) == Float64
            @test all(isfinite.(result_burned))

            # Test that burned result has fewer samples
            @test size(result_burned, 2) < size(result, 2)
        end
    end  # end @testset

end  # end Chain Utilities
