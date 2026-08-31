using Random

mutable struct _SequenceUnitRNG <: Random.AbstractRNG
    values::Vector{Float64}
    index::Int
end

function Random.rand(rng::_SequenceUnitRNG)
    value = rng.values[rng.index]
    rng.index += 1
    return value
end

@testset "Uncertainty Quantification Utilities" begin

    @testset "distribution_to_matrix" begin
        # Test with univariate distribution
        @testset "Univariate Distribution" begin
            dist_uni = Normal(0, 1)
            mat_uni = CriSTool.distribution_to_matrix(dist_uni, 100)
            @test size(mat_uni) == (1, 100)
            @test eltype(mat_uni) == Float64
        end

        # Test with multivariate distribution
        @testset "Multivariate Distribution" begin
            dist_multi = MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0])
            mat_multi = CriSTool.distribution_to_matrix(dist_multi, 100)
            @test size(mat_multi) == (2, 100)
            @test eltype(mat_multi) == Float64
        end

        # Test with product_distribution
        @testset "Product Distribution" begin
            prior = Distributions.product_distribution([Normal(0, 1), Uniform(-1, 1)])
            mat_prod = CriSTool.distribution_to_matrix(prior, 100)
            @test size(mat_prod) == (2, 100)
        end

        @testset "Product Distribution preserves component order" begin
            prior = Distributions.product_distribution([
                Distributions.Dirac(2.0), Distributions.Dirac(-1.0)
            ])
            mat = CriSTool.distribution_to_matrix(prior, 4;
                                                  rng = Random.Xoshiro(12))
            @test mat == [2.0 2.0 2.0 2.0;
                          -1.0 -1.0 -1.0 -1.0]
        end

        # Test with TriangularDist (common in parameter estimation)
        @testset "TriangularDist Prior" begin
            prior = Distributions.product_distribution([
                                                           TriangularDist(15.0, 65.0, 30.0),
                                                           TriangularDist(0.15, 2.5, 1.0)
                                                       ])
            mat = CriSTool.distribution_to_matrix(prior, 500)
            @test size(mat) == (2, 500)
            # Check bounds are respected
            @test all(mat[1, :] .>= 15.0)
            @test all(mat[1, :] .<= 65.0)
            @test all(mat[2, :] .>= 0.15)
            @test all(mat[2, :] .<= 2.5)
        end

        # Test error handling
        @testset "Error Handling" begin
            @test_throws ArgumentError CriSTool.distribution_to_matrix(Normal(0, 1), 0)
            @test_throws ArgumentError CriSTool.distribution_to_matrix(Normal(0, 1), -1)
        end
    end

    @testset "create_product_prior" begin
        @testset "Basic Functionality" begin
            dists = [
                TriangularDist(15, 65, 30),
                TriangularDist(0.15, 2.5, 1.0),
                TriangularDist(1e-4, 5, 1.0),
                TriangularDist(1.0, 3.5, 2.0)
            ]
            prior = CriSTool.create_product_prior(dists)
            @test prior isa Distributions.Product
            @test length(prior) == 4
        end

        @testset "Sampling from Product Prior" begin
            dists = [Normal(0, 1), Uniform(-1, 1)]
            prior = CriSTool.create_product_prior(dists)
            samples = rand(prior, 100)
            @test size(samples) == (2, 100)
        end

        @testset "Error Handling" begin
            @test_throws ArgumentError CriSTool.create_product_prior(Distributions.UnivariateDistribution[])
        end
    end

    @testset "prior_to_matrix" begin
        @testset "With Product Distribution" begin
            prior = Distributions.product_distribution([Normal(0, 1), Uniform(-1, 1)])
            mat = CriSTool.prior_to_matrix(prior, 100)
            @test size(mat) == (2, 100)
        end

        @testset "With Vector of Distributions" begin
            dists = [Normal(0, 1), Uniform(-1, 1)]
            mat = CriSTool.prior_to_matrix(dists, 100)
            @test size(mat) == (2, 100)
        end
    end

    @testset "Product prior helpers" begin
        # product_distribution replaces the deprecated Factored type
        prior = Distributions.product_distribution([Normal(0, 1), Uniform(-1, 1)])
        @test prior isa Distributions.Product
        x = [0.5, 0.5]
        @test Distributions.pdf(prior, x) > 0
        @test isfinite(Distributions.logpdf(prior, x))
        # prior_to_matrix draws (n_params, n_samples)
        mat = prior_to_matrix(prior, 50; rng = Random.Xoshiro(1))
        @test size(mat) == (2, 50)
        @test all(-1 .<= mat[2, :] .<= 1)
    end


    @testset "ABCDE_Turner Helpers" begin
        @testset "logsumexp_stable" begin
            # Basic functionality
            @test CriSTool.KissABC.logsumexp_stable([0.0, 0.0]) ≈ log(2.0)
            @test CriSTool.KissABC.logsumexp_stable([1.0, 2.0, 3.0]) ≈
                  log(exp(1.0) + exp(2.0) + exp(3.0))

            # Numerical stability with large negative values
            result = CriSTool.KissABC.logsumexp_stable([-1000.0, -1000.0])
            @test result ≈ -1000.0 + log(2.0)
            @test isfinite(result)

            # Single element
            @test CriSTool.KissABC.logsumexp_stable([5.0]) ≈ 5.0
        end

        @testset "log_kernel_weight" begin
            # Gaussian kernel
            @test CriSTool.KissABC.log_kernel_weight(1.0, 0.0, 1.0; kernel = :gaussian) ≈
                  -0.5
            @test CriSTool.KissABC.log_kernel_weight(0.0, 0.0, 1.0; kernel = :gaussian) ≈
                  0.0
            @test CriSTool.KissABC.log_kernel_weight(2.0, 0.0, 1.0; kernel = :gaussian) ≈
                  -2.0

            # Laplace kernel
            @test CriSTool.KissABC.log_kernel_weight(1.0, 0.0, 1.0; kernel = :laplace) ≈
                  -1.0
            @test CriSTool.KissABC.log_kernel_weight(0.0, 0.0, 1.0; kernel = :laplace) ≈ 0.0

            # None kernel (always returns 0)
            @test CriSTool.KissABC.log_kernel_weight(100.0, 0.0, 1.0; kernel = :none) ≈ 0.0
        end

        @testset "group_range" begin
            @test CriSTool.KissABC.group_range(1, 10) == 1:10
            @test CriSTool.KissABC.group_range(2, 10) == 11:20
            @test CriSTool.KissABC.group_range(3, 5) == 11:15
        end

        @testset "sample_from_logweights" begin
            # Supply known inverse-CDF draws so each branch is checked
            # deterministically rather than through arbitrary count bands.
            rng = _SequenceUnitRNG([0.05, 0.20, 0.90], 1)
            logws = log.([0.1, 0.3, 0.6])
            @test [CriSTool.KissABC.sample_from_logweights(rng, logws)
                   for _ in 1:3] == [1, 2, 3]
        end
    end

    @testset "ABCDE_Turner Basic" begin
        @testset "Toy Posterior Unimodal" begin
            # Simple 2D normal target
            prior = Distributions.product_distribution([Normal(0, 5), Normal(0, 5)])
            cost(x) = sum(abs2.(x .- [1.0, 1.0]))  # minimum at (1,1)
            res = CriSTool.ABCDE_Turner(prior, x -> cost(collect(x)), 0.5;
                                        nparticles = 64, generations = 30, K = 4,
                                        HPC = true, p_crossover = 0.9,
                                        rng = Random.Xoshiro(23))
            @test minimum(res.C.particles) < 1.0  # should find good solutions
            @test length(res.P) == 2
        end

        @testset "nparticles Adjustment" begin
            prior = Distributions.product_distribution([Normal(0, 1)])
            cost(x) = abs(x[1])

            # 100 particles with K=8 should adjust to 104 (13*8)
            res = @test_logs (:info, r"Adjusted nparticles") match_mode = :any CriSTool.ABCDE_Turner(prior,
                                                                                                     x -> cost(collect(x)),
                                                                                                     1.0;
                                                                                                     nparticles = 100,
                                                                                                     generations = 5,
                                                                                                     K = 8,
                                                                                                     HPC = true,
                                                                                                     rng = Random.Xoshiro(24))
            @test 0 < length(res.C.particles) <= 104
        end
    end
end
