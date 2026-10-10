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

    @testset "parameter_samples preserves product component order" begin
        ordered_prior = Distributions.product_distribution([
            Distributions.Dirac(2.0), Distributions.Dirac(-1.0)])
        ordered_draws = parameter_samples(["nucleation.rate", "growth.speed"], ordered_prior;
            n_samples = 4, rng = Random.Xoshiro(12))
        @test ordered_draws.values == [2.0 2.0 2.0 2.0;
                                       -1.0 -1.0 -1.0 -1.0]
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
