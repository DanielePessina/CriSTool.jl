"""
Smoke tests for the Bayesian inference entry points: `nuts_model`,
`rename_chain`, `kinetic_parameter_symbols`, and `MCMC_Routine`.

The tests use a tiny NUTS run on synthetic data to keep the suite fast.
"""

using Test
using CriSTool
using Distributions
using Random
using Turing
using ComponentArrays
using MCMCChains
import Makie
using JLD2

@testset "Bayesian Inference" begin

    Random.seed!(11)
    nucl, gr, agg, br = nucl_CNT(), growth_empirical(), noaggregation(), nobreakage()
    solver = MoM()
    lf = logMLE(weighting = [1.0, 1.0])

    save_grid = collect(0.0:60.0:300.0)
    truth = ComponentVector(nucl = (Aj = 38.0, γ = 0.6),
                            gr = (Ag = 1.0, g = 3.0),
                            agg = Float64[], br = Float64[])
    _, ref = runsimulation(truth; nucl = nucl, gr = gr, agg = agg, br = br,
                           initial_concentration = 18.0, solver = solver,
                           save_idx = save_grid,
                           temp_profile = CriSTool.ConstantTemperature(295.0))
    noisy_c = ref.concentration .* (1.0 .+ 0.05 .* randn(length(save_grid)))
    σ2 = (0.05 .* abs.(noisy_c) .+ 0.02) .^ 2
    meas = [CrystallisationExperiment(;
                                      observables = (;
                                          concentration = Observable(; time = save_grid,
                                                                     mean = noisy_c,
                                                                     variance = σ2),
                                          d43 = Observable(; time = save_grid, mean = ref.d43,
                                                           variance = fill(0.1, length(save_grid)))),
                                      temperature = 295.0, exp_id = 1)]

    prior = [TriangularDist(25.0, 50.0, 38.0), TriangularDist(0.3, 1.0, 0.6),
             TriangularDist(0.3, 3.0, 1.0), TriangularDist(2.0, 4.0, 3.0)]

    @testset "kinetic_parameter_symbols" begin
        syms = CriSTool.kinetic_parameter_symbols(nucl, gr, agg, br)
        @test syms == [:Aⱼ, :γ, :Ag, :g]
        @test length(syms) == 4
    end

    @testset "nuts_model" begin
        model = CriSTool.nuts_model(meas, prior, nucl, gr, agg, br;
                                    solver = solver, lossfunction = lf)
        @test occursin("DynamicPPL.Model", string(typeof(model)))
        # Wrong prior length must be rejected up front.
        @test_throws ArgumentError CriSTool.nuts_model(meas, prior[1:2], nucl, gr, agg,
                                                       br; solver = solver,
                                                       lossfunction = lf)
    end

    @testset "rename_chain" begin
        syms = [:Aⱼ, :γ, :Ag, :g]
        rng = MersenneTwister(42)
        chain = Chains(rand(rng, 50, 4, 2), [:θ1, :θ2, :θ3, :θ4])
        renamed = CriSTool.rename_chain(chain, syms)
        @test names(renamed, :parameters) == syms
        @test_throws ArgumentError CriSTool.rename_chain(chain, syms[1:2])
    end

    @testset "MCMC_Routine" begin
        sampler = NUTS(5, 0.65; adtype = AutoForwardDiff(chunksize = 4))
        # No outputdir: zero filesystem writes.
        outdir = mktempdir()
        chain = CriSTool.MCMC_Routine(meas, prior, nucl, gr, agg, br;
                                      solver = solver, lossfunction = lf,
                                      sampler = sampler, n_samples = 10, n_chains = 1,
                                      verbosity = 0)
        @test chain isa MCMCChains.Chains
        @test names(chain, :parameters) == [:Aⱼ, :γ, :Ag, :g]
        @test size(chain.value, 1) == 10
        @test isempty(readdir(outdir))

        # Explicit outputdir: chain persisted, plots written, files created.
        chain2 = CriSTool.MCMC_Routine(meas, prior, nucl, gr, agg, br;
                                       solver = solver, lossfunction = lf,
                                       sampler = sampler, n_samples = 10, n_chains = 1,
                                       outputdir = outdir, showplot = false, verbosity = 0)
        files = readdir(outdir)
        @test !isempty(files)
        @test any(f -> endswith(f, ".jld2"), files)
        @test any(f -> endswith(f, "Pairplot.png"), files)
        @test any(f -> endswith(f, "ChDensity.png"), files)
        # Custom symbols kwarg.
        chain3 = CriSTool.MCMC_Routine(meas, prior, nucl, gr, agg, br;
                                       solver = solver, lossfunction = lf,
                                       sampler = sampler, n_samples = 10, n_chains = 1,
                                       symbols = [:a, :b, :c, :d], verbosity = 0)
        @test names(chain3, :parameters) == [:a, :b, :c, :d]

        # Exercise MCMCThreads with more than one chain. Each sampling task
        # must obtain its own prepared loss setup.
        chain4 = CriSTool.MCMC_Routine(meas, prior, nucl, gr, agg, br;
                                       solver = solver, lossfunction = lf,
                                       sampler = sampler, n_samples = 6, n_chains = 2,
                                       verbosity = 0)
        @test size(chain4.value, 1) == 6
        @test size(chain4.value, 3) == 2
    end

    @testset "ChainStatsPlots renders (Makie)" begin
        chain = Chains(rand(MersenneTwister(7), 30, 2, 1), [:Aⱼ, :γ])
        outdir = mktempdir()
        fig = CriSTool.ChainStatsPlots(chain; saveplot = true,
                                       savestring = "stats_test", savedir = outdir,
                                       showplot = false)
        @test fig isa Makie.Figure
        @test any(f -> endswith(f, "ChDensity.png"), readdir(outdir))
    end
end
