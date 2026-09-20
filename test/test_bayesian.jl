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
using Metaheuristics
import OptimizationOptimJL
import Makie
using JLD2

@testset "Bayesian Inference" begin

    Random.seed!(11)
    nucl, gr, agg, br = nucl_CNT(), growth_empirical(), noaggregation(), nobreakage()
    solver = MoM()
    lf = logMLE(weighting = [1.0, 1.0])

    save_grid = collect(0.0:3600.0:18000.0)
    truth = ComponentVector(nucl = (ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
                            gr = (growth_coefficient = 1e-9 / 60, growth_order = 3.0),
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
                                          variance = fill((1e-6)^2, length(save_grid)))),
                                      temperature = 295.0, exp_id = 1)]

    prior = [TriangularDist(25.0, 50.0, 38.0), TriangularDist(0.0003, 0.001, 0.0006),
             TriangularDist(0.3e-9 / 60, 3e-9 / 60, 1e-9 / 60),
             TriangularDist(2.0, 4.0, 3.0)]

    @testset "kinetic_parameter_symbols" begin
        syms = CriSTool.kinetic_parameter_symbols(nucl, gr, agg, br)
        @test syms == [:ln_nucleation_prefactor, :surface_energy,
                       :growth_coefficient, :growth_order]
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
        syms = [:ln_nucleation_prefactor, :surface_energy,
                :growth_coefficient, :growth_order]
        rng = MersenneTwister(42)
        chain = Chains(rand(rng, 50, 4, 2), [:θ1, :θ2, :θ3, :θ4])
        renamed = CriSTool.rename_chain(chain, syms)
        @test names(renamed, :parameters) == syms
        @test_throws ArgumentError CriSTool.rename_chain(chain, syms[1:2])
    end

    @testset "PE_Routine returns a bounded minimum" begin
        Random.seed!(12)
        lower_bounds = [37.0, 0.5, 0.9, 2.5]
        upper_bounds = [39.0, 0.9, 1.1, 3.5]
        result = CriSTool.PE_Routine(
            lf, meas, lower_bounds, upper_bounds, nucl, gr, agg, br;
            solver = solver, nparticles = 4, generations = 1, savetxt = false,
            verbosity = 0, HPC = true, parallel_evaluation = false)
        minimizer = Metaheuristics.minimizer(result)
        loss_problem = CrystallisationProblem(;
            kinetics_nucleationfunction = nucl,
            kinetics_growthfunction = gr,
            kinetics_aggregationfunction = agg,
            kinetics_breakagefunction = br,
            solver = solver)
        @test length(minimizer) == length(lower_bounds)
        @test all(lower_bounds .<= minimizer .<= upper_bounds)
        @test Metaheuristics.minimum(result) ≈ loss(lf, loss_problem, minimizer, meas)
    end

    @testset "PE_Routine_Optimisation direct entry point" begin
        lower_bounds = [37.0, 0.5, 0.9, 2.5]
        upper_bounds = [39.0, 0.9, 1.1, 3.5]
        initial_parameters = [38.0, 0.6, 1.0, 3.0]
        result = CriSTool.PE_Routine_Optimisation(
            lf, meas, lower_bounds, upper_bounds, nucl, gr, agg, br;
            solver = solver,
            searchalgo = OptimizationOptimJL.Optim.Fminbox(
                OptimizationOptimJL.Optim.NelderMead()),
            x0 = initial_parameters,
            searchoptions = Dict(:f_reltol => 1.0),
            savetxt = false, verbosity = 0, HPC = true)
        @test length(result.u) == length(initial_parameters)
        @test all(isfinite, result.u)
        @test isfinite(result.objective)
    end

    @testset "ABCDE_Routine direct entry point" begin
        lower_bounds = [37.0, 0.5, 0.9, 2.5]
        upper_bounds = [39.0, 0.9, 1.1, 3.5]
        optimal_parameters = [38.0, 0.6, 1.0, 3.0]
        abc_prior = Distributions.product_distribution([
            Uniform(lower_bounds[1], upper_bounds[1]),
            Uniform(lower_bounds[2], upper_bounds[2]),
            Uniform(lower_bounds[3], upper_bounds[3]),
            Uniform(lower_bounds[4], upper_bounds[4]),
        ])
        result, metadata = CriSTool.ABCDE_Routine(
            lf, meas, optimal_parameters, abc_prior, nucl, gr, agg, br;
            solver = solver, nparticles = 8, generations = 1,
            confidenceinterval = 0.5, saveplot = false,
            verbosity = 0, HPC = true, test = :wilks)
        @test hasproperty(result, :P)
        @test metadata["res"] === result
        @test metadata["optimalparameters"] == optimal_parameters
        @test isfinite(metadata["target"])
    end

    @testset "MCMC_Routine" begin
        sampler = NUTS(5, 0.65; adtype = AutoForwardDiff(chunksize = 4))
        # No outputdir: zero filesystem writes.
        outdir = mktempdir()
        chain = cd(outdir) do
            CriSTool.MCMC_Routine(meas, prior, nucl, gr, agg, br;
                                  solver = solver, lossfunction = lf,
                                  sampler = sampler, n_samples = 10, n_chains = 1,
                                  verbosity = 0)
        end
        @test chain isa MCMCChains.Chains
        @test names(chain, :parameters) == [:ln_nucleation_prefactor, :surface_energy,
                                            :growth_coefficient, :growth_order]
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
        chain = Chains(rand(MersenneTwister(7), 30, 2, 1),
                       [:ln_nucleation_prefactor, :surface_energy])
        outdir = mktempdir()
        fig = CriSTool.ChainStatsPlots(chain; saveplot = true,
                                       savestring = "stats_test", savedir = outdir,
                                       showplot = false)
        @test fig isa Makie.Figure
        @test any(f -> endswith(f, "ChDensity.png"), readdir(outdir))
    end
end
