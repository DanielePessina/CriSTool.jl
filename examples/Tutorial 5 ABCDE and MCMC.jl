"""
Tutorial 5: Posterior sampling with ABCDE and Turing NUTS.

Two complementary inference routines on the same synthetic dataset:
  1. ABCDE (likelihood-free) via the new `run_abc` entry point.
  2. NUTS (gradient MCMC) via Turing.
Same forward model, prior bounds, and loss function for both. Runs in
~1 minute on `julia --threads=4`; no external workbook needed.
"""

using CriSTool
using ComponentArrays
using Distributions, Random
using Turing
using Statistics

# Turing model lives at module scope so AD precompilation can specialise.
# Kinetics, solver, and loss are passed in (no closures over module-level state).
@model function nuts_model(data, lb, ub, nucl, gr, agg, br, solver, loss)
    Aj ~ TriangularDist(lb[1], ub[1], 0.5 * (lb[1] + ub[1]))
    γ  ~ TriangularDist(lb[2], ub[2], 0.5 * (lb[2] + ub[2]))
    Ag ~ TriangularDist(lb[3], ub[3], 0.5 * (lb[3] + ub[3]))
    g  ~ TriangularDist(lb[4], ub[4], 0.5 * (lb[4] + ub[4]))
    problem = CrystallisationProblem(; kinetics_nucleationfunction = nucl,
                                                   kinetics_growthfunction = gr, kinetics_aggregationfunction = agg,
                                                   kinetics_breakagefunction = br, solver = solver)
                                                   L = loss(loss, problem, [Aj, γ, Ag, g], [data])
    Turing.@addlogprob!(-L)
end

function main()
    Random.seed!(42)

    # Forward model.
    nucl, gr, agg, br = nucl_CNT(), growth_empirical(), noaggregation(), nobreakage()
    solver = MoM()
    loss   = logMLE(weighting = (1.0, 1.0))
    truth  = ComponentVector(nucl = (Aj = 38.0, γ = 0.6),
                             gr   = (Ag = 1.0, g = 3.0),
                             agg  = Float64[], br = Float64[])

    save_grid     = collect(0.0:30.0:300.0)
    T_K, loading  = 295.0, 0.0
    C0            = 18.0

    # Synthesise one repeat-measurement dataset with 5% multiplicative noise.
    _, ref = runsimulation(truth; nucl=nucl, gr=gr, agg=agg, br=br,
                            initial_concentration=C0, solver=solver,
                            save_idx=save_grid,
                            temp_profile=CriSTool.ConstantTemperature(T_K),
                            loading=loading)
    noisy_c = ref.concentration .* (1.0 .+ 0.05 .* randn(length(save_grid)))
    σ2      = (0.05 .* abs.(noisy_c) .+ 0.02) .^ 2
    meas    = CrystallisationExperiment(;
                                                          observables = (;
                                                          concentration = SeriesObservable(; time = save_grid, mean = noisy_c,
                                                          variance = σ2),
                                                          d43 = ScalarObservable(; value = ref.d43[end], variance = 0.1),
                                                          d50q = ScalarObservable(; value = ref.d43[end], variance = 0.1)),
                                                          temperature = T_K, loading = loading, exp_id = 1)

    lb = [25.0, 0.30, 0.30, 2.0]
    ub = [50.0, 1.00, 3.00, 4.0]

    # 1. ABCDE.
    println("ABCDE...")
    abc_prior = Factored([Uniform(lb[i], ub[i]) for i in eachindex(lb)]...)
    t = time()
    _, abc_meta = run_abc(loss, [meas], collect(truth), abc_prior, nucl, gr, agg, br;
                           solver = solver,
                           sampler = ABCDESampler(α = 0),
                           extrastring = "tutorial5_abc",
                           nparticles = 256, generations = 200,
                           confidenceinterval = 0.9,
                           saveplot = false, verbosity = 0)
    abc_means = abc_meta["meanparameters"]
    println("  $(round(time() - t, digits=1))s, mean = ", round.(abc_means, digits=3))

    # 2. NUTS — same forward model and loss, gradient-based MCMC over a
    #    triangular-prior likelihood-as-loss formulation.
    println("NUTS...")
    t = time()
    chain = sample(nuts_model(meas, lb, ub, nucl, gr, agg, br, solver, loss),
                    NUTS(50, 0.65; adtype = AutoForwardDiff(chunksize = 4)),
                    100; progress = false)
    nuts_means = [mean(chain[:Aj]), mean(chain[:γ]), mean(chain[:Ag]), mean(chain[:g])]
    println("  $(round(time() - t, digits=1))s, mean = ", round.(nuts_means, digits=3))

    # 3. Side-by-side comparison.
    println("\nparam │ truth │ ABCDE │ NUTS")
    for (i, name) in enumerate(("Aj", "γ", "Ag", "g"))
        println(rpad(name, 6), "│ ", rpad(round(truth[i], digits = 3), 6),
                "│ ", rpad(round(abc_means[i], digits = 3), 6),
                "│ ", round(nuts_means[i], digits = 3))
    end
end

main()
