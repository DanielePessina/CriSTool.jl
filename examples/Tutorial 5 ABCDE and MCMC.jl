#=Tutorial 5: NUTS and ABCDE sample one estimation problem with named priors and joint prediction samples.
Sampling is intentionally small for a runnable example; increase counts for a study.
=#
using CriSTool, Distributions, Random
# Loading Turing enables sample(problem, NUTS(), n). Import only the names used.
using Turing: NUTS, MCMCThreads
# Loading CairoMakie and PairPlots enables CriSTool's plotting functions.
using CairoMakie, PairPlots

function main()
    crystal_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    measured_runs = load_measurements(joinpath(@__DIR__, "fake-experimental-dataset.csv"))
    # Priors select the sampled parameters; the default loss is the Gaussian likelihood.
    bayesian_problem = EstimationProblem(crystal_model, measured_runs;
        priors = (growth = (growth_order = Uniform(1.0, 4.0),),))

    # NUTS with supplied measurement variances; noise is never fitted.
    # `sample` takes the same arguments as Turing.sample: rng, problem, sampler, draws.
    posterior = sample(MersenneTwister(42), bayesian_problem, NUTS(20, 0.65), 20; progress = false)
    @assert successful_retcode(posterior)
    # The posterior predicts with the runs, solver and settings it was fitted with.
    posterior_predictions = predict(posterior)
    println(prediction_summary(posterior_predictions, :concentration))
    # Median and 5-95% band per run and observable, over the measured means.
    # Plotting functions return a figure; saving is the caller's choice.
    results_dir = mkpath(joinpath(@__DIR__, "results"))
    prediction_figure = plot_predictions(posterior_predictions, measured_runs;
        observables = (:concentration, :d43))
    save(joinpath(results_dir, "Tutorial 5 NUTS Predictions.png"), prediction_figure)
    # Posterior marginals over the problem's priors, labelled by named path.
    save(joinpath(results_dir, "Tutorial 5 NUTS Pairplot.png"), plot_posterior_pairplot(posterior))
    # Trace and density plots of the NUTS chain.
    save(joinpath(results_dir, "Tutorial 5 NUTS Chain.png"),
        plot_chain_diagnostics(posterior.original; title = "NUTS"))
    # Several chains on threads: sample(problem, NUTS(), MCMCThreads(), n_samples, n_chains).
    # sample(bayesian_problem, NUTS(20, 0.65), MCMCThreads(), 20, 4; progress = false)

    # ABC accepts draws whose discrepancy is below an explicit threshold; remake
    # swaps the likelihood for a mean absolute error discrepancy.
    abc_problem = remake(bayesian_problem; loss = MeanAbsoluteError())
    abc_posterior = sample(MersenneTwister(7), abc_problem, ABCDE(), 16; target = 10.0, n_generations = 2,
        parallel = EnsembleThreads())
    # MaxIters means the generations ran out before every particle reached the target.
    println("ABC retcode: ", abc_posterior.retcode)
    # Predict a new isothermal run from the same posterior draws.
    predict(abc_posterior, [RunConditions(temperature = 293.15, concentration = 18.0)];
        saveat = 0.0:600.0:21600.0)
end
main()
