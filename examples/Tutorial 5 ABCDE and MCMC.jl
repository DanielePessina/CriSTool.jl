#=Tutorial 5: Bayesian and ABC specifications share named priors and joint prediction samples.
Sampling is intentionally small for a runnable example; increase counts for a study.
=#
using CriSTool, Distributions, Turing, Random

function main()
    crystal_model = CrystallisationModel(LysozymeSystem();
        nucleation = nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = growth_empirical(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    measured_runs = load_measurements(joinpath(@__DIR__, "fake-experimental-dataset.csv"))
    selected_priors = (growth = (growth_order = Uniform(1.0, 4.0),),)

    # NUTS with supplied measurement variances; noise is never fitted.
    bayesian_spec = BayesianSpec(priors = selected_priors, sampler = NUTS(20, 0.65), n_samples = 20)
    posterior = fit(prepare_fit(crystal_model, measured_runs, bayesian_spec);
        rng = MersenneTwister(42), progress = false)
    # The posterior predicts with the runs, solver and settings it was fitted with.
    posterior_predictions = predict(posterior)
    println(prediction_summary(posterior_predictions, :concentration))
    # Median and 5-95% band per run and observable, over the measured means.
    results_dir = joinpath(@__DIR__, "results")
    plot_predictions(posterior_predictions, measured_runs; observables = (:concentration, :d43),
        savename = "Tutorial 5 NUTS", savedir = results_dir, showplot = false)
    # Posterior marginals over the BayesianSpec priors, labelled by named path.
    plot_posterior_pairplot(posterior; savename = "Tutorial 5 NUTS", savedir = results_dir, showplot = false)

    # ABC accepts draws whose discrepancy is below an explicit threshold.
    abc_spec = ABCSpec(priors = selected_priors, target = 10.0, n_particles = 16, n_generations = 2)
    abc_posterior = fit(prepare_fit(crystal_model, measured_runs, abc_spec); threaded = true)
    # Predict a new isothermal run from the same posterior draws.
    predict(abc_posterior, [RunConditions(temperature = 293.15, concentration = 18.0)];
        saveat = 0.0:600.0:21600.0)
end
main()
