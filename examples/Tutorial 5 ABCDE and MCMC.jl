#=Tutorial 5: separate Bayesian/ABC specifications, common joint prediction samples.
Sampling is intentionally small for a runnable example; increase counts for a study.
=#
using CriSTool, Distributions, Turing, Random
include(joinpath(@__DIR__, "model_helpers.jl"))

function main()
    crystal_model = tutorial_model()
    measured_runs = load_measurements(joinpath(@__DIR__, "fake-experimental-dataset.csv"))
    selected_priors = (growth = (growth_order = Uniform(1.0, 4.0),),)
    bayesian_preparation = prepare_fit(crystal_model, measured_runs,
        BayesianSpec(priors = selected_priors); solver = MoM())
    posterior = fit(bayesian_preparation; sampler = NUTS(20, 0.65),
        n_samples = 20, n_chains = 1, rng = MersenneTwister(42), progress = false)
    posterior_predictions = predict(crystal_model, measured_runs, parameter_samples(posterior))
    println(prediction_summary(posterior_predictions, :concentration))
    # ABC uses an explicit threshold on a discrepancy, without a fitted noise model.
    abc_preparation = prepare_fit(crystal_model, measured_runs,
        ABCSpec(priors = selected_priors, loss = mae(), target = 10.0); solver = MoM())
    abc_posterior = fit(abc_preparation; nparticles = 16, generations = 2, HPC = true)
    predict(crystal_model, measured_runs, parameter_samples(abc_posterior))
end
main()
