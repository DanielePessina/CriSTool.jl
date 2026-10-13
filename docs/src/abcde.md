# ABC sampling

ABCDE uses the same model, experiment preparation and selected kinetic paths as
optimisation and Bayesian inference. Define explicit priors and an explicit
acceptance threshold for the chosen discrepancy:

```julia
using CriSTool, Distributions
abc_specification = ABCSpec(
    priors = (growth = (growth_coefficient = Uniform(1e-12, 1e-9),),),
    loss = MeanAbsoluteError(), target = 0.1, sampler = ABCDESampler(),
    n_particles = 128, n_generations = 100)
abc_prepared = prepare_fit(crystal_model, experiments, abc_specification; solver = MoM())
abc_posterior = fit(abc_prepared; parallel = EnsembleThreads())
abc_draws = abc_posterior.samples
abc_predictions = predict(abc_posterior)
```

`target` is required and must be a finite number. Thresholds have the
units/scale of the discrepancy and are a scientific choice; they are not
inferred from an optimum or from a confidence interval. `early_stop=true` stops
once the target is reached. `parallel = EnsembleThreads()` evaluates particles in parallel; the experiments
of each discrepancy evaluation then run serially.
`ABCDETurnerSampler` remains a distinct sampler option. Native posterior output
is retained as `backend_result`; joint draws use the common prediction interface.
The sampler implementation is vendored in `CriSTool.KissABC`.
