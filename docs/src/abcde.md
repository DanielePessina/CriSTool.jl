# ABC sampling

ABCDE uses the same model, experiment preparation and selected kinetic paths as
optimisation and Bayesian inference. Define explicit priors and an explicit
acceptance threshold for the chosen discrepancy:

```julia
using CriSTool, Distributions
abc_specification = ABCSpec(
    priors = (growth = (growth_coefficient = Uniform(1e-12, 1e-9),)),
    loss = mae(), target = 0.1)
abc_prepared = prepare_fit(crystal_model, experiments, abc_specification; solver = MoM())
abc_posterior = fit(abc_prepared; sampler = ABCDESampler(),
    nparticles = 128, generations = 100)
abc_draws = parameter_samples(abc_posterior)
```

Thresholds have the units/scale of the discrepancy and are a scientific choice.
They are not inferred from an optimum or silently assigned by an F-test.
`ABCDETurnerSampler` remains a distinct sampler option. Native posterior output
is retained as `backend_result`; joint draws use the common prediction interface.
The sampler implementation is vendored in `CriSTool.KissABC`.
