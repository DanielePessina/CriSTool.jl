# CriSTool

CriSTool is a Julia package for crystallisation population-balance simulation,
kinetic parameter estimation and uncertainty propagation. It supports batch,
MSMPR and fed-batch operations with MoM, QMOM, DQMOM, finite-volume and WENO
solvers. All physical inputs use numeric SI units.

From a checkout, run `julia --project=. -e 'using Pkg; Pkg.instantiate()'`.
Julia 1.10–1.13 is supported. Runnable tutorials use a separate environment:
`julia --project=examples -e 'using Pkg; Pkg.instantiate()'`.

## Simulate your model

```julia
using CriSTool

crystal_model = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
    growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))

batch_run = RunConditions(temperature = 293.15, concentration = 18.0)
trajectory = simulate(crystal_model, batch_run; solver = MoM(), saveat = 0.0:3600.0:28800.0)
trajectory.concentration[end]
trajectory.d43[end]
```

A model is a material system plus kinetic laws bound to named values. Calling a
law with its parameter names binds them; `CNTNucleation()` alone is the unbound law.
`RunConditions` hold everything else about a run: temperature, initial
concentration, an optional seed and the reactor operation.

## Fit and predict

```julia
using Metaheuristics, Distributions
using Turing: NUTS   # Turing is optional; loading it enables BayesianSpec fitting

experiments = load_measurements("measurements.csv")  # each run starts from its time-zero data
fit_spec = OptimisationSpec(bounds = (growth = (growth_coefficient = (1e-12, 1e-9),),),
    algorithm = DE(N = 32))
fitted = fit(prepare_fit(crystal_model, experiments, fit_spec; solver = QMOM()))
fitted.parameters                        # (growth = (growth_coefficient = ...,),)

bayes_spec = BayesianSpec(priors = (growth = (growth_coefficient = Uniform(1e-12, 1e-9),),),
    n_samples = 500)
posterior = fit(prepare_fit(crystal_model, experiments, bayes_spec; solver = QMOM()))
predictions = predict(posterior)         # same runs, solver and settings as the fit
bands = prediction_summary(predictions, :concentration)
```

The specification selects which kinetic parameters vary; everything else stays
as bound in the model. Material properties, run conditions and supplied
measurement noise are never fitted. Likelihood fitting requires supplied
positive variances for scored measurements; MAE does not. Predictions describe
physical trajectories; adding measurement noise is a separate, explicit step
(`measurement_samples`).

Turing is optional: `fit` on a `BayesianSpec` needs `using Turing` (`using Turing: NUTS`
avoids a clash between Turing's and CriSTool's `predict`).

Plotting is a package extension: run `using CairoMakie` to enable
`plot_predictions` and `ChainStatsPlots`, and add `using PairPlots` for
`plot_posterior_pairplot`. `using CriSTool` alone does not load Makie.

Read the [development documentation](https://danielepessina.github.io/CriSTool.jl/dev/),
the [model workflow guide](docs/src/model-workflow.md), and the
[migration guide](docs/src/migration.md). `runsimulation` remains available as
the low-level interface for automatic differentiation and sensitivity work.

Run tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.
Build docs with `julia --project=docs docs/make.jl` after instantiating `docs`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the testing standard and the release checklist.

BSD-3-Clause; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
