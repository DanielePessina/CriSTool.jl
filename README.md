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

crystal_model = CrystallisationModel(
    system = LysozymeSystem(),
    nucleation = KineticModel(nucl_CNT(); parameters = (
        ln_nucleation_prefactor = 38.0, surface_energy = 0.0006)),
    growth = KineticModel(growth_empirical(); parameters = (
        growth_coefficient = 1e-9 / 60, growth_order = 3.0)))

batch_run = CrystallisationProblem(crystal_model;
    initial_conditions = (concentration = 18.0,),
    temperature = ConstantTemperature(293.15), solver = MoM())
trajectory = simulate(batch_run; saveat = 0.0:3600.0:28800.0)
trajectory.success
trajectory.concentration[end]
trajectory.d43[end]

```

The model binds kinetic laws to named values and can be reused across experiments
and solvers. Simulation initial conditions are explicit. During transient fitting,
experiments supply initial observations at time zero; absent seed declarations
mean unseeded. A separate specification selects which kinetic parameters vary.
Material properties, initial conditions and supplied measurement noise are never fitted.

## Fit and predict

```julia
experiments = load_measurements("measurements.csv")
fit_spec = OptimisationSpec(bounds = (
    growth = (growth_coefficient = (1e-12, 1e-9),)), loss = logMLE())
prepared = prepare_fit(crystal_model, experiments, fit_spec; solver = MoM())
# using Metaheuristics
# fitted = fit(prepared; algorithm = Metaheuristics.DE(N = 32))

# Joint draws can come from NUTS, ABC, bootstrap fits, or user input.
draws = ParameterSamples(growth = (growth_coefficient = [1e-11, 2e-11],))
predictions = predict(crystal_model, experiments, draws; solver = MoM())
bands = prediction_summary(predictions, :concentration)
```

Likelihood fitting requires supplied positive variances for scored measurements.
MAE does not. Predictions describe physical trajectories by default; adding a
supplied measurement-noise distribution is an explicit separate operation.

Read the [development documentation](https://danielepessina.github.io/CriSTool.jl/dev/),
the [model workflow guide](docs/src/model-workflow.md), and the
[migration guide](docs/src/migration.md). Existing low-level solver interfaces
remain available for advanced numerical work; positional inference routines are
no longer the recommended interface.

Run tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.
Build docs with `julia --project=docs docs/make.jl` after instantiating `docs`.

BSD-3-Clause; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
