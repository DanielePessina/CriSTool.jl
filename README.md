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
problem = CrystallisationProblem(crystal_model, batch_run, (0.0, 28_800.0))
trajectory = solve(problem, MoM(); saveat = 3600.0)
successful_retcode(trajectory)
trajectory[:concentration][end]
trajectory[:d43][end]
```

A model is a material system plus kinetic laws bound to named values. Calling a
law with its parameter names binds them; `CNTNucleation()` alone is the unbound law.
`RunConditions` hold everything else about a run: temperature, initial
concentration, an optional seed and the reactor operation. As with an
`ODEProblem`, the problem holds no numerical method: the solver (`MoM()`,
`QMOM()`, `DQMOM()`, `FiniteVol()`, `WENO()`) is the second argument of `solve`,
and `remake(problem; conditions, tspan, p)` changes the run.

## Fit and predict

```julia
using Metaheuristics, Distributions
using Turing: NUTS   # Turing is optional; loading it enables solve(problem, NUTS())

experiments = load_measurements("measurements.csv")  # each run starts from its time-zero data
estimation = EstimationProblem(crystal_model, experiments; solver = QMOM(),
    bounds = (growth = (growth_coefficient = (1e-12, 1e-9),),),
    priors = (growth = (growth_coefficient = Uniform(1e-12, 1e-9),),))
fitted = solve(estimation, DE(N = 32))   # optimisers search inside the bounds
fitted.parameters                        # (growth = (growth_coefficient = ...,),)

posterior = solve(estimation, NUTS(); n_samples = 500)   # samplers draw under the priors
predictions = predict(posterior)         # same runs, solver and settings as the fit
bands = prediction_summary(predictions, :concentration)
```

The problem's bounds or priors select which kinetic parameters vary; everything else stays
as bound in the model. Material properties, run conditions and supplied
measurement noise are never fitted. Likelihood fitting requires supplied
positive variances for scored measurements; MAE does not. Predictions describe
physical trajectories; adding measurement noise is a separate, explicit step
(`measurement_samples`).

As with simulation, the problem holds no algorithm: `solve(problem, alg)`
takes a Metaheuristics algorithm, an Optimization.jl optimiser (`LBFGS()`),
`NUTS()` or `ABCDE()`, and `remake(problem; loss, solver, ...)` changes the
study. Turing is optional: `solve(problem, NUTS())` needs `using Turing`.

Plotting is a package extension: run `using CairoMakie` to enable
`plot_predictions` and `plot_chain_diagnostics`, and add `using PairPlots` for
`plot_posterior_pairplot`. `using CriSTool` alone does not load Makie.

Read the [development documentation](https://danielepessina.github.io/CriSTool.jl/dev/),
the [model workflow guide](docs/src/model-workflow.md), and the
[migration guide](docs/src/migration.md). `solve(remake(problem; p = values), solver)`
is the differentiable interface for automatic differentiation and sensitivity work.

Run tests with `julia --project=. -e 'using Pkg; Pkg.test()'`.
Build docs with `julia --project=docs docs/make.jl` after instantiating `docs`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the testing standard and the release checklist.

BSD-3-Clause; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
