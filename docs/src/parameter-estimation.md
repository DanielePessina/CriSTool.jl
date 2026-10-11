# Kinetic parameter estimation

Construct a `CrystallisationModel` as in [Simulation](simulation.md), load or
construct experiments, and select free kinetic values by name:

```julia
using CriSTool, Metaheuristics
experiments = load_measurements("measurements.csv")
fit_specification = OptimisationSpec(bounds = (
    growth = (growth_coefficient = (1e-12, 1e-9), growth_order = (1.0, 4.0)),),
    algorithm = DE(N = 32, options = Options(iterations = 100, seed = 42)),
    loss = GaussianNLL())
prepared_fit = prepare_fit(crystal_model, experiments, fit_specification;
    solver = MoM(), solve_options = (reltol = 1e-7,))
fitted_result = fit(prepared_fit; threaded = true)
fitted_result.model
fitted_result.parameters
fitted_result.objectives
fitted_result.backend_result
```

Only named selected kinetic parameters vary. All others retain their model
values. Bounds are required for bounded optimisation and current selected values
must be in bounds. The input model is not mutated. Native backend diagnostics
are retained. The algorithm belongs to the specification; for an Optimization.jl
algorithm also set `adtype` and `searchoptions` there. Numerical solve options
belong to preparation. `fit(prepared; threaded=true)` evaluates Metaheuristics
populations in parallel, each candidate on a private copy of the solve templates.
`problem_options` passes extra numerical problem settings to every run.

### Threads across experiments

Each loss evaluation solves one ODE per experiment. With Julia started with
several threads (`julia -t 4`), `parallel = EnsembleThreads()` (re-exported from
SciMLBase) solves those experiments concurrently, for the loss and for its
ForwardDiff gradient alike:

```julia
prepared_fit = prepare_fit(crystal_model, experiments, fit_specification;
    solver = MoM(), parallel = EnsembleThreads())
fitted_result = fit(prepared_fit)                          # uses the prepared setting
fitted_result = fit(prepared_fit; parallel = EnsembleSerial())  # override for one call
```

The default is `EnsembleSerial()`. Scores are reduced in experiment order, so
threaded and serial losses, gradients and fits are bitwise identical. The
speed-up is bounded by the number of experiments and by the slowest one, and it
composes with `threaded = true` (population or chain parallelism) through
Julia's shared thread pool: prefer `parallel` when there are few candidates or
chains and many experiments, and `threaded` otherwise. Any other ensemble type
raises an `ArgumentError`.

Transient experiments read their initial values from time-zero observations
through `FromObservation` placeholders in their `RunConditions`; by default only
the concentration. Initialising samples are excluded from scoring. Preparation
does not fit initial conditions or silently move the start to a later
measurement. The default `mode = TransientMode()` integrates each run over its
observation times. With `mode = SteadyMode(SteadyStateOptions(...))`, each
experiment's conditions must be numeric (no placeholders) and use an autonomous
MSMPR operation; the run relaxes from those values, and steady observations are
targets, not initial conditions. Callable kinetic laws require
`SteadyMode(SteadyStateOptions(autonomous = true))` (or `SteadyMode(autonomous = true)`),
an explicit declaration that their rate does not depend on time. The package
cannot infer that property from arbitrary user code.

Gaussian `GaussianNLL()` requires supplied positive variances for scored measurements.
No noise is fitted and no missing variance is filled automatically. `MeanAbsoluteError()` does
not require variances. Named weights associate factors with observables across
heterogeneous experiments; they are resolved and checked once by `prepare_fit`. Custom observables can use `observable_projections`
at preparation without adding global methods.

## Bayesian inference

```julia
using Distributions, Turing
bayesian_spec = BayesianSpec(priors = (
    growth = (growth_coefficient = Uniform(1e-12, 1e-9),),),
    sampler = NUTS(), n_samples = 1000, n_chains = 4)
bayesian_fit = prepare_fit(crystal_model, experiments, bayesian_spec; solver = MoM())
posterior = fit(bayesian_fit; threaded = true, progress = false)
joint_draws = posterior.samples
posterior_predictions = predict(posterior)
```

Priors are explicit and never automatically centred on a fitted optimum.
`threaded` runs chains in parallel (`MCMCThreads`) and `parallel = EnsembleThreads()`
solves the experiments inside each log-density evaluation on threads; other keywords such as `rng` and `progress`
go to `Turing.sample`. Prepared mutable state is private per sampling task. The
native chain is kept in `posterior.backend_result`. See [ABCDE](abcde.md) for
likelihood-free sampling and [Uncertainty](uq-ensembles.md) for predictions.

## Custom losses

Advanced users may subtype `AbstractPELossFunction` and implement
`loss(custom_loss, setup, parameters)`, where `setup` is the internal
`CriSTool.LossSetup` built by `prepare_fit` and `parameters` is the full kinetic
vector. `loss(prepared_fit, values)` and `loss_terms(prepared_fit, values)` call
these methods with the selected values expanded. `loss_terms` defaults to the
total objective; extend `loss_terms(custom_loss, setup, parameters)` to expose
named terms or multiobjective accounting. A custom negative log likelihood must explicitly
implement `is_likelihood(custom_loss)=true` before use with `BayesianSpec`, and
owns validation of its supplied, fixed noise assumptions.
