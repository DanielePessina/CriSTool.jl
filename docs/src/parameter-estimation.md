# Kinetic parameter estimation

Construct a `CrystallisationModel` as in [Simulation](simulation.md), load or
construct experiments, and select free kinetic values by name:

```julia
using CriSTool, Metaheuristics
experiments = load_measurements("measurements.csv")
fit_specification = OptimisationSpec(bounds = (
    growth = (growth_coefficient = (1e-12, 1e-9), growth_order = (1.0, 4.0))),
    loss = logMLE())
prepared_fit = prepare_fit(crystal_model, experiments, fit_specification;
    solver = MoM(), solve_options = (reltol = 1e-7,))
fitted_result = fit(prepared_fit; algorithm = DE(N = 32,
    options = Options(iterations = 100, parallel_evaluation = false)))
fitted_result.model
fitted_result.parameters
fitted_result.objectives
fitted_result.backend_result
```

Only named selected kinetic parameters vary. All others retain their model
values. Bounds are required for bounded optimisation and current selected values
must be in bounds. The input model is not mutated. Native backend diagnostics
are retained. For an Optimization.jl algorithm, supply `algorithm`, `adtype` and
`searchoptions`; numerical options belong to preparation.

Transient experiments require initial concentration observations at time zero.
`initial_from` maps other required initial values and seeds; initialising samples
are excluded from scoring. Preparation does not fit initial conditions or silently
move the start to a later measurement. In steady mode, supply explicit
`relaxation_initial` and an autonomous MSMPR operation; steady observations are
targets, not initial conditions.

Gaussian `logMLE()` requires supplied positive variances for scored measurements.
No noise is fitted and no missing variance is filled automatically. `mae()` does
not require variances. Named weights associate factors with observables across
heterogeneous experiments. Custom observables can use `observable_projections`
at preparation without adding global methods.

## Bayesian inference

```julia
using Distributions, Turing
bayesian_spec = BayesianSpec(priors = (
    growth = (growth_coefficient = Uniform(1e-12, 1e-9),)))
bayesian_fit = prepare_fit(crystal_model, experiments, bayesian_spec; solver = MoM())
posterior = fit(bayesian_fit; sampler = NUTS(), n_samples = 1000, n_chains = 4)
joint_draws = parameter_samples(posterior)
```

Priors are explicit and never automatically centred on a fitted optimum.
`nuts_model(prepared)` exposes the Turing model for advanced sampling. Prepared
mutable state is private per sampling task. See [ABCDE](abcde.md) for
likelihood-free sampling and [Uncertainty](uq-ensembles.md) for predictions.

## Custom losses

Advanced users may implement `loss(custom_loss, setup, parameters)`.
`loss_terms` defaults to the total objective; extend it to expose named terms or
multiobjective accounting. A custom negative log likelihood must explicitly
implement `is_likelihood(custom_loss)=true` before use with `BayesianSpec`, and
owns validation of its supplied, fixed noise assumptions.
