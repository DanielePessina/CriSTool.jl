# Kinetic parameter estimation

Construct a `CrystallisationModel` as in [Simulation](simulation.md), load or
construct experiments, and select free kinetic values by name:

```julia
using CriSTool
using Metaheuristics: DE, Options   # Metaheuristics also exports an unrelated `sample`
experiments = load_measurements("measurements.csv")
estimation_problem = EstimationProblem(crystal_model, experiments;
    bounds = (growth = (growth_coefficient = (1e-12, 1e-9), growth_order = (1.0, 4.0)),),
    loss = GaussianNLL(), solver = MoM(), solve_options = (reltol = 1e-7,))
fitted_result = solve(estimation_problem, DE(N = 32, options = Options(iterations = 100, seed = 42));
    parallel = EnsembleThreads())
successful_retcode(fitted_result)   # check the retcode before using the values
fitted_result.u                      # selected values as a flat vector, in selection order
fitted_result.objective              # total loss at u
fitted_result.parameters             # the same values, named like the bounds
fitted_result.objectives             # named observable objectives summing to objective
fitted_result.model                  # the model holding the fitted values
fitted_result.original               # the optimiser's own result
```

`EstimationProblem` follows the SciML problem/solve pattern: the problem holds
the model, the experiments, the selected parameters, the loss and the
simulation settings. `solve(problem, optimiser)` runs an optimiser and returns a
`FitResult` shaped like a SciML optimisation solution (`u`, `objective`,
`retcode`, `original`); `sample(problem, sampler, n)` draws from the posterior
and returns an `InferenceResult`.
Construction resolves every run and builds its solve templates once, so later
`loss` evaluations and solves reuse them. `remake(problem; loss = MeanAbsoluteError())`
builds the problem again with the named keywords replaced.

Only named selected kinetic parameters vary. All others retain their model
values. Optimisers need bounds and the current selected values must lie inside
them; samplers need priors. A problem may hold both, naming the same parameters
in the same order, so one problem can be optimised and then sampled. The input
model is not mutated. Native backend diagnostics are retained. The ODE settings
`alg`, `solve_options` and `callback_factory` belong to the problem, as they do
for `solve(problem::CrystallisationProblem, solver)`. `problem_options` passes
extra `CrystallisationProblem` settings to every run.

The algorithm decides the call, what it returns and which keywords it takes:

| Algorithm | Selection | Call | Result | Keywords |
| --- | --- | --- | --- | --- |
| Metaheuristics (`DE`, `PSO`, `ECA`, `NSGA2`, ...) | `bounds` | `solve(problem, alg)` | `FitResult` (NSGA2: one per Pareto candidate) | `parallel` |
| Optimization.jl optimiser (`LBFGS()` from OptimizationOptimJL, ...) | `bounds` | `solve(problem, alg)` | `FitResult` | `adtype = AutoForwardDiff()`, `parallel`, and Optimization.jl options such as `maxiters` |
| Turing sampler (`NUTS()`, needs `using Turing`) | `priors` | `sample([rng], problem, alg, n_samples)` or `sample([rng], problem, alg, MCMCThreads(), n_samples, n_chains)` | `InferenceResult` | `parallel`, and `Turing.sample` options such as `progress` |
| `ABCDE()`, `ABCDETurner()` | `priors` | `sample([rng], problem, alg, n_particles)` | `InferenceResult` | `target` (required), `n_generations = 128`, `stopping = AllGenerations()`, `parallel` |

`sample` is the `StatsBase`/`AbstractMCMC` function that Turing also exports.
`Metaheuristics` exports a different `sample`, so with both loaded import the
Metaheuristics names you use (`using Metaheuristics: DE, Options`) or write
`using CriSTool: sample`. A Metaheuristics fit succeeds when its best objective
is finite; an Optimization.jl fit takes the optimiser's retcode.

A Metaheuristics algorithm runs on a private copy, so its options, status and
RNG are not consumed and the same instance can be solved again. Results record
the algorithm in `result.alg` and the problem in `result.estimation_problem`.

### Threads

`parallel = EnsembleThreads()` (re-exported from SciMLBase) is the
parallelism keyword of `solve` and `sample`. Start Julia with several threads (`julia -t 4`).
An estimation has two levels that could run on threads: the candidates (a
Metaheuristics population, ABC particles or MCMC chains) and the experiments
solved inside one loss evaluation. Threading both only adds scheduling overhead
once the outer level fills the threads, so the algorithm decides which level
receives `parallel`; the other runs serially:

| Algorithm | Level that receives `parallel` |
| --- | --- |
| Metaheuristics (`DE`, `PSO`, `SA`, `NSGA2`) | population candidates, each on a private copy of the solve templates |
| Optimization.jl optimiser (`LBFGS`, ...) | experiments of each loss and ForwardDiff gradient evaluation |
| Turing sampler | experiments of each log-density evaluation; chains are threaded by passing `MCMCThreads()`, and then `parallel` must stay serial |
| `ABCDE()`, `ABCDETurner()` | particles, each on a private copy of the solve templates |

The default is `EnsembleSerial()`, which runs every level serially. Scores are
reduced in experiment order, so threaded and serial losses, gradients and fits
are bitwise identical. Experiment-level speed-up is bounded by the number of
experiments and by the slowest one. Any other ensemble type raises an
`ArgumentError`.

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
heterogeneous experiments; they are resolved and checked once by `EstimationProblem`. Custom observables can use the
`observable_projections` keyword of `EstimationProblem` without adding global methods.

## Bayesian inference

```julia
using Distributions
using Turing: NUTS, MCMCThreads   # loading Turing enables sample(problem, NUTS(), n)
bayesian_problem = EstimationProblem(crystal_model, experiments;
    priors = (growth = (growth_coefficient = Uniform(1e-12, 1e-9),),), solver = MoM())
posterior = sample(bayesian_problem, NUTS(), MCMCThreads(), 1000, 4; progress = false)
joint_draws = posterior.samples
posterior_predictions = predict(posterior)
```

Turing is a weak dependency: `sample(problem, NUTS(), n)` is defined by the
`CriSToolTuringExt` extension, which `using Turing` (or `import Turing`) loads.
Any Turing sampler works in place of `NUTS()`. CriSTool's `predict` extends the
same `StatsAPI.predict` that Turing exports, so `using CriSTool, Turing` works
without a name clash. ABC sampling and everything else do not need Turing.

Posterior sampling needs a likelihood loss (`GaussianNLL()`, the default) and
uses the supplied measurement variances; noise is never fitted. Priors are
explicit and never automatically centred on a fitted optimum. `MCMCThreads()`
runs the chains in parallel; with one chain, `parallel = EnsembleThreads()`
solves the experiments inside each log-density evaluation on threads. Prepared
mutable state is private per sampling task. The native chain is kept in
`posterior.original`. See [ABCDE](abcde.md) for likelihood-free
sampling and [Uncertainty](uq-ensembles.md) for predictions.

## Custom losses

Advanced users may subtype `AbstractPELossFunction` and implement
`loss(custom_loss, setup, parameters)`, where `setup` is the internal
`CriSTool.LossSetup` built by `EstimationProblem` and `parameters` is the full kinetic
vector. `loss(estimation_problem, values)` and `loss_terms(estimation_problem, values)` call
these methods with the selected values expanded. `loss_terms` defaults to the
total objective; extend `loss_terms(custom_loss, setup, parameters)` to expose
named terms or multiobjective accounting. A custom negative log likelihood must explicitly
implement `is_likelihood(custom_loss)=true` before posterior sampling with a Turing sampler, and
owns validation of its supplied, fixed noise assumptions.
