# Parameter estimation

CriSTool provides two main entry points:

- `PE_Routine`: metaheuristic optimization (Metaheuristics.jl).
- `PE_Routine_Optimisation`: Optimization.jl-based algorithms (BBO, LBFGS, etc).

Both routines minimize a loss function built from experimental measurements.

For a fully worked end-to-end example (PE + ABCDE + NUTS) see
[Tutorial 2 — Parameter Estimation](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%202%20Parameter%20Estimation.jl)
and [Tutorial 5 — ABCDE and MCMC](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%205%20ABCDE%20and%20MCMC.jl).

## Basic workflow

```julia
using CriSTool
using Distributions
using Metaheuristics

# Load experiments (synthetic dataset shipped with the package)
path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.csv")
experiments = load_measurements(path)

# Model and bounds
nucl_f = nucl_CNT()
growth_f = growth_energy()
PE_lb = [10.0, 0.00015, -20.0, 1.0]
PE_ub = [65.0, 0.0025, 0.0, 3.5]

solver = MoM()
lossfn = logMLE(weighting = [1.0, 1.0])

# Metaheuristic search
optres = PE_Routine(lossfn, experiments, PE_lb, PE_ub,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = solver,
                    nparticles = 256,
                    generations = 128)

optimal_params = minimizer(optres)
```

`PE_Routine` uses the supplied lower and upper bounds in the same flat
parameter order used by `runsimulation`. The returned result is a
Metaheuristics.jl optimisation result; `minimizer(optres)` extracts the
parameter vector used by later ABCDE and MCMC steps.

## Loss functions

Built-in loss function types (see `src/core/loss_types.jl` and
`src/inference/parameter_estimation.jl`):

- `logMLE`: weighted Gaussian negative log-likelihood over the active
  observables
- `mae`: weighted mean absolute error over the active observables

Each loss function is a struct subtype of `AbstractPELossFunction` and has
methods of `loss`:

```julia
problem = CrystallisationProblem(; kinetics_nucleationfunction = nucl_f,
                                 kinetics_growthfunction = growth_f,
                                 solver = solver)
L = loss(lossfn, problem, optimal_params, experiments)
```

Named weights associate each factor with its observable even when experiments
use different observable subsets or field orders:

```julia
lossfn = mae(weighting = (; concentration = 1.0, d43 = 0.5))
setup = prepare_loss(problem, experiments)
objectives = batchLF_procMO(lossfn, setup, optimal_params)
setup.observable_names # labels of the multiobjective result
```

For homogeneous schemas, vector weights retain the first experiment's field
order; unspecified factors default to 1.0. With heterogeneous schemas the global
names are sorted. Nonuniform vector weights then require an explicit
`observable_order` in `prepare_loss`; named weights are simpler. Scalar MAE sums
per-observable mean errors pooled over included points across all experiments.
Its multiobjective result contains those same terms. `logMLE` sums Gaussian NLL
terms and uses measured variances by default; `RelativeVariance(percent)` selects
a relative-error model.

`prepare_loss(problem, experiments)` applies measured experiment temperature,
initial concentration and declared initial crystals. Only the concentration point
at the integration start is excluded from scoring because it supplies C0. A late
first concentration sample cannot supply C0 when another observable begins earlier.
`prepare_loss(configured_problems, experiments)` instead preserves each configured
problem and scores all targets, including the first concentration sample. Use
`exclude_initial_concentration` to record an intentional selection policy.

Unsuccessful numerical solves receive a failure penalty. Invalid parameter sizes,
unknown observables and exceptions in user kinetics or custom observables propagate;
they cannot silently become a plausible objective value.

## Adding a new loss function

To add a new loss function:

1) Define a new struct in `src/core/loss_types.jl` that subtypes
   `AbstractPELossFunction`.
2) Implement `loss` for it in `src/inference/parameter_estimation.jl`.

Minimal example:

```julia
# src/core/loss_types.jl
Base.@kwdef @concrete struct mse_loss <: AbstractPELossFunction
    string::String = "MSE"
end

# src/inference/parameter_estimation.jl
function loss(::mse_loss, problem::CrystallisationProblem,
              parameters, experiments::Vector{<:AbstractExperiment})
    total = 0.0
    for expt in experiments
        _, sol = runsimulation(parameters;
                               nucl = problem.kinetics_nucleationfunction,
                               gr = problem.kinetics_growthfunction,
                               agg = problem.kinetics_aggregationfunction,
                               br = problem.kinetics_breakagefunction,
                               solver = problem.solver,
                               initial_concentration = initial_concentration(expt),
                               save_idx = expt.observables.concentration.time,
                               temp_profile = ConstantTemperature(expt.temperature),
                               initial_crystals = expt.initial_crystals)
        conc = expt.observables.concentration
        total += sum((sol.concentration .- conc.mean).^2)
    end
    return total
end
```

Use it just like the built-in loss functions:

```julia
lossfn = mse_loss()
optres = PE_Routine(lossfn, experiments, PE_lb, PE_ub,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = MoM())
```

## MCMC posterior sampling (Turing NUTS)

`nuts_model` builds a ready-to-sample Turing model from the same loss used
by `PE_Routine`/`run_abc`. You provide one prior distribution per parameter
(any `Distributions` distribution — triangular on the MLE is the usual
choice):

```julia
prior = [TriangularDist(PE_lb[i], PE_ub[i], optimal_params[i])
         for i in eachindex(optimal_params)]

model = nuts_model(experiments, prior, nucl_f, growth_f,
                   noaggregation(), nobreakage();
                   solver = solver, lossfunction = lossfn)

chain = Turing.sample(model, NUTS(1000, 0.65; adtype = AutoForwardDiff(chunksize = 4)),
                      1000; progress = false)
chain = rename_chain(chain, kinetic_parameter_symbols(nucl_f, growth_f,
                                                      noaggregation(), nobreakage()))
```

Chain parameters are sampled as `θ[1]`, `θ[2]`, … and renamed afterwards
with `rename_chain`; `kinetic_parameter_symbols` infers descriptive ASCII names
from each kinetic model's `symbols` declaration.

`MCMC_Routine` wraps the whole flow (model build + sampling + rename +
persistence) and mirrors `run_abc`:

```julia
chain = MCMC_Routine(experiments, prior, nucl_f, growth_f,
                     noaggregation(), nobreakage();
                     solver = solver, lossfunction = lossfn,
                     sampler = NUTS(1000, 0.65; adtype = AutoForwardDiff(chunksize = 4)),
                     n_samples = 1000, n_chains = 4,
                     outputdir = "mcmc_results")  # nothing = no file writes
```

`MCMC_Routine` returns the named chain; with `outputdir` set it persists the
chain (`.jld2`) and writes posterior pair, trace/density and
measurements-vs-ensemble plots.

When a problem template has an explicit `initial_state`, loss preparation retains
its initial crystal population and other solvent initial values. It applies each
experiment's measured initial concentration and constant temperature. Explicit
`initial_crystals` on an experiment replace the template population. Preparation
creates a new problem and leaves the template unchanged.

## Simulation options in prepared losses

Direct simulation and prepared loss evaluation use the same solver tolerances
and iteration limits. QMOM uses an absolute tolerance for each raw moment,
scaled to its physical dimensions; `tolerance_mode = :auto` preserves those
floors and adapts each component separately. Each evaluation creates fresh
callback state, including when parameter candidates run concurrently.

Keep simulation settings separate from optimizer options:

```julia
setup = prepare_loss(problem, experiments;
    solve_options = (; reltol = 1e-7, maxiters = 100_000, maxtime = 30.0))
objective_value = loss(lossfn, setup, optimal_params)
```

An optional `algorithm` keyword overrides the selected SciML time-stepper.
The default has no special prepared-loss wall-clock limit. Set `maxtime` or
`maxiters` explicitly when an inference workflow needs a candidate budget.

For a custom SciML callback, pass `callback_factory = odeproblem -> callback`.
The factory runs once per solve and should construct fresh mutable state.
Its callback composes with the package domain, extinction and CFL callbacks.
Use `save_positions = (false, false)` when callback events should not add
measurement predictions. The callback must preserve the population and solvent
state layout. Errors raised by the factory propagate to the caller.

`solve_options` accepts a named tuple of SciML options. State, parameter,
integration-span and output-selection overrides (`u0`, `p`, `tspan`, `saveat`,
`save_idxs`, and saving controls) are reserved by the prepared observation
contract. Raw `callback` and `merge_callbacks` overrides are also reserved;
use the factory to retain package safety callbacks.
