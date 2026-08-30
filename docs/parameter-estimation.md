# Parameter estimation

CriSTool provides two main entry points:

- `PE_Routine`: metaheuristic optimization (Metaheuristics.jl).
- `PE_Routine_Optimisation`: Optimization.jl-based algorithms (BBO, LBFGS, etc).

Both routines minimize a loss function built from experimental measurements.

For a fully worked end-to-end example (PE + ABCDE + NUTS) see
[Tutorial 2 — Parameter Estimation](<../examples/Tutorial 2 Parameter Estimation.jl>)
and [Tutorial 5 — ABCDE and MCMC](<../examples/Tutorial 5 ABCDE and MCMC.jl>).

## Basic workflow

```julia
using CriSTool
using Distributions
using Metaheuristics

# Load experiments (synthetic dataset shipped with the package)
path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.xlsx")
experiments = load_experiments(path, "Unseeded_PE", 0.0)

# Model and bounds
nucl_f = nucl_CNT()
growth_f = growth_energy()
PE_lb = [10.0, 0.15, -10.0, 1.0]
PE_ub = [65.0, 2.5, 10.0, 3.5]

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

Weights follow the observable field order. The default `[1.0, 1.0]` retains
the concentration/size convention; additional observables receive weight 1.0
unless explicit weights are supplied. Moment solvers (`MoM` and `QMOM`) use
the measured `d43` observable; discretised solvers use `d50q` when both legacy
size fields are present. `logMLE` uses measured variances by
default, with `RelativeVariance(percent)` available for relative-error data:

```julia
lossfn = logMLE(weighting = [1.0, 0.5, 1.0],
                variance_model = RelativeVariance(5.0))
```

The `problem` carries kinetics and solver; per-experiment conditions
(temperature, loading, initial concentration) are read from each
`CrystallisationExperiment` inside the loss.

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
                               loading = expt.loading)
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
with `rename_chain`; `kinetic_parameter_symbols` infers the names (`:Aⱼ`,
`:γ`, `:Ag`, `:g`, …) from the kinetics' own symbols.

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
