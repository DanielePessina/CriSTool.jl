# Parameter estimation

CriSTool provides two main entry points:

- `PE_Routine`: metaheuristic optimization (Metaheuristics.jl).
- `PE_Routine_Optimisation`: Optimization.jl-based algorithms (BBO, LBFGS, etc).

Both routines minimize a loss function built from experimental measurements.

For a fully worked end-to-end example (PE + ABCDE + NUTS) see
`CriSTool/examples/Tutorial 2 Parameter Estimation.jl` and
`Tutorial 5 ABCDE and MCMC.jl`.

## Basic workflow

```julia
using CriSTool
using Distributions

# Load experiments (synthetic dataset shipped with the package)
path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.xlsx")
experiments = load_experiments(path, "Unseeded_PE", 0.0)

# Model and bounds
nucl_f = nucl_CNT()
growth_f = growth_energy()
PE_lb = [10.0, 0.15, -10.0, 1.0]
PE_ub = [65.0, 2.5, 10.0, 3.5]

solver = MoM()
lossfn = logMLE(weighting = (1.0, 1.0))

# Metaheuristic search
optres = PE_Routine(lossfn, experiments, PE_lb, PE_ub,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = solver,
                    nparticles = 256,
                    generations = 128)

optimal_params = minimizer(optres)
```

## Loss functions

Built-in loss function types (see `Structs.jl` and `PELossFunctions.jl`):

- `logMLE`: weighted negative log-likelihood (concentration trajectory +
  final particle size)
- `mae`: weighted mean absolute error

Each loss function is a struct subtype of `AbstractPELossFunction` and has
methods of `loss`:

```julia
problem = CrystallisationProblem(; kinetics_nucleationfunction = nucl_f,
                                 kinetics_growthfunction = growth_f,
                                 solver = solver)
L = loss(lossfn, problem, optimal_params, experiments)
```

The `problem` carries kinetics and solver; per-experiment conditions
(temperature, loading, initial concentration) are read from each
`CrystallisationExperiment` inside the loss.

## Adding a new loss function

To add a new loss function:

1) Define a new struct in `Structs.jl` that subtypes `AbstractPELossFunction`.
2) Implement `loss` for it in `PELossFunctions.jl`.

Minimal example:

```julia
# Structs.jl
Base.@kwdef @concrete struct mse_loss <: AbstractPELossFunction
    string::String = "MSE"
end

# PELossFunctions.jl
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