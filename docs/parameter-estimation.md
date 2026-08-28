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

# Load measurements (synthetic dataset shipped with the package)
path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.xlsx")
measurements = makerepeatmeasurements(path, "Unseeded_PE", [0.0])

# Model and bounds
nucl_f = nucl_CNT()
growth_f = growth_energy()
PE_lb = [10.0, 0.15, -10.0, 1.0]
PE_ub = [65.0, 2.5, 10.0, 3.5]

solver = MoM()
lossfn = logMLE(weighting = (1.0, 1.0))

# Metaheuristic search
optres = PE_Routine(lossfn, measurements, PE_lb, PE_ub,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = solver,
                    nparticles = 256,
                    generations = 128)

optimal_params = minimizer(optres)
```

## Loss functions

Built-in loss function types (see `Structs.jl` and `PELossFunctions.jl`):

- `logMLE`
- `logMLE_mo`
- `logMLE_Indiana`
- `logMLE_Han`
- `mae`

Each loss function is a struct subtype of `AbstractPELossFunction` and has
one or more methods of `parameterestimation_lossfunction`.

## Adding a new loss function

To add a new loss function:

1) Define a new struct in `Structs.jl` that subtypes `AbstractPELossFunction`.
2) Implement `parameterestimation_lossfunction` in `PELossFunctions.jl`.

Minimal example:

```julia
# Structs.jl
Base.@kwdef @concrete struct mse_loss <: AbstractPELossFunction
    string::String = "MSE"
end

# PELossFunctions.jl
function parameterestimation_lossfunction(::mse_loss,
                                          datasets::Vector{<:AbstractMeasurements},
                                          parameters,
                                          nucl, growth, agg, br,
                                          solver)
    total = 0.0
    for m in datasets
        _, sol = runsimulation(parameters;
                               nucl = nucl, gr = growth, agg = agg, br = br,
                               solver = solver,
                               initial_concentration = m.concentrationmean[1],
                               save_idx = m.time)
        total += sum((sol.concentration .- m.concentrationmean).^2)
    end
    return total
end
```

Use it just like the built-in loss functions:

```julia
lossfn = mse_loss()
optres = PE_Routine(lossfn, measurements, PE_lb, PE_ub,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = MoM())
```
