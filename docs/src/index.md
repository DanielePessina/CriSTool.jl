# CriSTool documentation

CriSTool models batch crystallisation as a population-balance problem. A
workflow usually combines four pieces:

1. kinetic models for nucleation, growth, aggregation, and breakage, with
   optional independent dissolution;
2. a parameter vector, either flat or named with `ComponentArrays`;
3. process conditions such as the temperature and saturation profiles; and
4. a numerical solver, such as `MoM()`, `QMOM(...)`, `DQMOM(...)`,
   `FiniteVol(...)`, or `WENO(...)`.

`runsimulation` combines those pieces and returns a problem plus a solution.
The same model/data objects are then reused by parameter estimation, ABCDE,
MCMC, sensitivity, and ensemble workflows.

## Installation

From a repository checkout, instantiate the package environment with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

After the `0.1.0` registry release, install CriSTool in another Julia
environment with:

```julia
using Pkg
Pkg.add("CriSTool")
```

For runnable tutorials, instantiate the example environment as well:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

```julia
using CriSTool

nucl, gr = nucl_CNT(), growth_empirical()
agg, br = noaggregation(), nobreakage()
parameters = [38.0, 0.0006, 1e-9 / 60, 3.0]
problem, solution = runsimulation(
    parameters;
    nucl = nucl,
    gr = gr,
    agg = agg,
    br = br,
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

if solution.success
    solution.concentration[end]
    solution.d43[end]
else
    error("simulation failed")
end
```

This example uses the compatibility order `[nucleation; growth; aggregation;
breakage]` because `diss` is omitted. With an independent dissolution model,
use the canonical `[nucleation; growth; dissolution; aggregation; breakage]`
order. Block lengths come from the selected kinetic models. For named access,
use `ComponentArray(parameters, paramaxis(nucl, gr, agg, br))`; see [Running
simulations](simulation.md).

## Find a workflow

| Question | Guide | Full example |
| --- | --- | --- |
| How do I run a simulation? | [Running simulations](simulation.md) | [Tutorial 1](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%201%20Running%20Simulations.jl) |
| Which solver should I use? | [Solvers](solvers.md) | [Tutorial 6](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%206%20Dissolution.jl), [Tutorial 7](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%207%20QMOM.jl) |
| How do I represent nucleation and growth? | [Kinetics](kinetics.md) | [Tutorial 1](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%201%20Running%20Simulations.jl) |
| How do I load experimental data? | [Measurements and data loading](measurements.md) | [Tutorial 2](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%202%20Parameter%20Estimation.jl) |
| How do I fit parameters? | [Parameter estimation](parameter-estimation.md) | [Tutorial 2](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%202%20Parameter%20Estimation.jl) |
| How do I use ABCDE or NUTS? | [ABCDE](abcde.md), [Parameter estimation](parameter-estimation.md) | [Tutorial 5](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%205%20ABCDE%20and%20MCMC.jl) |
| How do I change `T(t)` or solubility? | [Temperature profiles](temperature-profiles.md), [Saturation models](saturation-models.md) | [Tutorial 1](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%201%20Running%20Simulations.jl) |
| How do I add my own kinetic or observable? | [Kinetics](kinetics.md), [Bringing your own system](bring-your-own-system.md) | [Tutorial 4](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%204%20Defining%20a%20Custom%20Kinetic.jl) |
| How do I model dissolution? | [Kinetics](kinetics.md), [Solvers](solvers.md) | [Tutorial 6](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%206%20Dissolution.jl) |
| How do I run sensitivity studies? | [Sensitivity analysis](sensitivity.md) | [Tutorial 3](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%203%20Sensitivity%20Analysis.jl) |
| How do I run ensemble uncertainty studies? | [Ensembles and uncertainty](uq-ensembles.md) | — |

## Guides

- [Running simulations](simulation.md)
- [Solvers](solvers.md)
- [Kinetics: nucleation, growth, aggregation, and breakage](kinetics.md)
- [Measurements and data loading](measurements.md)
- [Temperature profiles](temperature-profiles.md)
- [Saturation models and supersaturation](saturation-models.md)
- [Parameter estimation](parameter-estimation.md)
- [Optimisation routines](optimisation.md)
- [ABCDE routine](abcde.md)
- [Sensitivity analysis](sensitivity.md)
- [Ensembles and uncertainty](uq-ensembles.md)
- [Bringing your own system](bring-your-own-system.md)

## API reference

Use the [API reference](api.md) for exported types, constructors, solver
outputs, and entry-point signatures.

## Tutorials

The examples are the full runnable references for common workflows. Start at
[Tutorials](tutorials.md) for the dependency requirements and a short
description of each script.
