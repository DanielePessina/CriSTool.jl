# CriSTool documentation

CriSTool models batch crystallisation as a population-balance problem. A
workflow usually combines four pieces:

1. kinetic models for nucleation, growth, aggregation, and breakage;
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

For runnable tutorials, instantiate the example environment as well:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

## Quick start

```julia
using CriSTool

parameters = [38.0, 0.0006, 1e-9 / 60, 3.0]
problem, solution = runsimulation(
    parameters;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

solution.success
solution.concentration[end]
solution.d43[end]
```

The parameter order is `[nucleation; growth; aggregation; breakage]`. The
parameter block lengths come from the selected kinetic models. For named
access, use `ComponentArray(parameters,
paramaxis(nucl, gr, agg, br))`; see [Running simulations](simulation.md).

## Find a workflow

| Question | Guide | Full example |
| --- | --- | --- |
| How do I run a simulation? | [Running simulations](simulation.md) | [Tutorial 1](<../examples/Tutorial 1 Running Simulations.jl>) |
| Which solver should I use? | [Solvers](solvers.md) | [Tutorial 6](<../examples/Tutorial 6 Dissolution.jl>), [Tutorial 7](<../examples/Tutorial 7 QMOM.jl>) |
| How do I represent nucleation and growth? | [Kinetics](kinetics.md) | [Tutorial 1](<../examples/Tutorial 1 Running Simulations.jl>) |
| How do I load experimental data? | [Measurements and data loading](measurements.md) | [Tutorial 2](<../examples/Tutorial 2 Parameter Estimation.jl>) |
| How do I fit parameters? | [Parameter estimation](parameter-estimation.md) | [Tutorial 2](<../examples/Tutorial 2 Parameter Estimation.jl>) |
| How do I use ABCDE or NUTS? | [ABCDE](abcde.md), [Parameter estimation](parameter-estimation.md) | [Tutorial 5](<../examples/Tutorial 5 ABCDE and MCMC.jl>) |
| How do I change `T(t)` or solubility? | [Temperature profiles](temperature-profiles.md), [Saturation models](saturation-models.md) | [Tutorial 1](<../examples/Tutorial 1 Running Simulations.jl>) |
| How do I add my own kinetic or observable? | [Kinetics](kinetics.md), [Bringing your own system](bring-your-own-system.md) | [Tutorial 4](<../examples/Tutorial 4 Defining a Custom Kinetic.jl>) |
| How do I model dissolution? | [Kinetics](kinetics.md), [Solvers](solvers.md) | [Tutorial 6](<../examples/Tutorial 6 Dissolution.jl>) |
| How do I run sensitivity studies? | [Sensitivity analysis](sensitivity.md) | [Tutorial 3](<../examples/Tutorial 3 Sensitivity Analysis.jl>) |
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

## Tutorials

The examples are the full runnable references for common workflows. Start at
[Tutorials](tutorials.md) for the dependency requirements and a short
description of each script.
