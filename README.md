# CriSTool

CriSTool is a Julia package for simulating batch crystallisation, fitting
kinetic parameters to measurements, and propagating parameter uncertainty
through population-balance models.

The package provides:

- population-balance solvers based on the Method of Moments (MoM), the
  Quadrature Method of Moments (QMOM), finite volumes, and WENO;
- nucleation, growth, aggregation, breakage, and signed dissolution kinetics;
- measurement ingestion from Excel workbooks and typed experiment containers;
- solver-aware initial crystal states from mass, d43, and distribution characteristics;
- parameter estimation with Metaheuristics.jl and Optimization.jl;
- likelihood-free ABCDE and Turing NUTS workflows;
- sensitivity analysis, ensemble simulation, and Makie plotting utilities.

## Installation

CriSTool declares Julia compatibility `^1.12` in `Project.toml`. From a
repository checkout, instantiate the package environment with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The runnable examples use a separate environment because some tutorials add
`GlobalSensitivity`, `QuasiMonteCarlo`, and other tutorial-only dependencies:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

## Quick start: one simulation

`runsimulation` takes kinetic models, a parameter vector, and a solver. The
flat parameter vector is ordered as `[nucleation; growth; aggregation;
breakage]`; no-op aggregation and breakage models contribute empty blocks.

```julia
using CriSTool

nucl = nucl_CNT()
gr   = growth_empirical()

parameters = [38.0, 0.6, 1.0, 3.0]
problem, solution = runsimulation(
    parameters;
    nucl = nucl,
    gr = gr,
    agg = noaggregation(),
    br = nobreakage(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)

if solution.success
    println("Final concentration: ", solution.concentration[end])
    println("Final d43: ", solution.d43[end], " μm")
end
```

The function returns the constructed `CrystallisationProblem` and a solution
trajectory. Moment solutions expose `concentration`, `d10`, `d32`, `d43`, and
`mu2`; finite-volume and WENO solutions also expose the resolved size
distribution and `d10q`, `d50q`, and `d90q` quantiles. QMOM additionally stores
raw moments and reconstructed quadrature; see [Solvers](docs/solvers.md).

## Choose a starting point

| If you want to… | Read | Run |
| --- | --- | --- |
| run a simulation or choose a solver | [Running simulations](docs/simulation.md), [Solvers](docs/solvers.md) | [Tutorial 1](<examples/Tutorial 1 Running Simulations.jl>) |
| load measurements from Excel | [Measurements and data loading](docs/measurements.md) | [Tutorial 2](<examples/Tutorial 2 Parameter Estimation.jl>) |
| fit kinetic parameters | [Parameter estimation](docs/parameter-estimation.md) | [Tutorial 2](<examples/Tutorial 2 Parameter Estimation.jl>) |
| compare ABCDE and NUTS | [ABCDE routine](docs/abcde.md), [Parameter estimation](docs/parameter-estimation.md) | [Tutorial 5](<examples/Tutorial 5 ABCDE and MCMC.jl>) |
| define a temperature or saturation model | [Temperature profiles](docs/temperature-profiles.md), [Saturation models](docs/saturation-models.md) | [Tutorial 1](<examples/Tutorial 1 Running Simulations.jl>) |
| add a kinetic family or other system component | [Kinetics](docs/kinetics.md), [Bringing your own system](docs/bring-your-own-system.md) | [Tutorial 4](<examples/Tutorial 4 Defining a Custom Kinetic.jl>) |
| model dissolution | [Kinetics](docs/kinetics.md), [Solvers](docs/solvers.md) | [Tutorial 6](<examples/Tutorial 6 Dissolution.jl>) |
| inspect QMOM nodes and weights | [Solvers](docs/solvers.md) | [Tutorial 7](<examples/Tutorial 7 QMOM.jl>) |
| compare real-data MoM and QMOM fits | [Parameter estimation](docs/parameter-estimation.md), [Solvers](docs/solvers.md) | [Tutorial 8](<examples/Tutorial 8 Real-data MoM versus QMOM.jl>) |
| run sensitivity studies | [Sensitivity analysis](docs/sensitivity.md) | [Tutorial 3](<examples/Tutorial 3 Sensitivity Analysis.jl>) |
| run ensemble uncertainty studies | [Ensembles and uncertainty](docs/uq-ensembles.md) | — |

The complete tutorial catalogue, dependencies, and run commands are in
[Tutorials](docs/tutorials.md). The guides in `docs/` are short explanations
of the same workflows; the scripts in `examples/` are the full runnable
versions.

## A structured parameter vector

The flat form is convenient for optimisers. When parameters need to be read or
edited by name, wrap them with the composite axis built from the four kinetic
models:

```julia
using ComponentArrays

axis = paramaxis(nucl, gr, noaggregation(), nobreakage())
parameters_named = ComponentArray(parameters, axis)

parameters_named.nucl.Aj
parameters_named.gr.g
```

See [Running simulations](docs/simulation.md) and [Kinetics](docs/kinetics.md)
for parameter axes and custom model definitions.

## Documentation and development

- [Documentation home](docs/index.md)
- [Tutorial catalogue](docs/tutorials.md)
- [Package audit and release checklist](AUDIT_v1.0.md) (development)

Run the package test suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```
