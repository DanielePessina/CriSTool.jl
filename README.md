# CriSTool

CriSTool is a Julia package for simulating batch crystallisation, fitting
kinetic parameters to measurements, and propagating parameter uncertainty
through population-balance models.

Read the [hosted development documentation](https://danielepessina.github.io/CriSTool.jl/dev/),
including the [API reference](https://danielepessina.github.io/CriSTool.jl/dev/api/)
and [tutorials](https://danielepessina.github.io/CriSTool.jl/dev/tutorials/).

The package provides:

- population-balance solvers based on the Method of Moments (MoM), the
  Quadrature Method of Moments (QMOM), Direct QMOM (DQMOM), finite volumes,
  and WENO;
- nucleation, growth, aggregation, breakage, and signed dissolution kinetics;
- measurement ingestion from CSV/table sources and typed experiment containers;
- solver-aware initial crystal states from mass, d43, and explicit lognormal or
  Gaussian distribution characteristics;
- parameter estimation with Metaheuristics.jl and Optimization.jl-compatible
  algorithms;
- likelihood-free ABCDE and Turing NUTS workflows;
- sensitivity analysis, ensemble simulation, and Makie plotting utilities.

## Installation

CriSTool declares Julia compatibility `^1.10` in `Project.toml`. The tutorial
environment currently targets Julia 1.12. From a repository checkout,
instantiate the package environment with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

After the `0.1.0` registry release, install CriSTool into another Julia
environment with:

```julia
using Pkg
Pkg.add("CriSTool")
```

The runnable examples use a separate environment because some tutorials add
`GlobalSensitivity`, `QuasiMonteCarlo`, and other tutorial-only dependencies:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

## Quick start: one simulation

`runsimulation` takes kinetic models, a parameter vector, and a solver. The
canonical flat parameter vector is ordered as `[nucleation; growth;
dissolution; aggregation; breakage]`. When `diss` is omitted, the compatibility
layout is `[nucleation; growth; aggregation; breakage]`; `nodissolution()`,
no-op aggregation, and no-op breakage contribute empty blocks.

```julia
using CriSTool

nucl = nucl_CNT()
gr   = growth_empirical()

parameters = [38.0, 0.0006, 1e-9 / 60, 3.0]
problem, solution = runsimulation(
    parameters;
    nucl = nucl,
    gr = gr,
    agg = noaggregation(),
    br = nobreakage(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

if solution.success
    println("Final concentration: ", solution.concentration[end])
    println("Final d43 (m): ", solution.d43[end])
end
```

The function returns the constructed `CrystallisationProblem` and a solution
trajectory. Moment solutions expose `concentration`, `d10`, `d32`, `d43`, and
`moment2`; finite-volume and WENO solutions also expose the resolved size
distribution and `d10q`, `d50q`, and `d90q` quantiles. QMOM and DQMOM expose
raw moments and quadrature data; DQMOM currently requires seeded initial
crystals. See the hosted [Solvers guide](https://danielepessina.github.io/CriSTool.jl/dev/solvers/).

## Choose a starting point

| If you want to… | Read | Run |
| --- | --- | --- |
| run a simulation or choose a solver | [Running simulations](https://danielepessina.github.io/CriSTool.jl/dev/simulation/), [Solvers](https://danielepessina.github.io/CriSTool.jl/dev/solvers/) | [Tutorial 1](<examples/Tutorial 1 Running Simulations.jl>) |
| load measurements from CSV | [Measurements and data loading](https://danielepessina.github.io/CriSTool.jl/dev/measurements/) | [Tutorial 2](<examples/Tutorial 2 Parameter Estimation.jl>) |
| fit kinetic parameters | [Parameter estimation](https://danielepessina.github.io/CriSTool.jl/dev/parameter-estimation/) | [Tutorial 2](<examples/Tutorial 2 Parameter Estimation.jl>) |
| compare ABCDE and NUTS | [ABCDE routine](https://danielepessina.github.io/CriSTool.jl/dev/abcde/), [Parameter estimation](https://danielepessina.github.io/CriSTool.jl/dev/parameter-estimation/) | [Tutorial 5](<examples/Tutorial 5 ABCDE and MCMC.jl>) |
| define a temperature or saturation model | [Temperature profiles](https://danielepessina.github.io/CriSTool.jl/dev/temperature-profiles/), [Saturation models](https://danielepessina.github.io/CriSTool.jl/dev/saturation-models/) | [Tutorial 1](<examples/Tutorial 1 Running Simulations.jl>) |
| add a kinetic family or other system component | [Kinetics](https://danielepessina.github.io/CriSTool.jl/dev/kinetics/), [Bringing your own system](https://danielepessina.github.io/CriSTool.jl/dev/bring-your-own-system/) | [Tutorial 4](<examples/Tutorial 4 Defining a Custom Kinetic.jl>) |
| model dissolution | [Kinetics](https://danielepessina.github.io/CriSTool.jl/dev/kinetics/), [Solvers](https://danielepessina.github.io/CriSTool.jl/dev/solvers/) | [Tutorial 6](<examples/Tutorial 6 Dissolution.jl>) |
| inspect QMOM nodes and weights | [Solvers](https://danielepessina.github.io/CriSTool.jl/dev/solvers/) | [Tutorial 7](<examples/Tutorial 7 QMOM.jl>) |
| run seeded DQMOM | [Solvers](https://danielepessina.github.io/CriSTool.jl/dev/solvers/), [Running simulations](https://danielepessina.github.io/CriSTool.jl/dev/simulation/) | — |
| compare real-data MoM and FiniteVol fits | [Parameter estimation](https://danielepessina.github.io/CriSTool.jl/dev/parameter-estimation/), [Solvers](https://danielepessina.github.io/CriSTool.jl/dev/solvers/) | [Tutorial 8](<examples/Tutorial 8 Real-data MoM versus QMOM.jl>) |
| run sensitivity studies | [Sensitivity analysis](https://danielepessina.github.io/CriSTool.jl/dev/sensitivity/) | [Tutorial 3](<examples/Tutorial 3 Sensitivity Analysis.jl>) |
| run ensemble uncertainty studies | [Ensembles and uncertainty](https://danielepessina.github.io/CriSTool.jl/dev/uq-ensembles/) | — |

The complete tutorial catalogue, dependencies, and run commands are in
the hosted [Tutorials](https://danielepessina.github.io/CriSTool.jl/dev/tutorials/).
The scripts in `examples/` are the full runnable versions.

## A structured parameter vector

The flat form is convenient for optimisers. When parameters need to be read or
edited by name, wrap them with the composite axis built from the selected
kinetic models:

```julia
using ComponentArrays

axis = paramaxis(nucl, gr, noaggregation(), nobreakage())
parameters_named = ComponentArray(parameters, axis)

parameters_named.nucl.ln_nucleation_prefactor
parameters_named.gr.growth_order
```

See the hosted [Running simulations](https://danielepessina.github.io/CriSTool.jl/dev/simulation/)
and [Kinetics](https://danielepessina.github.io/CriSTool.jl/dev/kinetics/)
for parameter axes and custom model definitions.

## Documentation and development

- [Hosted documentation](https://danielepessina.github.io/CriSTool.jl/dev/)
- [Hosted API reference](https://danielepessina.github.io/CriSTool.jl/dev/api/)
- [Hosted tutorial catalogue](https://danielepessina.github.io/CriSTool.jl/dev/tutorials/)

Run the package test suite with:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Build the local documentation site with:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

## License

CriSTool is released under the BSD-3-Clause license; see [LICENSE](LICENSE).
The vendored KissABC notice is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
