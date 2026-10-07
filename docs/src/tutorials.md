# Tutorials

The scripts in `examples/` are complete, runnable demonstrations of the
package's common workflows. They use the package environment plus a small
example environment defined in `examples/Project.toml`.

## Run a tutorial

From the repository root, instantiate the example environment once:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

Then run a script by quoting its filename because the filenames contain
spaces:

```sh
julia --project=examples "examples/Tutorial 1 Running Simulations.jl"
```

Replace the script name to run another tutorial. Tutorials that use Makie open
a figure window or display a figure in the active Julia environment. The
inference tutorials perform sampling, so their runtime depends on the number
of particles, generations, samples, and chains configured in the script.

## Catalogue

| Tutorial | What it demonstrates | Extra input or dependency |
| --- | --- | --- |
| [1 — Running Simulations](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%201%20Running%20Simulations.jl) | Reuses a named model across solver families and reports concentration and d43. | CairoMakie |
| [2 — Parameter Estimation](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%202%20Parameter%20Estimation.jl) | Loads measured initial conditions and fits selected named kinetic parameters with a separate specification. | `examples/fake-experimental-dataset.csv`, Metaheuristics, Turing |
| [3 — Sensitivity Analysis](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%203%20Sensitivity%20Analysis.jl) | Builds a parameter-to-output forward map and computes Sobol and DGSM sensitivity measures. | GlobalSensitivity, QuasiMonteCarlo |
| [4 — Defining a Custom Kinetic](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%204%20Defining%20a%20Custom%20Kinetic.jl) | Defines a custom callable growth law with one named parameter schema. | ComponentArrays, CairoMakie |
| [5 — ABCDE and MCMC](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%205%20ABCDE%20and%20MCMC.jl) | Uses explicit Bayesian and ABC specifications with common named prediction samples. | Turing, Distributions |
| [6 — Dissolution](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%206%20Dissolution.jl) | Runs signed scalar dissolution with MoM, finite volume, and WENO from a seeded initial population. | CairoMakie |
| [7 — QMOM](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%207%20QMOM.jl) | Evolves raw moments, reconstructs a Gaussian quadrature, and plots the nodes alongside `d43`. | CairoMakie |
| [8 — Real-data MoM versus QMOM](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%208%20Real-data%20MoM%20versus%20QMOM.jl) | Fits the same named kinetics to the bundled real dataset using MoM and QMOM. | Metaheuristics, Turing, your CSV dataset |
| [9 — MSMPR Dynamic versus Steady State](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%209%20MSMPR%20Dynamic%20versus%20Steady%20State.jl) | Runs an autonomous fixed-flow MSMPR dynamically and compares the trajectory with the analytic relaxation (concentration and seed moment), computing the scaled residual against the feed steady state. Shows that the final dynamic point is not automatically steady and that callable inlet profiles are transient-only. | CairoMakie |
| [10 — Fed-Batch Mixing and Seeded Growth](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%2010%20Fed-Batch%20Mixing%20and%20Seeded%20Growth.jl) | Runs a fed-batch with a clear feed (pure mixing: volume, concentration and conserved seed inventory) and with constant seeded growth (d43 shifts by G·Δt, inventory conserved, solute consumed). | CairoMakie |

The scripts call `main()` at the end, so they can be run directly from the
command line. They also keep setup values near the top of `main()` to make it
easy to replace the kinetics, parameters, time grid, or experiment data.

## Which guide to read with each tutorial?

- Tutorial 1 pairs with [Running simulations](simulation.md) and
  [Temperature profiles](temperature-profiles.md).
- Tutorials 2 and 5 pair with [Measurements and data loading](measurements.md),
  [Parameter estimation](parameter-estimation.md), and [ABCDE](abcde.md).
- Tutorial 3 pairs with [Sensitivity analysis](sensitivity.md). For ensemble
  propagation, see [Ensembles and uncertainty](uq-ensembles.md).
- Tutorial 4 pairs with [Kinetics](kinetics.md) and [Bringing your own system](bring-your-own-system.md).
- Tutorial 6 pairs with [Kinetics](kinetics.md), [Saturation models](saturation-models.md),
  and [Solvers](solvers.md).
- Tutorial 7 pairs with [Solvers](solvers.md), especially the QMOM output and
  `quadrature(solution, time_index)` sections.
- Tutorials 9 and 10 pair with [Reactor operations and steady state](operations.md).
  Tutorial 9 also pairs with [Running simulations](simulation.md) (configured
  problems and custom SciML solves).

## Synthetic data

Tutorial 2 reads `examples/fake-experimental-dataset.csv` (long format).
Tutorial 5 uses the same synthetic CSV fixture. The CSV is a tutorial fixture; use the loader
options in [Measurements and data loading](measurements.md) to adapt the
workflow to another table layout.
