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
| [1 — Running Simulations](<../examples/Tutorial 1 Running Simulations.jl>) | Runs the same kinetics under constant, ramp, and callable temperature profiles and compares concentration and `d43`. | CairoMakie |
| [2 — Parameter Estimation](<../examples/Tutorial 2 Parameter Estimation.jl>) | Loads an Excel workbook, balances measurement variance, estimates a point optimum with `PE_Routine`, then runs ABCDE and NUTS. | `examples/fake-experimental-dataset.xlsx`, Metaheuristics, Turing |
| [3 — Sensitivity Analysis](<../examples/Tutorial 3 Sensitivity Analysis.jl>) | Builds a parameter-to-output forward map and computes Sobol and DGSM sensitivity measures. | GlobalSensitivity, QuasiMonteCarlo |
| [4 — Defining a Custom Kinetic](<../examples/Tutorial 4 Defining a Custom Kinetic.jl>) | Adds a growth model from a user script using a subtype, `paramaxis`, and `growthrate`. | ComponentArrays, CairoMakie |
| [5 — ABCDE and MCMC](<../examples/Tutorial 5 ABCDE and MCMC.jl>) | Creates a noisy synthetic experiment and compares likelihood-free ABCDE with Turing NUTS using the same forward model and loss. | Turing, Distributions; no workbook |
| [6 — Dissolution](<../examples/Tutorial 6 Dissolution.jl>) | Runs signed scalar dissolution with MoM, finite volume, and WENO from a seeded initial population. | CairoMakie |
| [7 — QMOM](<../examples/Tutorial 7 QMOM.jl>) | Evolves raw moments, reconstructs a Gaussian quadrature, and plots the nodes alongside `d43`. | CairoMakie |
| [8 — Real-data MoM versus QMOM](<../examples/Tutorial 8 Real-data MoM versus QMOM.jl>) | Runs substantial optimization and four-chain NUTS fits on your workbook, first with CNT + empirical growth under MoM and then with scalar aggregation + empirical breakage under QMOM. | Metaheuristics, Turing, your workbook |

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

## Synthetic data

Tutorial 2 reads `examples/fake-experimental-dataset.xlsx`, using the
`Unseeded_PE` sheet. Tutorial 5 creates its experiment in memory, so it does
not depend on a workbook. The workbook is a tutorial fixture; use the loader
options in [Measurements and data loading](measurements.md) to adapt the
workflow to another table layout.
