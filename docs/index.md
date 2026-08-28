# CriSTool docs

This folder collects short, usage-first guides for the main workflows in CriSTool.

## Quick start

```julia
using CriSTool

params = [38.0, 0.7, 1.0, 3.0]
nucl = nucl_CNT()
growth = growth_empirical()

problem, solution = runsimulation(
    params;
    nucl = nucl,
    gr = growth,
    agg = noaggregation(),
    br = nobreakage(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
)

if solution.success
    println("Final concentration: ", solution.concentration[end])
    println("Final d43: ", solution.d43[end])
end
```

## Guides

- [Running simulations](simulation.md)
- [Kinetics: nucleation and growth](kinetics.md)
- [Solvers](solvers.md)
- [Parameter estimation](parameter-estimation.md)
- [Optimisation routines](optimisation.md)
- [ABCDE routine](abcde.md)
- [Measurements and data loading](measurements.md)
- [Temperature profiles](temperature-profiles.md)
- [Saturation models and supersaturation](saturation-models.md)
- [Ensembles and uncertainty](uq-ensembles.md)

## Tutorials

Runnable scripts in `CriSTool/examples/`:

- `Tutorial 1 Running Simulations.jl` — three temperature profiles side-by-side.
- `Tutorial 2 Parameter Estimation.jl` — PE + ABCDE + Turing NUTS on experimental data.
- `Tutorial 3 Sensitivity Analysis.jl` — global and local sensitivity workflows.
- `Tutorial 4 Defining a Custom Kinetic.jl` — three-step (subtype + `paramaxis` + rate) pattern for adding a kinetic family from a user script.
- `Tutorial 5 ABCDE and MCMC.jl` — `run_abc` (likelihood-free) vs Turing NUTS on synthetic data; self-contained.

Tutorials 2 and 5 use `examples/fake-experimental-dataset.xlsx` (sheet `Unseeded_PE`), a 5-experiment synthetic dataset with irregular sampling and heteroscedastic noise.
