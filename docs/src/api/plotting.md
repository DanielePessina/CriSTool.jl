# Plotting and diagnostics

These functions are defined by package extensions. Run `using CairoMakie` (or
another Makie backend) to enable `plot_predictions` and `plot_chain_diagnostics`, and
`using CairoMakie, PairPlots` to enable `plot_posterior_pairplot`. These
functions return a `Makie.Figure` and never save or display it; save it with
Makie's `save(path, figure)`.

## Plotting a solution directly

With a Makie backend loaded, a `CrystallisationSolution` can be passed to any
point-based plot function (`lines`, `scatter`, `stairs`, `scatterlines`, and
their `!` forms). Values are in SI units (s, kg/m³, m); rescale or label the
axes yourself, for example with `axis = (xlabel = "Time [s]",)`.

| Call | Draws |
| --- | --- |
| `lines(solution)` or `plot(solution)` | concentration against time |
| `lines(solution, :d43)` | any name accepted by `observable_values`: `:concentration`, `:d10`, `:d32`, `:d43`, `:moment2`, mesh quantiles `:d10q`/`:d50q`/`:d90q`, solvent and reactor states |
| `lines(solution, solver, time_index)` | mesh solutions only: number density against the cell centres of the `FiniteVol` or `WENO` solver that produced it, at saved time `time_index` |

The mesh is a property of the solver rather than the solution, so the solver
is passed explicitly. Observables that are not one value per saved time (such as
population matrices) raise an `ArgumentError`. Steady-state results and
prediction ensembles have no conversion; use `plot_predictions` for
ensembles.

```julia
using CairoMakie
solver = FiniteVol(meshsize = 80, lmax = 50e-6)
solution = solve(CrystallisationProblem(model, conditions, (0.0, 7200.0)), solver;
    saveat = 0.0:600.0:7200.0)
lines(solution, :d43; axis = (xlabel = "Time [s]", ylabel = "d43 [m]"))
lines(solution, solver, 5)   # size distribution at the fifth saved time
```

## Figures

```@docs
plot_predictions
plot_posterior_pairplot
plot_chain_diagnostics
```
