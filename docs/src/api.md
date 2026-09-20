# API reference

The API is grouped by the decisions users make. Start with the [simulation
reference](api/simulation.md), then use the page for the subsystem you are
changing or composing.

| Need | Reference |
| --- | --- |
| construct a problem, run a solve, inspect results | [Simulation and results](api/simulation.md) |
| set temperature or solubility | [Process conditions](api/conditions.md) |
| choose or extend kinetic models | [Kinetic models and rates](api/kinetics.md) |
| choose a solver or inspect quadrature | [Solvers and quadrature](api/solvers.md) |
| load measurements or evaluate losses | [Measurements and losses](api/measurements.md) |
| estimate parameters or propagate uncertainty | [Inference and uncertainty](api/inference.md) |
| create plots or chain diagnostics | [Plotting and diagnostics](api/plotting.md) |

## Parameter layout

For callers without an independent dissolution model, the compatibility layout
is:

```text
[p_nucleation; p_growth; p_aggregation; p_breakage]
```

For new code with independent dissolution, use:

```text
[p_nucleation; p_growth; p_dissolution; p_aggregation; p_breakage]
```

The structured top-level fields are `nucl`, `gr`, `diss`, `agg`, and `br` in
the five-block form. The four-block overload omits `diss` entirely; it is not a
zero-filled dissolution block.

## Cross-cutting contracts

- Public numerical units are SI: seconds, Kelvin, kg/m³, metres, and rates in
  m/s.
- `runsimulation` returns `(problem, solution)` and validates parameter-block
  lengths from the selected kinetic models.
- `Observable` is always a time series, even for one observation.
- `PE_Routine`, `run_abc`, and `MCMC_Routine` default to
  `outputdir = nothing`, so library calls do not write into the current
  directory implicitly.
- ForwardDiff and finite-difference paths are the tested AD story. Reverse-mode
  Enzyme-through-ODE support is deferred.

Custom nucleation families may subtype
`CriSTool.AbstractFPNucleationFunction` by qualification/import; that base is
not currently exported. The standard custom-rate signature is
`(fn, parameters, problem, state, t)`.
