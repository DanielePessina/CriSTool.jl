# API reference

The API is grouped by the decisions users make. Start with the [simulation
reference](api/simulation.md), then use the page for the subsystem you are
changing or composing.

| Need | Reference |
| --- | --- |
| build a model, set run conditions, simulate, inspect results | [Simulation and results](api/simulation.md) |
| relax a constant autonomous MSMPR to steady state | [Simulation and results](api/simulation.md) |
| set temperature or solubility | [Process conditions](api/conditions.md) |
| choose or extend kinetic models | [Kinetic models and rates](api/kinetics.md) |
| choose a solver or inspect quadrature | [Solvers and quadrature](api/solvers.md) |
| load measurements or evaluate losses | [Measurements and losses](api/measurements.md) |
| estimate parameters or propagate uncertainty | [Inference and uncertainty](api/inference.md) |
| create plots or chain diagnostics | [Plotting and diagnostics](api/plotting.md) |

## Named kinetic parameters

Calling a law with all of its parameter names, e.g.
`EmpiricalGrowth(growth_coefficient = 1e-9, growth_order = 2.0)`, returns a
bound `KineticModel`; `KineticModel(law; parameters=(...))` is the long form.
`OptimisationSpec.bounds`, `BayesianSpec.priors` and `ABCSpec.priors` select
kinetic parameters using nested names such as `growth.growth_coefficient`.
Unselected values stay fixed. `ParameterSamples` uses those same paths for
joint uncertainty draws.

## Cross-cutting contracts

- Numeric SI: seconds, Kelvin, kg/m³, metres, rates in m/s.
- `simulate(model, conditions; solver, saveat)` returns the physical solution;
  `runsimulation` is the low-level interface for autodiff and sensitivity work.
- `RunConditions` hold temperature, initial concentration and solvent values,
  the seed and the operation; `FromObservation` reads a value from time-zero data.
- `Observable` is a time series, including one-point measurements.
- Transient fitting resolves `FromObservation` conditions from time-zero
  observations; those samples are not scored.
- Likelihood requires supplied positive scored variances; noise is never fitted.
- Fits return a new model and retain native backend diagnostics.
- `predict(result)` reuses the fit's solver, mode and options; predicting from
  user samples requires an explicit `solver`.
- Predictions retain failed draws; exclusion from summaries is explicit.
- ForwardDiff and finite differences are tested; reverse Enzyme remains deferred.

The lower-level solver engine still uses the internal block order nucleation,
growth, dissolution, aggregation, breakage. Named model users do not assemble
those vectors. Generated SciML and dispatch-based extension interfaces remain
available for advanced users.

Custom nucleation families may subtype
`CriSTool.AbstractNucleationFunction` by qualification/import; that base is
not currently exported. The standard custom-rate signature is
`(fn, parameters, problem, state, t)`.
