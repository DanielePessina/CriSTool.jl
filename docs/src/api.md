# API reference

The API is grouped by the decisions users make. Start with the [simulation
reference](api/simulation.md), then use the page for the subsystem you are
changing or composing.

| Need | Reference |
| --- | --- |
| construct a problem, run a solve, inspect results | [Simulation and results](api/simulation.md) |
| relax a constant autonomous MSMPR to steady state | [Simulation and results](api/simulation.md) |
| set temperature or solubility | [Process conditions](api/conditions.md) |
| choose or extend kinetic models | [Kinetic models and rates](api/kinetics.md) |
| choose a solver or inspect quadrature | [Solvers and quadrature](api/solvers.md) |
| load measurements or evaluate losses | [Measurements and losses](api/measurements.md) |
| estimate parameters or propagate uncertainty | [Inference and uncertainty](api/inference.md) |
| create plots or chain diagnostics | [Plotting and diagnostics](api/plotting.md) |

## Named kinetic parameters

`KineticModel(law; parameters=(...))` binds values to their law.
`OptimisationSpec.bounds`, `BayesianSpec.priors` and `ABCSpec.priors` select
kinetic parameters using nested names such as `growth.growth_coefficient`.
Unselected values stay fixed. `ParameterSamples` uses those same paths for
joint uncertainty draws.

## Cross-cutting contracts

- Numeric SI: seconds, Kelvin, kg/m³, metres, rates in m/s.
- `simulate(configured_problem)` returns the physical solution.
- `Observable` is a time series, including one-point measurements.
- Transient fitting requires mapped time-zero initial observations.
- Likelihood requires supplied positive scored variances; noise is never fitted.
- Fits return a new model and retain native backend diagnostics.
- Predictions retain failed draws; exclusion from summaries is explicit.
- ForwardDiff and finite differences are tested; reverse Enzyme remains deferred.

The lower-level solver engine still uses the internal block order nucleation,
growth, dissolution, aggregation, breakage. Named model users do not assemble
those vectors. Generated SciML and dispatch-based extension interfaces remain
available for advanced users.

Custom nucleation families may subtype
`CriSTool.AbstractFPNucleationFunction` by qualification/import; that base is
not currently exported. The standard custom-rate signature is
`(fn, parameters, problem, state, t)`.
