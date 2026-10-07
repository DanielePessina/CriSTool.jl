# One model across simulation, fitting and prediction

A `CrystallisationSystem` describes material properties and solubility.
`LysozymeSystem()` is an ordinary instance with explicit lysozyme defaults.
`KineticModel` binds one law to named values. A `CrystallisationModel` collects
nucleation, growth, dissolution, aggregation and breakage; omitted optional slots
use no-op laws.

A `CrystallisationProblem(model; ...)` adds experiment conditions, explicit
simulation initial conditions and a numerical solver. `simulate` returns a
physical solution. Advanced callers can still obtain a generated SciML ODE,
control callbacks and convert its solution using the solver interfaces.

A `CrystallisationExperiment` stores measured observables and declared process
conditions. `initial_from` maps time-zero observations to initial solvent values
and optional seed characteristics. Initialising measurements are excluded from
scoring. Every other observable retains its own time grid.

`OptimisationSpec`, `BayesianSpec` and `ABCSpec` separately describe studies.
Their nested names select kinetic parameters; unlisted values remain fixed.
No initial condition, material property or noise parameter can be selected.
`prepare_fit` resolves names and constructs solve templates once; `loss(prepared,
values)` exposes the selected-vector objective for advanced users and autodiff.

`fit` returns a `FitResult` for optimisation or `InferenceResult` for sampling.
An optimisation result contains a new fitted model, named estimates, observable
objectives and the native backend result. Input models are not mutated.

`ParameterSamples` carries joint kinetic draws in rows and samples in columns,
with common sample IDs. `predict` reuses each draw across all requested runs.
`PredictionResult.ensembles` retains each run's trajectories and diagnostics.
Physical summaries require explicit `skip_failed=true` to exclude failed solves.
Supplied additive noise can be drawn with `measurement_samples`; it never changes
the ODE or estimates noise parameters.

[Simulation](simulation.md), [fitting](parameter-estimation.md) and
[prediction](uq-ensembles.md) give complete examples.
