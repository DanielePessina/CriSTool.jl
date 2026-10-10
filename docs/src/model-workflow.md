# One model across simulation, fitting and prediction

A `CrystallisationSystem` describes material properties and solubility.
`LysozymeSystem()` is an ordinary instance with explicit lysozyme defaults.
`KineticModel` binds one law to named values. Calling a built-in law with all
of its parameter names, such as `EmpiricalGrowth(growth_coefficient = 1e-9,
growth_order = 2.0)`, returns a bound `KineticModel`. `CrystallisationModel(system; nucleation,
growth, dissolution, aggregation, breakage)` collects the five slots; omitted
optional slots use no-op laws.

`RunConditions` describe one run: temperature, initial concentration, extra
solvent values, an optional seed and the reactor operation.
`simulate(model, conditions; solver, saveat)` returns a physical solution.
`CrystallisationProblem(model, conditions; solver)` builds the configured
problem explicitly. Advanced callers can still obtain a generated SciML ODE,
control callbacks and convert its solution using the solver interfaces.

A `CrystallisationExperiment` stores measured observables and the
`RunConditions` it was run under. A `FromObservation(:name)` placeholder in
those conditions is read from the experiment's time-zero observation of that
observable. The initial concentration defaults to `FromObservation(:concentration)`,
so a direct simulation must supply a number. Placeholders may stand for the concentration,
another solvent value or any numeric seed field. Initialising measurements are
excluded from scoring. Every other observable retains its own time grid.

`OptimisationSpec`, `BayesianSpec` and `ABCSpec` separately describe studies.
Their nested names select kinetic parameters; unlisted values remain fixed.
No initial condition, material property or noise parameter can be selected.
`prepare_fit(model, experiments, spec; solver, mode, problem_options)` resolves
names and constructs solve templates once (`mode` is `TransientMode()` or
`SteadyMode(SteadyStateOptions(...))`); `loss(prepared, values)` exposes the
selected-vector objective for advanced users and autodiff. `fit(prepared;
threaded=false)` runs the study.

`fit` returns a `FitResult` for optimisation or `InferenceResult` for sampling.
An optimisation result contains a new fitted model, named estimates, observable
objectives and the native backend result. Input models are not mutated.

`ParameterSamples` carries joint kinetic draws in rows and samples in columns,
with common sample IDs. `predict(result)` repeats the fitted experiments, and
`predict(result, runs)` takes new experiments or `RunConditions`; both reuse the
fit's solver, mode and options. `predict(model, runs, samples; solver)` propagates
user-supplied draws and requires an explicit solver. Each draw is reused across
all requested runs.
`PredictionResult.ensembles` retains each run's trajectories and diagnostics.
Physical summaries require explicit `skip_failed=true` to exclude failed solves.
Supplied additive noise can be drawn with `measurement_samples`; it never changes
the ODE or estimates noise parameters.

[Simulation](simulation.md), [fitting](parameter-estimation.md) and
[prediction](uq-ensembles.md) give complete examples.
