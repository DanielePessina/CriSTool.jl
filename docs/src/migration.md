# Migration to named model workflows

This redesign intentionally changes the recommended interface:

| Earlier workflow | Model workflow |
| --- | --- |
| Kinetic law plus an ordered flat vector | `law(name = value, ...)` binds a `KineticModel` |
| Rebuild physical keyword settings in each routine | Reuse `CrystallisationModel` |
| `runsimulation` returns problem and solution | `simulate(model, conditions; solver, saveat)` returns solution |
| Separate fixed and free variants | Select a subset in a study specification |
| Positional PE/MCMC/ABCDE routines | `prepare_fit` then `fit` |
| Prior arrays indexed by kinetic block order | Named `BayesianSpec` or `ABCSpec` |
| First concentration sample treated as C0 | Require mapped time-zero observations |
| Missing likelihood variance gets a fallback | Supplied positive variance or an error loss |
| Seed mesh rescaled to recover omitted mass | Checked support/domain and no silent rescaling |
| Anonymous ensemble matrices | Named joint `ParameterSamples` with sample IDs |

The numerical solver implementations and solution accessors remain reusable.
`runsimulation` stays exported as the low-level interface for automatic
differentiation and sensitivity work. Legacy inference runners
(`PE_Routine`, `MCMC_Routine`, `run_abc`) and vector-assembly helpers
(`prepare_loss`, `LossSetup`, `nuts_model`) are internal and no longer exported.
There is no promise of public backward compatibility for these old assembly
paths. Advanced configured solver and SciML interfaces retain their distinct
capabilities.

## Run conditions, law binding and study settings

A second cleanup moved run conditions out of problems and experiments and
moved study settings into the specifications.

| Before | After |
| --- | --- |
| `CrystallisationModel(system = s, nucleation = ..., growth = ...)` | `CrystallisationModel(s; nucleation = ..., growth = ...)` |
| `KineticModel(nucl_CNT(); parameters = (ln_nucleation_prefactor = 38.0, surface_energy = 6e-4))` | `nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4)` (the long form still works) |
| `CrystallisationProblem(model; initial_conditions = (concentration = 18.0,), temperature = ConstantTemperature(293.15), initial_crystals = seed, operation = op, solver)` | `CrystallisationProblem(model, RunConditions(temperature = 293.15, concentration = 18.0, crystals = seed, operation = op); solver)` |
| `simulate(problem; saveat)` | `simulate(model, conditions; solver, saveat)`; `simulate(problem; saveat)` still works |
| `CrystallisationExperiment(...; temperature, initial_from = (concentration = :concentration,), seed_shape, operation, relaxation_initial)` | `CrystallisationExperiment(; observables, exp_id, conditions = RunConditions(...), metadata)` |
| `initial_from = (crystals = (mass_concentration = :seed_mass, d43 = :d43),)` with `seed_shape = (family = :lognormal, geometric_std = 1.2)` | `crystals = LogNormalInitialCrystals(mass_concentration = FromObservation(:seed_mass), d43 = FromObservation(:d43), geometric_std = 1.2)` |
| `relaxation_initial = (...)` for steady fits | numeric values in the experiment's `RunConditions` |
| `load_measurements(path; initial_from, seed_shape, operation, relaxation_initial)` | `load_measurements(path; conditions = (crystals = ..., operation = ...))` |
| `expt.temperature`, `expt.initial_crystals` | `expt.conditions.temperature`, `expt.conditions.crystals` |
| `fit(prepared; algorithm = DE(...), adtype, searchoptions)` | `OptimisationSpec(; bounds, algorithm = DE(...), adtype, searchoptions)`, then `fit(prepared)` |
| `fit(prepared; sampler = NUTS(), n_samples, n_chains)` | `BayesianSpec(; priors, sampler = NUTS(), n_samples, n_chains)` |
| `fit(prepared; sampler, nparticles, generations, earlystop)` | `ABCSpec(; priors, target, sampler, n_particles, n_generations, early_stop)` |
| `HPC = true` | `fit(prepared; threaded = true)` |
| ABC target derived by a Wilks test from `confidenceinterval` | `ABCSpec(target = ...)` takes an explicit finite number |
| `parameter_samples(posterior)` | `posterior.samples` |
| `predict(model, runs, samples)` with a default solver | `predict(result)` or `predict(result, runs)` reuse the fit's solver, mode and options; `predict(model, runs, samples; solver)` requires `solver` |
| `examples/model_helpers.jl` | removed; the tutorials use the package API directly |

Placeholders are resolved per experiment and the time-zero observations they
consume are not scored. Steady fits relax from explicit numeric conditions, so
their conditions must not contain placeholders.
