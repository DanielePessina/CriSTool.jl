# Migration to named model workflows

This redesign intentionally changes the recommended interface:

| Earlier workflow | Model workflow |
| --- | --- |
| Kinetic law plus an ordered flat vector | `law(name = value, ...)` binds a `KineticModel` |
| Rebuild physical keyword settings in each routine | Reuse `CrystallisationModel` |
| `runsimulation` returns problem and solution | `solve(CrystallisationProblem(model, conditions, tspan), solver; saveat)` returns solution |
| Separate fixed and free variants | Select a subset in a study specification |
| Positional PE/MCMC/ABCDE routines | `prepare_fit` then `fit` |
| Prior arrays indexed by kinetic block order | Named `BayesianSpec` or `ABCSpec` |
| First concentration sample treated as C0 | Require mapped time-zero observations |
| Missing likelihood variance gets a fallback | Supplied positive variance or an error loss |
| Seed mesh rescaled to recover omitted mass | Checked support/domain and no silent rescaling |
| Anonymous ensemble matrices | Named joint `ParameterSamples` with sample IDs |

The numerical solver implementations and solution accessors remain reusable.
`solve(remake(problem; p = values), solver)` is the low-level interface for
automatic differentiation and sensitivity work. The vector-assembly helpers
(`prepare_loss`, `LossSetup`, `run_ensemble`) are internal and no longer
exported. Advanced configured solver and SciML interfaces retain their distinct
capabilities.

## Removed positional routines

The positional inference, ensemble and plotting routines were deleted. Their
replacements are:

| Removed | Replacement |
| --- | --- |
| `PE_Routine`, `PE_Routine_Optimisation`, `MH_minimizer` | `fit(prepare_fit(model, experiments, OptimisationSpec(; bounds, algorithm)))`; the minimiser is `result.parameters` |
| `batchLF_procSO`, `batchLF_procMO` | `loss(prepared_fit, values)` and `loss_terms(prepared_fit, values)` |
| `MCMC_Routine`, positional `nuts_model(experiments, prior, nucl, ...)` | `fit(prepare_fit(model, experiments, BayesianSpec(; priors)))` |
| `run_abc`, `ABCDE_Routine`, `ABCDE_Turner_Routine` | `fit(prepare_fit(model, experiments, ABCSpec(; priors, target, sampler)))` with `ABCDESampler()` or `ABCDETurnerSampler()` |
| `kinetic_parameter_symbols` | named paths in `result.samples.names` |
| positional `run_ensemble(distribution or matrix, measurements, nucl, ...)`, `run_ensemble_fixed` | `predict(result)`, `predict(result, runs)` or `predict(model, runs, samples; solver)` |
| `EnsembleMoMSolution`, `EnsembleFVSolution` | `PredictionResult` holding `PredictionEnsemble`s; summarise with `prediction_summary` |
| `distribution_to_matrix`, `prior_to_matrix`, `create_product_prior` | `parameter_samples(names, distribution)` with `product_distribution([...])` |
| `chains_to_matrix` (export) | `result.samples`; the chain stays in `result.backend_result` |
| `plot_measurements_vs_ensemble`, `plot_ps_measurements_vs_ensemble`, `plot_measurements_vs_simulation`, `plot_ps_measurements_vs_simulation`, `ChainMeasurementPlots`, `ABCplot` | `plot_predictions(predictions, experiments)` |
| `ChainPairPlots`, DataFrame `plot_posterior_pairplot` methods | `plot_posterior_pairplot(result)` or `plot_posterior_pairplot(samples; prior, reference)` |
| `CriSTool.forwardsensitivity(problem, saveat)` (SciMLSensitivity `ODEForwardSensitivityProblem`, MoM and FiniteVol batch only) | `ForwardDiff.jacobian(θ -> solve(remake(problem; p = θ), solver; saveat).concentration, θ)`, which covers every solver and operation; see [Sensitivity analysis](sensitivity.md) |
| `CRISTOOL_PALETTE`, terminal start/end panels, `outputdir` persistence | removed; plotting functions return a `Makie.Figure`; save it with `save(path, figure)` |
| `ChainStatsPlots(chain; burnin, title, show_title, saveplot, savestring, savedir, showplot)` | `plot_chain_diagnostics(chain; burnin, title)`; an empty `title` draws none |
| `savename`, `savedir`, `showplot` keywords of `plot_predictions` and `plot_posterior_pairplot` | removed; the figure is returned, never saved or displayed: `save(path, figure)`, `display(figure)` |

## Run conditions, law binding and study settings

A second cleanup moved run conditions out of problems and experiments and
moved study settings into the specifications.

| Before | After |
| --- | --- |
| `CrystallisationModel(system = s, nucleation = ..., growth = ...)` | `CrystallisationModel(s; nucleation = ..., growth = ...)` |
| `KineticModel(nucl_CNT(); parameters = (ln_nucleation_prefactor = 38.0, surface_energy = 6e-4))` | `CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4)` (the long form still works) |
| `CrystallisationProblem(model; initial_conditions = (concentration = 18.0,), temperature = ConstantTemperature(293.15), initial_crystals = seed, operation = op, solver)` | `CrystallisationProblem(model, RunConditions(temperature = 293.15, concentration = 18.0, crystals = seed, operation = op), tspan)`; the solver goes to `solve` |
| `simulate(problem; saveat)` | `solve(problem, solver; saveat)` |
| `CrystallisationExperiment(...; temperature, initial_from = (concentration = :concentration,), seed_shape, operation, relaxation_initial)` | `CrystallisationExperiment(; observables, exp_id, conditions = RunConditions(...), metadata)` |
| `initial_from = (crystals = (mass_concentration = :seed_mass, d43 = :d43),)` with `seed_shape = (family = :lognormal, geometric_std = 1.2)` | `crystals = LogNormalInitialCrystals(mass_concentration = FromObservation(:seed_mass), d43 = FromObservation(:d43), geometric_std = 1.2)` |
| `relaxation_initial = (...)` for steady fits | numeric values in the experiment's `RunConditions` |
| `load_measurements(path; initial_from, seed_shape, operation, relaxation_initial)` | `load_measurements(path; conditions = (crystals = ..., operation = ...))` |
| `expt.temperature`, `expt.initial_crystals` | `expt.conditions.temperature`, `expt.conditions.crystals` |
| `fit(prepared; algorithm = DE(...), adtype, searchoptions)` | `OptimisationSpec(; bounds, algorithm = DE(...), adtype, searchoptions)`, then `fit(prepared)` |
| `fit(prepared; sampler = NUTS(), n_samples, n_chains)` | `BayesianSpec(; priors, sampler = NUTS(), n_samples, n_chains)`; requires `using Turing` (a weak dependency; `sampler = nothing` means `NUTS()`) |
| `fit(prepared; sampler, nparticles, generations, earlystop)` | `ABCSpec(; priors, target, sampler, n_particles, n_generations, early_stop)` |
| `HPC = true` | `fit(prepared; parallel = EnsembleThreads())` |
| `fit(prepared; threaded = true)` | `fit(prepared; parallel = EnsembleThreads())`; the specification picks candidate or experiment threading |
| ABC target derived by a Wilks test from `confidenceinterval` | `ABCSpec(target = ...)` takes an explicit finite number |
| `parameter_samples(posterior)` | `posterior.samples` |
| `predict(model, runs, samples)` with a default solver | `predict(result)` or `predict(result, runs)` reuse the fit's solver, mode and options; `predict(model, runs, samples; solver)` requires `solver` |
| `examples/model_helpers.jl` | removed; the tutorials use the package API directly |

Placeholders are resolved per experiment and the time-zero observations they
consume are not scored. Steady fits relax from explicit numeric conditions, so
their conditions must not contain placeholders.

## Simulation modes and prepared losses

| Before | After |
| --- | --- |
| `prepare_fit(...; mode = :steady, steady_options = SteadyStateOptions(...))` | `prepare_fit(...; mode = SteadyMode(SteadyStateOptions(...)))` |
| `predict(model, runs, samples; solver, mode = :steady, steady_options)` | `predict(model, runs, samples; solver, mode = SteadyMode(...))` |
| `mode = :transient` | `mode = TransientMode()` (the default) |
| `loss(lf, problem, params, experiments)` on a template problem | `loss(prepare_fit(model, experiments, spec), values)` |
| `logMLE().string`, `logMLE().symbols`, `mae().string` | removed |
| `logMLE(weighting = [1.0, 1.0])` default | `weighting = (;)`: every observable weighs 1 |
| `ensemble.concentration` on a `PredictionEnsemble` | `observable_values(ensemble, :concentration)` |
| `prepare_fit(...; initial_time)` (always rejected) | not a keyword; runs start at time zero |

`PredictionEnsemble` is no longer an `AbstractSolution`. Observable weights are
resolved once when the loss is prepared and stored with the prepared setup.

## One solution type and method-based solvers

The four solver-specific result types were merged, and solver configuration
flags became values rather than symbols.

| Before | After |
| --- | --- |
| `CrystallisationMoMSolution`, `CrystallisationFVSolution`, `CrystallisationQMOMSolution`, `CrystallisationDQMOMSolution` | `CrystallisationSolution{Population}` with `population::MomentPopulation`, `MeshPopulation` or `QuadraturePopulation`; dispatch on `CrystallisationSolution{<:MeshPopulation}` |
| `solution.numberdensity`, `solution.voldensity`, `solution.d50q` | `solution.population.numberdensity` (etc.), or `state_vars(solution)` / `size_metrics(solution)` |
| `solution.moments`, `solution.nodes`, `solution.weights` | `solution.population.moments` / `.nodes` / `.weights`; MoM solutions now also carry `moments` |
| `solution.quadrature_nodes`, `solution.quadrature_weights`, `inversion_diagnostics`, `projection_diagnostics` | `solution.population.nodes`, `.weights`, `.diagnostics` |
| `qmom_quadrature(solution, i)`, `dqmom_quadrature(solution, i)` | `quadrature(solution, i)` |
| `getmomentsizes(problem, solution)` | `size_metrics(solution)` |
| `get_characteristic_size(solution)` | `solution.d43[end]` or `solution.population.d50q[end]` |
| `nmoments(::QMOM)` | `moment_order(solver)` |
| `steady.concentration`, `steady.d43`, `steady.time`, `steady.moments` | `steady.equilibrium.concentration` (etc.); `steady.solution` is the one-sample `CrystallisationSolution` |
| `timestepping_algorithm = :ssprk43` (`:auto`) | `timestepping_algorithm = SSPRK43()` (`nothing`) |
| `CriSTool.OrdinaryDiffEq.Euler()` and other algorithms through CriSTool's `OrdinaryDiffEq` dependency | load the algorithm's subpackage yourself (for example `using OrdinaryDiffEqLowOrderRK: Euler`, or `using OrdinaryDiffEq`); CriSTool depends only on `OrdinaryDiffEqTsit5` and `OrdinaryDiffEqSSPRK` |
| `tolerance_mode = :auto` (`:scalar`) | `auto_abstol = true` (`false`) |
| `QMOM(inversion_algorithm = :wheeler)`, solver `string` fields | removed |
| `runsimulation(params, nucl, gr, agg, br, c0; ...)`, `runsimulation(params; nucl, gr, ..., initial_crystals)` | `solve(remake(problem; p = params), solver; saveat)` |
| `DistributionInitialCrystals(...; weighting = :number)`, `weighting = :volume` | `weighting = NumberWeighted()` (default), `weighting = VolumeWeighted()`; the `weighting` field holds the singleton |

Every seed description now passes through one resolution. Lognormal and
Gaussian seeds become a `DistributionInitialCrystals`, and mesh solvers
receive exact cell integrals of its density whichever entry point built the
state (model conditions, experiments, `initial_state_from_characteristics` or
feeds). Previously experiment seeds were point-sampled on the cell centres and
rescaled to the target mass.

## Kinetic laws: names, dissolution slot and parameter schema

Kinetic laws and losses now use CamelCase type names. There are no aliases;
replace each old name.

| Before | After |
| --- | --- |
| `nucl_CNT` | `CNTNucleation` |
| `nucl_CNTnoS` | `CNTNucleationNoSupersaturation` |
| `nucl_empirical` | `EmpiricalNucleation` |
| `nucl_empirical_energy` | `ArrheniusNucleation` |
| `nucl_secondary` | `SecondaryNucleation` |
| `nucl_prim_plus_second`, `nucl_CNT_plus_second` | `CombinedNucleation(primary, secondary)` for any two nucleation laws; parameters nest under `primary` and `secondary` |
| `growth_empirical` | `EmpiricalGrowth` |
| `growth_empirical_length` | `EmpiricalLengthGrowth` |
| `growth_energy`, `growth_energy_est` | `ArrheniusGrowth`, with `activation_energy` a parameter between `log10_growth_coefficient` and `growth_order` |
| `growth_BCF` | `BCFGrowth` |
| `growth_BpS` | `BirthSpreadGrowth` |
| `growth_dissolution` | `ArrheniusDissolution`, in the `dissolution` slot |
| `growth_dissolution_length` | `ArrheniusLengthDissolution`, in the `dissolution` slot |
| `growth_energy_dissolution` | `growth = ArrheniusGrowth(...)` with `activation_energy = 53e3`, plus `dissolution = ArrheniusDissolution(...)` |
| `nodissolution`, `nondissolution` | `NoDissolution` |
| `noaggregation` | `NoAggregation` |
| `aggr_scalar` | `ConstantAggregation` |
| `aggr_linear` | `SizeSumAggregation` |
| `aggr_linearvol` | `VolumeSumAggregation` |
| `aggr_avg` | `AverageAggregation` |
| `nobreakage` | `NoBreakage` |
| `breakage_empirical` | `EmpiricalBreakage` |
| `breakage_uniform` | `UniformBreakage` |
| `logMLE` | `GaussianNLL` |
| `mae` | `MeanAbsoluteError` |
| `AbstractFPScalarGrowthFunction`, `AbstractFPLengthGrowthFunction` | `AbstractScalarGrowthFunction`, `AbstractLengthGrowthFunction` |
| `AbstractFPScalarDissolutionFunction`, `AbstractFPLengthDissolutionFunction` | `AbstractScalarDissolutionFunction`, `AbstractLengthDissolutionFunction` |
| `AbstractFPNucleationFunction`, `AbstractFPGrowthFunction` | `AbstractNucleationFunction`, `AbstractGrowthFunction` |

Other changes to kinetic laws:

| Before | After |
| --- | --- |
| A dissolution law in the `growth` slot | Dissolution laws form their own family and go in the `dissolution` slot; `AbstractDissolutionFunction` is no longer a subtype of `AbstractGrowthFunction` |
| `nucl_CNT_fixed`, `nucl_empirical_fixed`, `growth_empirical_fixed` | bind the ordinary law and leave its parameters out of the fit selection; switch nucleation off with `NoNucleation()` (the `CrystallisationModel` default) |
| `law.nparams`, `law.string`, `law.symbols` | `paramaxis(law)` is the only parameter declaration |
| custom laws without `paramaxis` (generic `θ1, θ2, ...` axis) | every law must declare `paramaxis` |
| `paramaxis(nucl, gr, agg, br)` | `paramaxis(nucl, gr, diss, agg, br)` |
| `ConstantSaturation`, `PolynomialSaturation`, `CallableSaturation`, `AbstractSaturationModel`, `lysozyme_saturation` | the `*Solubility` names |
| `net_growthrate` | `net_growth_rate` |
| `AbstractExperiment`, `AbstractObservable`, `AbstractCrystallisationProblem`, `AbstractDDNucleationFunction`, `AbstractDDGrowthFunction`, `AbstractDDScalarGrowthFunction` | removed |

## Configured problems: one source per value

`CrystallisationProblem` now holds the `CrystallisationModel` and the run
conditions instead of copies of their fields, and its only public constructor is
`CrystallisationProblem(model, conditions, tspan, p; solvent_dynamics,
solid_mass_concentration_threshold)`; *SciML-shaped simulation* below lists the
latest changes.

| Before | After |
| --- | --- |
| `CrystallisationProblem(; kinetics_nucleationfunction, parameterset_nucleation, crystal_density, initial_concentration, ...)` | `CrystallisationProblem(model, RunConditions(...), tspan)`; bind kinetics in the `CrystallisationModel` |
| `problem.kinetics_growthfunction` (and the other four slots) | `problem.model.growth.law` |
| `problem.parameterset_growth` (and the other four slots) | `kinetic_parameter_vector(problem).gr`, or `problem.model.growth.parameters` for the named values |
| `problem.crystal_density`, `.volume_shape_factor`, `.molecular_volume`, `.saturation_model`, `.material_system` | `problem.model.system.crystal_density` (etc.); `problem.model.system` |
| `problem.initial_concentration` | `initial_concentration(problem)` (stored once in `problem.conditions`) |
| `problem.initial_state` (full state vector) | `initial_state_from_characteristics(problem, solver, seed)`; the seed in `problem.conditions` becomes a solver population at `solve` time |
| `problem.R_gas_constant`, `problem.boltzmann_constant` | module constants `CriSTool.CRISTOOL_GAS_CONSTANT`, `CriSTool.CRISTOOL_BOLTZMANN_CONSTANT` |
| `crystallisation_odeproblem(solver, problem, saveat)` solver method | `crystallisation_odeproblem(problem, solver)`; `problem.p` becomes the ODE parameter `p` |
| `system.offset` for an entry of `CrystallisationSystem(properties = (offset = ...,))` | `system.properties.offset` (`system_properties(context).offset` in callable laws) |
| `context.system` in callable laws (`KineticContext` field) | removed; `system_properties(context)` for extra properties, `context.problem.model.system` for the system |
| `CallableNucleation((values, context) -> 0.0; parameters = (;))` to switch nucleation off | `NoNucleation()`, the default `nucleation` of `CrystallisationModel`; built in and time invariant, so steady solves need no `autonomous = true` for it |
| Unknown `CrystallisationProblem` keywords copied onto the problem | `ArgumentError` naming the unsupported settings |

## SciML-shaped simulation

Simulation now follows the SciML problem/solve pattern. A
`CrystallisationProblem` no longer stores the solver; it gains a time span and
the kinetic values `p`, and the solver is the second argument of `solve`, as
the algorithm is for an `ODEProblem`. There are no deprecation shims.

| Before | After |
| --- | --- |
| `simulate(model, conditions; solver, saveat)` | `solve(CrystallisationProblem(model, conditions, tspan), solver; saveat)` |
| `CrystallisationProblem(model, conditions; solver)` | `CrystallisationProblem(model, conditions, tspan)`; pass the solver to `solve(problem, solver)` |
| `simulate(problem; saveat)` | `solve(problem, solver; saveat)`; `saveat` is a step or a vector of times inside `tspan` |
| `runsimulation(params, problem; save_idx)` | `solve(remake(problem; p = params), solver; saveat)` |
| `runsimulation(problem; save_idx)` | `solve(problem, solver; saveat)` |
| `algorithm = Tsit5()` on a direct simulation | `alg = Tsit5()` on `solve`; `prepare_fit` and `predict` keep `algorithm` |
| `solution.time`, `ensemble.time` | `solution.t`, `ensemble.t` |
| `solution.success` | `solution.retcode` (`ReturnCode.T`) or `successful_retcode(solution)` |
| `observable_values(solution, :d43)` | also `solution[:d43]` (any named trajectory) |
| `solve_steadystate(problem; options, initial_guess, algorithm)` | `solve(CrystallisationSteadyStateProblem(problem; options, u0), solver; alg)` |
| `steady.success` | `steady.retcode` (`Success`, `MaxIters`, `Failure` or the ODE code) or `successful_retcode(steady)`; the ODE code is `steady.diagnostics.ode_retcode` |
| `steady.equilibrium.time` | `steady.equilibrium.t` |
| `crystallisation_odeproblem(problem, saveat, values)` | `crystallisation_odeproblem(remake(problem; p = values), solver)` |
| `crystallisation_solution(problem, ode_solution)` | `crystallisation_solution(problem, solver, ode_solution)` |
| `initial_state_from_characteristics(problem, seed)` | `initial_state_from_characteristics(problem, solver, seed)` |
| `seed_domain_diagnostics(problem, seed)` | `seed_domain_diagnostics(solver, seed)` |
| `problem.solver`, `problem.initial_population`, `problem.temp_profile` | the solver is a `solve` argument; `problem.conditions` holds temperature, solvent values, seed and operation |
| `rate(law, values, prob::CrystallisationProblem, state, t)` in custom laws | leave `prob` untyped: laws receive the internal solve-time problem, read with `supersaturation(prob, state, t)`, `solvent_state(prob, state)` and `prob.model` |
| `CrystallisationProblem(...)` throwing on a seed outside the mesh or an unsupported solver/kinetics combination | the same `ArgumentError` from `solve` |
