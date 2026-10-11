# Changelog

## 0.1.0 — unreleased

- Opt-in `static` test group (`test_args = ["static"]`, also in `"all"`): asserts zero
  allocations per ODE right-hand-side call for Float64 and ForwardDiff dual state and
  parameters on every solver, plus JET optimisation analysis of those right-hand sides.
  CI runs it as its own job and adds a non-blocking `JET.report_package` job.
- REPL display: run conditions, seeds, solvers, operations and feeds, kinetic
  and crystallisation models, solutions (transient and steady), fits,
  inference results and predictions print a readable summary, with a compact
  one-line form inside containers. Sizes are shown in µm; stored values are
  unchanged.
- Opt-in multithreading across experiments: `prepare_fit`, `fit` and `predict`
  accept `parallel = EnsembleThreads()` (`EnsembleSerial()` and `EnsembleThreads()`
  are re-exported from SciMLBase). Loss values, ForwardDiff gradients, fits and
  predictions are bitwise identical to serial. `predict` now defaults to
  `EnsembleSerial()` instead of always threading its (run, draw) pairs.

- Plotting moved to package extensions: `using CairoMakie` (or another Makie
  backend) enables `plot_predictions` and `ChainStatsPlots`; `PairPlots`
  additionally enables `plot_posterior_pairplot`. `using CriSTool` no longer
  loads Makie.
- Makie conversion traits for solutions: `lines(solution)`, `lines(solution, :d43)`
  (any `observable_values` name) and `lines(solution, solver, time_index)` for
  mesh size distributions; `plot(solution)` draws concentration against time.
- Turing is a weak dependency: `fit` on a `BayesianSpec` needs `using Turing`
  (`CriSToolTuringExt`), and `BayesianSpec`'s default `sampler` is now `nothing`,
  meaning `Turing.NUTS()`. `using CriSTool` no longer loads Turing, and
  `CriSTool.predict` is no longer Turing's `predict`: import Turing selectively
  (`using Turing: NUTS`) to avoid a name clash on `predict`. The AIS-versus-
  DynamicPPL ambiguity guard moved into the extension.
- Depend on `OrdinaryDiffEqTsit5`, `OrdinaryDiffEqSSPRK` and `SciMLBase` instead
  of the full `OrdinaryDiffEq`. Other time steppers come from the user's own
  environment (for example `OrdinaryDiffEqSDIRK` or `OrdinaryDiffEq`).
- FlexiChains is a weak dependency; `CriSToolFlexiChainsExt` resolves the
  vendored AIS `bundle_samples` dispatch ambiguity when FlexiChains is loaded.
- Removed the unexported `forwardsensitivity` helper and the SciMLSensitivity
  dependency. Differentiate `runsimulation(parameters, problem)` with
  ForwardDiff for local sensitivities; it covers every solver and operation.
- Named material and kinetic models reused across simulation, selected-parameter
  optimisation, explicit-prior Bayesian/ABC inference and joint predictions.
- Callable scalar/node-dependent kinetics with supported named parameter and
  physical context interfaces; generic material properties fail when required
  values are missing instead of inheriting lysozyme values.
- Time-zero initial-observation mappings and explicit seed shapes, preserved by
  data loading, balancing and bootstrap. Likelihood scoring requires supplied
  positive variances; no default noise variance is invented.
- Distribution-based seed populations with number/volume conversion, checked
  improper moments and mesh mass-loss diagnostics without silent renormalisation.
- Fitted models returned without input mutation, named joint uncertainty draws,
  retained failed samples and explicit optional future-measurement noise.
- Positional inference/ensemble routines are no longer exported; see the named
  workflow and migration guides. Low-level numerical interfaces remain available.

- Dynamic batch, fixed-volume MSMPR, and variable-volume fed-batch population
  balances, with independent initial seeds and crystal-bearing or clear feeds.
- Explicit autonomous MSMPR steady solving by dynamic relaxation, checked with
  scaled full-state residuals, physical constraints, product flows and inventory
  balance diagnostics.
- Configured physical problems and prepared setups supported across simulation,
  parameter estimation, ABC, NUTS/MCMC and posterior prediction.
- Named observables and weights, pooled objectives across irregular experiment
  subsets, local observable projections and retained failed-sample diagnostics.
- Convenience preparation rejects late concentration samples used as initial
  conditions; projections cannot shadow built-in outputs or configured solvent
  fields from any experiment.
- Preserved initial populations, independent initial concentration, temperature
  profiles and reactor configuration during prepared simulation.
- Public conversion of package-generated SciML solutions into physical results.
- Corrected MoM population/solvent layout, finite-volume gradient signs and OSPRE
  reconstruction, and positive WENO smoothness indicators and boundary units.
- Consistent direct/prepared solver tolerances and solve-local callback ownership,
  including derivative-safe AutoAbstol and dimensional QMOM tolerances.
- WENO enforces a solve-local CFL step bound while preserving a caller's
  smaller timestep cap. Length-dependent rate calculations and mesh output
  construction avoid repeated work; unused WENO caches are removed.
- Prepared finite-volume and WENO losses validate the number of scalar kinetic
  parameters, including empty parameter blocks.
- FV/WENO boundary-flow diagnostics use the solved candidate's kinetic
  parameters, including when checking prepared steady-state mass balances.
- Default ensemble observables follow each prepared experiment's system;
  one-parameter ensembles also accept univariate distributions.
- Captured-data NUTS models avoid DynamicPPL's generated argument-conversion
  failure on Julia 1.12/1.13 while preserving conditioning and task-local setups.
- Steady reconstruction failures become failed candidates during inference.
  User kinetic errors during mesh boundary post-processing still propagate.
  Mesh steady results preserve and report machine-roundoff undershoots while
  rejecting larger negative populations.
- Direct and prepared steady workflows share the 100-residence-time relaxation
  budget; unsupported steady reactor configurations reject during preparation.
  Convergence saves the actual equilibrium after extinction events, and steady
  observables contain only that final sample even when callbacks save events.
- ABC thresholds selected for the discrepancy, with explicit targets for custom
  discrepancies; supplied optimizer configuration is preserved.
- New reactor guides, dynamic-versus-steady MSMPR and fed-batch tutorials, and
  independent conservation, reconstruction and gradient regression tests.

All dimensional numeric inputs use SI units. Julia compatibility remains
1.10–1.13. DQMOM requires a positive seeded population; unsupported combinations
report explicit errors. Reverse Enzyme differentiation, GPU solving, units
adapters and multidimensional/multivessel models remain outside this release.

This version has not been registered, tagged or published. Release gates and
the final tested commit must be accepted before publication.
- `Compat.jl` is a dependency; supported but non-exported names
  (`CRISTOOL_GAS_CONSTANT`, `CRISTOOL_BOLTZMANN_CONSTANT`,
  `AbstractDiscretisedSolver`, `KissABC`) are marked `public` on Julia 1.11 and
  later.
- Added `CONTRIBUTING.md` and a CompatHelper workflow. The Distributions
  compat range is now open-ended (`0.25.88`) instead of capped at `0.25.126`.
