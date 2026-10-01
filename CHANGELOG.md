# Changelog

## 0.1.0 — unreleased

- Dynamic batch, fixed-volume MSMPR, and variable-volume fed-batch population
  balances, with independent initial seeds and crystal-bearing or clear feeds.
- Explicit autonomous MSMPR steady solving by dynamic relaxation, checked with
  scaled full-state residuals, physical constraints, product flows and inventory
  balance diagnostics.
- Configured physical problems and prepared setups supported across simulation,
  parameter estimation, ABC, NUTS/MCMC and posterior prediction.
- Named observables and weights, pooled objectives across irregular experiment
  subsets, local observable projections and retained failed-sample diagnostics.
- Preserved initial populations, independent initial concentration, temperature
  profiles and reactor configuration during prepared simulation.
- Public conversion of package-generated SciML solutions into physical results.
- Corrected MoM population/solvent layout, finite-volume gradient signs and OSPRE
  reconstruction, and positive WENO smoothness indicators and boundary units.
- Consistent direct/prepared solver tolerances and solve-local callback ownership,
  including derivative-safe AutoAbstol and dimensional QMOM tolerances.
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
