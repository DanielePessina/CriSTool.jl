# Measurements and data loading

CriSTool represents each experimental run as a `CrystallisationExperiment`:
a typed `NamedTuple` of per-observable containers (`Observable` for
time series with their own grid, `Observable` for final-state scalars
like d43) plus the run conditions (temperature, initial crystal state, `exp_id`). There is
no `Dict{Symbol,Any}` anywhere; adding a new observable (pH, mass, PSD, ...)
means adding a field to the `NamedTuple`, not a new container type.

Tutorial 2 shows the complete path from the bundled workbook to parameter
estimation: [Tutorial 2 — Parameter Estimation](<../examples/Tutorial 2 Parameter Estimation.jl>).

## Loading from the standard sheet format

```julia
using CriSTool

path = joinpath(@__DIR__, "..", "examples", "fake-experimental-dataset.xlsx")
experiments = load_experiments(path, "Unseeded_PE")
```

For a different tabular layout, use the table-driven loader:

```julia
experiments = load_measurements(path, "Unseeded_PE";
    observables = (; concentration = (:Concentration, :Concentration_var),
                   particle_size = (:PS, :PS_var)),
    scalar_observables = (:particle_size,),
    metadata_cols = (; temperature = :Temperature, system = :System),
    initial_crystals_cols = (; mass_concentration = :SeedMass,
                             d43 = :SeedD43,
                             distribution = :SeedDistribution,
                             spread = :SeedSpread),
    temperature_transform = T -> T + 273.15)
```

Each observable may have its own time grid. Series observables retain all
usable rows; scalar observables use the final available row. Extra metadata is
retained in `experiment.metadata`.

`load_experiments(path, sheet_name)` expects a sheet where each
experiment is grouped by `Exp_ID` (Python-importer long format) with columns
such as `Time`, `Concentration`, `Concentration_var`, `Temperature`,
and optional `PS`/`PS_var`. Use the `filters` keyword for arbitrary source
columns and `initial_crystals_cols` to load initial seed characteristics.

The particle size (last timepoint) is stored twice as `d43` and `d50q`
scalar observables: the MoM-based losses compare against `d43`, the
discretised-solver losses against `d50q` (matching the legacy behaviour).

The standard long-format sheet has one row per experiment, time, and
observable sample. The columns used by the default loader are:

| Column | Meaning |
| --- | --- |
| `Exp_ID` | experiment identifier used for grouping |
| `Time` | observation time |
| `Concentration` | measured concentration |
| `Concentration_var` | concentration variance, when replicate measurements exist |
| `PS`, `PS_var` | optional particle-size measurement and variance |
| `Temperature` | experiment temperature |

Use `load_measurements` when the workbook uses different column names, has
additional observables, or needs a temperature transform.

## Balancing repeated measurements and PSD variance

```julia
experiments = repeatmeasurementbalancer(experiments, 3)   # concentration floor
experiments = psd_measurementbalancer(experiments, 8)     # particle-size floor

# Any named observable can be balanced directly.
experiments = balance_variances(experiments; obs = :particle_size,
                                min_rel_std_pc = 8)
```

## Accessing observables

```julia
expt = experiments[1]
expt.observables.concentration.time    # measurement grid (minutes)
expt.observables.concentration.mean    # mean concentration per timepoint
expt.observables.concentration.variance
expt.observables.d43.mean             # final d43 scalar (µm)
expt.observables.d43.variance
initial_concentration(expt)            # first concentration timepoint
expt.temperature                       # Kelvin
expt.initial_crystals
expt.exp_id
```

## Alternative loaders

- `load_experiments_legacy(path, n_sheets)` / `(path, sheet_ids)`: thin
  importers for legacy workbooks organized as `c i`/`q i`/`d i` sheets.
- `load_experiments_legacy_single(path, n_sheets)`: same format, single
  (unreplicated) traces; observables carry `variance = nothing`.

## Resampling

`bootstrap_measurements(experiments, n_bootstrap; seed)` samples each named
observable independently, retaining the first point of each series as its
experiment anchor and preserving metadata. The legacy
`bootstrap_repeatmeasurements` wrapper remains available for the concentration
plus particle-size workflow.

## Loss evaluation

Losses consume every supported observable in the experiment. The legacy loader
still stores particle size under both `d43` and `d50q`; only the solver-relevant
one is included to avoid double-counting.

All loss functions take a `CrystallisationProblem` (kinetics + solver), the
parameter vector, and the experiments:

```julia
problem = CrystallisationProblem(; kinetics_nucleationfunction = nucl_CNT(),
                                 kinetics_growthfunction = growth_empirical(),
                                 solver = MoM())
L = loss(logMLE(), problem, [38.0, 0.6, 1.0, 3.0], experiments)
```
