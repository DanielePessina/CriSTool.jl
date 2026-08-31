# Measurements and data loading

CriSTool represents each experimental run as a `CrystallisationExperiment`:
a typed `NamedTuple` of per-observable `Observable` containers. Every
observable is a time series on its own grid; a single final-state measurement
is a one-element series. The container also stores the run conditions
(temperature, initial crystal state, `exp_id`). There is
no `Dict{Symbol,Any}` anywhere; adding a new observable (pH, mass, PSD, ...)
means adding a field to the `NamedTuple`, not a new container type.

Tutorial 2 shows the complete path from the bundled dataset to parameter
estimation: [Tutorial 2 — Parameter Estimation](<../examples/Tutorial 2 Parameter Estimation.jl>).

## Loading from the standard CSV format

```julia
using CriSTool

path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.csv")
experiments = load_measurements(path)
```

For a different tabular layout, pass an explicit per-observable schema. The
same schema works for CSV and JSON files:

```julia
experiments = load_measurements(path;
    observables = (; concentration = ObservableColumns(
                       mean = :Concentration, variance = :Concentration_var),
                   particle_size = ObservableColumns(
                       time = :ParticleTime, mean = :PS, variance = :PS_var)),
    metadata_cols = (; temperature = :Temperature, system = :System),
    initial_crystals_cols = (; mass_concentration = :SeedMass,
                             d43 = :SeedD43,
                             distribution = :SeedDistribution,
                             spread = :SeedSpread),
    temperature_transform = T -> T + 273.15)
```

`experiments_from_table` is the semantic normalizer. `load_measurements` is a
thin CSV/JSON file adapter over it. Use `ObservableColumns` when the source
uses different column names or when observables have different time columns.

JSON files use an array of records with the same field names as the CSV rows:

```julia
experiments = load_measurements("measurements.json"; format = :json)
```

Each observable retains all usable rows on its own time grid. Extra metadata is
retained in `experiment.metadata`.

`load_measurements(path)` expects a CSV or JSON file in the long format where each
experiment is grouped by `Exp_ID` with columns such as `Time`,
`Concentration`, `Concentration_var`, `Temperature`, and optional
`PS`/`PS_var`. Use the `filters` keyword for arbitrary source columns and
`initial_crystals_cols` to load initial seed characteristics.

When a `PS` column is present, every usable particle-size row (blank rows are
skipped) is stored as the `d43` time series. No sentinel values are required
in the file. Map additional metrics (for example a `d50q` series alongside
`d43`) through `load_measurements` when the loss should compare against both.

The long-format CSV has one row per experiment, time, and observable sample.
The columns used by the default loader are:

| Column | Meaning |
| --- | --- |
| `Exp_ID` | experiment identifier used for grouping |
| `Time` | observation time |
| `Concentration` | measured concentration |
| `Concentration_var` | concentration variance, when replicate measurements exist |
| `PS`, `PS_var` | optional particle-size measurement and variance at each sampled time |
| `Temperature` | experiment temperature in degrees Celsius |

Use `load_measurements` when the file uses different column names, has
additional observables, or needs a different temperature transform.

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
expt.observables.d43.time             # particle-size measurement times
expt.observables.d43.mean             # d43 at those times (µm)
expt.observables.d43.variance         # per-time variance, or nothing
initial_concentration(expt)            # first concentration timepoint
expt.temperature                       # Kelvin
expt.initial_crystals
expt.exp_id
```

## Resampling

`bootstrap_measurements(experiments, n_bootstrap; seed)` samples each named
observable independently, retaining the first point of each series as its
experiment anchor and preserving metadata. The
`bootstrap_repeatmeasurements` wrapper selects the concentration and `d43`
series for the standard workflow.

## Loss evaluation

Losses consume every observable in the experiment: each named observable
contributes an objective term over all of its time points.

All loss functions take a `CrystallisationProblem` (kinetics + solver), the
parameter vector, and the experiments:

```julia
problem = CrystallisationProblem(; kinetics_nucleationfunction = nucl_CNT(),
                                 kinetics_growthfunction = growth_empirical(),
                                 solver = MoM())
L = loss(logMLE(), problem, [38.0, 0.6, 1.0, 3.0], experiments)
```
