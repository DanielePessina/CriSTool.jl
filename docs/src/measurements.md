# Measurements and data loading

CriSTool represents each experimental run as a `CrystallisationExperiment`:
a typed `NamedTuple` of per-observable `Observable` containers. Every
observable is a time series on its own grid; a single final-state measurement
is a one-element series. The container also stores the `RunConditions` the run
used, an integer `exp_id` and metadata. There is
no `Dict{Symbol,Any}` anywhere; adding a new observable (pH, mass, PSD, ...)
means adding a field to the `NamedTuple`, not a new container type.

Tutorial 2 shows the complete path from the bundled dataset to parameter
estimation: [Tutorial 2 — Parameter Estimation](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%202%20Parameter%20Estimation.jl).

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
                             geometric_std = :SeedGeometricStd),
    temperature_transform = identity)
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
`initial_crystals_cols` to load initial seed characteristics. The `conditions`
keyword takes any `RunConditions` keywords shared by every loaded run (see
below); `temperature` may be supplied there only when the table has no
temperature column.
The bundled fixture already stores Kelvin; use an affine transform such as
`T -> T + 273.15` only when adapting a separate Celsius source file.

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
| `Temperature` | experiment temperature in Kelvin |

Numeric measurement values use seconds, kg/m³, and metres. Use
`temperature_transform` only when adapting a source file with a different
temperature convention.

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
expt.observables.concentration.time    # measurement grid (seconds)
expt.observables.concentration.mean    # mean concentration per timepoint
expt.observables.concentration.variance
expt.observables.d43.time             # particle-size measurement times
expt.observables.d43.mean             # d43 at those times (m)
expt.observables.d43.variance         # per-time variance, or nothing
initial_concentration(expt)            # first concentration timepoint
expt.conditions.temperature            # AbstractTemperature (Kelvin)
expt.conditions.crystals               # initial seed description, or nothing
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
contributes an objective term over all of its time points. Evaluate a loss
through a prepared fit, which selects the kinetic parameters by name:

```julia
using Metaheuristics
fit_specification = OptimisationSpec(
    bounds = (growth = (growth_coefficient = (1e-12, 1e-9), growth_order = (1.0, 4.0)),),
    algorithm = DE(N = 16), loss = MeanAbsoluteError())
prepared_fit = prepare_fit(crystal_model, experiments, fit_specification; solver = MoM())
objective_value = loss(prepared_fit, [1e-9 / 60, 3.0])
```

The vector follows the order of the selected names. `loss_terms` reports the
per-observable objectives.

## Initial conditions from observations

Every experiment carries `RunConditions`. A `FromObservation(:name)` value is
read from the experiment's own time-zero observation of `:name`. The initial
concentration defaults to `FromObservation(:concentration)`, so transient
fitting requires an exact sample at time zero; a late first measurement fails
preparation. Seeds use the same placeholder in any numeric field:

```julia
seeded_experiment = CrystallisationExperiment(
    observables = measured_observables, exp_id = 1,
    conditions = RunConditions(temperature = 293.15,
        crystals = LogNormalInitialCrystals(
            mass_concentration = FromObservation(:seed_mass),
            d43 = FromObservation(:d43), geometric_std = 1.2)))
```

Placeholder observables must have time-zero values. A width may itself be a
placeholder. `DistributionInitialCrystals(length_distribution;
mass_concentration = FromObservation(:seed_mass))` reads only the mass. No seed
means an unseeded run. The time-zero samples consumed by placeholders are
excluded from scoring.

`load_measurements` and `experiments_from_table` accept the same settings through
`conditions = (...)`, a `NamedTuple` of `RunConditions` keywords applied to every
loaded run:

```julia
experiments = load_measurements(path;
    conditions = (crystals = LogNormalInitialCrystals(
        mass_concentration = 0.1, d43 = 20e-6, geometric_std = 1.2),))
```

A placeholder in shared conditions is resolved separately for each run, so per-run
seed values only need the named observable mapped through `observables`.
Seeds come either from `initial_crystals_cols` or from `conditions`, not both.
Bootstrap and variance balancing retain conditions and unselected observables.
