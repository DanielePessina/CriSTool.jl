# Measurements and data loading

CriSTool provides helper constructors for experimental measurements,
usually loaded from Excel files via `makerepeatmeasurements`.

## Synthetic dataset (shipped with the package)

A 5-experiment synthetic workbook lives at
`CriSTool/examples/fake-experimental-dataset.xlsx` (sheet `Unseeded_PE`).
It has irregular sampling and heteroscedastic 3 % / 8 % noise on
concentration / particle size, generated from the same forward model
used in Tutorial 5. Tutorials 2 and 5 load it directly:

```julia
using CriSTool

path = joinpath(@__DIR__, "fake-experimental-dataset.xlsx")
measurements = makerepeatmeasurements(path, "Unseeded_PE", [0.0])
```

The third argument is the loading filter — pass either a scalar (`0.0`)
or a vector of loadings (`[0.0]`, `[0.0, 0.5]`) to combine multiple
loading levels into one `Vector{<:CrystallisationRepeatMeasurements}`.

## Common pattern (with a real workbook)

```julia
using CriSTool

measurements = makerepeatmeasurements(
    "experimental_dataset.xlsx",
    "Unseeded_PE",
    0.0,
    (nothing, 22),
)

# Balance repeated measurements and PSD variance
measurements = repeatmeasurementbalancer(measurements, 3)
measurements = psd_measurementbalancer(measurements, 8)
```

This form expects a sheet where each experiment is grouped by `Exp_ID`
with columns such as `Time`, `Concentration`, `Temperature`, and `Loading`.

## Alternative loaders

If your workbook is organized by multiple sheets named `c i`, `q i`, `d i`,
use the simpler overload:

```julia
measurements = makerepeatmeasurements("my_data.xlsx", 3)
```

## Measurement types

- `CrystallisationRepeatMeasurements`: mean and variance per timepoint,
  used for parameter estimation and ABCDE.
- `CrystallisationSingleMeasurements`: single-trace data.

Most parameter estimation functions expect a
`Vector{<:AbstractMeasurements}`.
