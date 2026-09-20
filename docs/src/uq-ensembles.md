# Ensembles and uncertainty

CriSTool includes ensemble utilities in `src/uncertainty/ensembles.jl`.
The main function is `run_ensemble`, which runs many forward simulations
and returns ensemble solution objects.

## Example: ensemble from a prior distribution

```julia
using CriSTool
using Distributions

path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.csv")
measurements = load_measurements(path)

nucl_f = nucl_CNT()
growth_f = growth_energy()
solver = MoM()

prior = Distributions.product_distribution([
    Uniform(10.0, 65.0),
    Uniform(0.00015, 0.0025),
    Uniform(-20.0, 0.0),
    Uniform(1.0, 3.5),
])

ensemble = run_ensemble(prior, measurements,
                        nucl_f, growth_f, noaggregation(), nobreakage(), solver;
                        n_samples = 256,
                        time_idx = 0.0:300.0:18000.0)

# Access ensemble statistics for the first measurement
ens1 = ensemble[1]
@show ens1.concentration_mean[end]
@show ens1.d43_mean[end]
```

Set `solver = QMOM(nquadrature = 3)` to run the same ensemble workflow with a
moment/quadrature solver. QMOM ensemble results use the moment-based
`EnsembleMoMSolution` container, so `d43`, `d32`, and concentration summaries
are available in the same fields. QMOM does not produce a `d50q` quantile.
`DQMOM(nquadrature = 3)` can be used when every measurement supplies a
nonempty `initial_crystals` population; its seeded-only contract is enforced
by the underlying simulation.

## Example: ensemble from an explicit sample matrix

```julia
using CriSTool
using Distributions

samples = rand(Distributions.product_distribution([
    Uniform(10.0, 65.0),
    Uniform(0.00015, 0.0025),
    Uniform(-20.0, 0.0),
    Uniform(1.0, 3.5),
]), 128)

ensemble = run_ensemble(samples, measurements,
                        nucl_CNT(), growth_energy(), noaggregation(), nobreakage(), MoM())
```

## Notes

- The ensemble uses each measurement's `temperature` and `initial_crystals`
  fields when running simulations.
- Pass `diss = growth_dissolution()` (and include its SI parameter block in the
  samples) to propagate an independent dissolution model. Its block follows
  the growth block, before aggregation and breakage.
- `use_measurement_time = true` (the default) spans each experiment's observed
  time range with `length(time_idx)` saved points. Set it to `false` to use
  `time_idx` exactly.
- The returned objects are `EnsembleMoMSolution` for MoM/QMOM/DQMOM and
  `EnsembleFVSolution` for FiniteVol/WENO.
