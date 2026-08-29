# Ensembles and uncertainty

CriSTool includes ensemble utilities in `src/uncertainty/ensembles.jl`.
The main function is `run_ensemble`, which runs many forward simulations
and returns ensemble solution objects.

## Example: ensemble from a prior distribution

```julia
using CriSTool
using Distributions

path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.xlsx")
measurements = load_experiments(path, "Unseeded_PE", 0.0)

nucl_f = nucl_CNT()
growth_f = growth_energy()
solver = MoM()

prior = Distributions.product_distribution([
    Uniform(10.0, 65.0),
    Uniform(0.15, 2.5),
    Uniform(-10.0, 10.0),
    Uniform(1.0, 3.5),
])

ensemble = run_ensemble(prior, measurements,
                        nucl_f, growth_f, noaggregation(), nobreakage(), solver;
                        n_samples = 256,
                        time_idx = 0.0:5.0:300.0)

# Access ensemble statistics for the first measurement
ens1 = ensemble[1]
@show ens1.concentration_mean[end]
@show ens1.d43_mean[end]
```

## Example: ensemble from an explicit sample matrix

```julia
using CriSTool
using Distributions

samples = rand(Distributions.product_distribution([
    Uniform(10.0, 65.0),
    Uniform(0.15, 2.5),
    Uniform(-10.0, 10.0),
    Uniform(1.0, 3.5),
]), 128)

ensemble = run_ensemble(samples, measurements,
                        nucl_CNT(), growth_energy(), noaggregation(), nobreakage(), MoM())
```

## Notes

- The ensemble uses each measurement's `temperature` and `loading` fields
  when running simulations.
- The returned objects are `EnsembleMoMSolution` or `EnsembleFVSolution`
  depending on the solver.
