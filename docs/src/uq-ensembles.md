# Uncertainty and predictions

Joint parameter samples may come from NUTS, ABC, bootstrap fits, a supplied joint
distribution, or named user vectors. Correlations and sample identity are retained.

```julia
joint_draws = ParameterSamples(growth = (
    growth_coefficient = [1e-11, 2e-11, 3e-11],))
predictions = predict(crystal_model, experiments, joint_draws;
    solver = MoM(), saveat = 0.0:600.0:21600.0,
    observables = (:concentration, :d43))
physical_bands = prediction_summary(predictions, :concentration)
```

Unspecified kinetic values come from the reference model. Each column of
`joint_draws.values` is reused across experiments. `predictions.ensembles` holds
one ensemble per run, with sample rows and time columns.
`predict` also accepts explicitly configured problems for future operating
conditions; experiments instead infer initial conditions from time-zero data.
Configured problems must use the reference model's material properties and law
configuration; varying conditions never silently swaps the fitted physics.

Failures retain their rows, IDs and diagnostics. Summaries fail unless exclusion
is explicitly requested with `skip_failed=true`, then report counts. Bounds
reduce invalid candidates but cannot guarantee every numerical solve succeeds.

`parameter_samples(posterior)` adapts NUTS/ABC results;
`parameter_samples(bootstrap_fit_results)` combines fits with identical selections.
`parameter_samples(names, joint_distribution; n_samples, rng)` preserves a
supplied joint distribution rather than sampling independent marginals.

## Future measurements

Physical bands describe kinetic uncertainty. To additionally predict noisy
instrument readings, supply a fixed additive distribution:

```julia
using Distributions, Random
future_readings = measurement_samples(predictions, :concentration;
    noise = Normal(0.0, 0.1), rng = MersenneTwister(42))
```

Alternatively supply `(experiment_id, observable, time) -> Distribution` to make
your interpolation and noise assumptions explicit. Measurement variance alone
does not automatically specify a noise distribution. This operation does not
modify trajectories or fit any noise parameter.
