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
Runs may be experiments, whose placeholder conditions resolve from time-zero
data, or numeric `RunConditions` for future operating conditions, which need
`saveat`. The model supplies material properties and law configuration for every
run; varying conditions never swaps the fitted physics. This form requires an
explicit `solver`.

`predict(...; parallel = EnsembleThreads())` solves every (experiment, draw) pair
on threads, each task on a private copy of the prepared templates. The default
`EnsembleSerial()` gives the same predictions without threads.

A fit or posterior predicts with the solver, mode and numerical options it was
prepared with:

```julia
fitted_runs = predict(posterior)                        # fitted experiments
future_run = RunConditions(temperature = 288.15, concentration = 20.0)
future_predictions = predict(posterior, [future_run]; saveat = 0.0:600.0:21600.0)
```

Failures retain their rows, IDs and diagnostics. Summaries fail unless exclusion
is explicitly requested with `skip_failed=true`, then report counts. Bounds
reduce invalid candidates but cannot guarantee every numerical solve succeeds.

`posterior.samples` holds the NUTS/ABC draws;
`parameter_samples(bootstrap_fit_results)` combines fits with identical selections.
`parameter_samples(names, joint_distribution; n_samples, rng)` preserves a
supplied joint distribution rather than sampling independent marginals.

## Plotting predictions and posteriors

Plotting is a package extension, so `using CriSTool` does not load Makie.
Load a Makie backend first: `using CairoMakie` enables `plot_predictions` and
`ChainStatsPlots`, and `using PairPlots` additionally enables
`plot_posterior_pairplot`. Calling a plotting function without them raises a
`MethodError` that names the package to load.

`plot_predictions` draws one row per run and one column per observable: the
median over successful draws and a quantile band (5–95 % by default). Passing
the experiments overlays their measured means with ± one standard deviation
error bars, matched by `exp_id`. Time is in minutes and sizes in µm. A single
draw, such as `predict(fit_result)`, plots as a line.

```julia
using CairoMakie, PairPlots
plot_predictions(fitted_runs, experiments; observables = (:concentration, :d43),
    savename = "posterior", savedir = "results")
plot_posterior_pairplot(posterior; savename = "posterior", savedir = "results")
```

`plot_posterior_pairplot` labels each panel with the named parameter path and,
for a posterior, overlays the specification's prior marginals. It also takes
`ParameterSamples` directly, with optional `prior` and `reference` values.
`ChainStatsPlots(posterior.backend_result)` gives NUTS trace and density plots.
Single simulations plot directly, for example `lines(solution, :d43)`; see
[Plotting and diagnostics](api/plotting.md).

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
