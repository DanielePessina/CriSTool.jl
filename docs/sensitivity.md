# Sensitivity analysis

Sensitivity analysis evaluates how changes in kinetic parameters affect a
chosen simulation output. The package supplies the forward model through
`runsimulation`; the sampling and sensitivity estimators come from
GlobalSensitivity.jl and QuasiMonteCarlo.jl.

The lower-level `forwardsensitivity` helper currently covers scalar growth and
independent scalar dissolution with MoM or FiniteVol, using the same numeric SI
parameters as `runsimulation`. It rejects length-dependent kinetics and binary
aggregation/breakage explicitly because those source terms need a separate
sensitivity implementation.

[Tutorial 3 — Sensitivity Analysis](<../examples/Tutorial 3 Sensitivity Analysis.jl>)
is the full runnable example. It uses the terminal concentration as its
scalar output and computes Sobol and DGSM measures over parameter bounds.

## Build a parameter-to-output map

Keep the parameter order identical to the order used by `runsimulation`:
`[nucleation; growth; aggregation; breakage]`. A sensitivity tool can then
call the forward map with each sampled parameter vector.

```julia
using CriSTool
using GlobalSensitivity
using QuasiMonteCarlo
using Distributions

nucl, gr = nucl_CNT(), growth_empirical()
agg, br  = noaggregation(), nobreakage()
solver   = MoM()
save_grid = 0.0:360.0:21600.0

simulate_terminal_concentration = function (parameters)
    _, solution = runsimulation(
        parameters;
        nucl = nucl,
        gr = gr,
        agg = agg,
        br = br,
        solver = solver,
        initial_concentration = 18.0,
        save_idx = save_grid,
    )
    return solution.concentration[end]
end
```

## Sobol indices

Generate two quasi-random design matrices over the bounds and pass them to
GlobalSensitivity's `gsa` function:

```julia
baseline = [38.0, 0.0007, 1e-9 / 60, 3.0]
lower = 0.5 .* baseline
upper = 1.5 .* baseline
A, B = QuasiMonteCarlo.generate_design_matrices(
    256, lower, upper, SobolSample())

sobol = gsa(
    simulate_terminal_concentration,
    Sobol(order = [0, 1, 2], nboot = 32, conf_level = 0.95),
    A,
    B;
    Ei_estimator = :Sobol2007,
)

sobol.S1  # first-order indices
sobol.ST  # total-order indices
```

`S1` measures the first-order contribution of a parameter. `ST` includes its
interactions with the other sampled parameters. Increase the design size and
bootstrap count when the reported indices need tighter numerical precision.

## DGSM indices

DGSM uses a prior for each parameter instead of the two-design-matrix Sobol
interface:

```julia
priors = [Uniform(lower[i], upper[i]) for i in eachindex(lower)]
dgsm = gsa(simulate_terminal_concentration, DGSM(), priors; samples = 256)
```

The complete sampling settings and reproducible seed are kept in Tutorial 3.
