# Sensitivity analysis

Sensitivity analysis evaluates how changes in kinetic parameters affect a
chosen simulation output. The package supplies the forward model through
`runsimulation(parameters, problem)`, the low-level flat-vector interface
intended for automatic differentiation and sensitivity work (use `simulate`
with a `CrystallisationModel` for ordinary runs); the sampling and sensitivity estimators come from
GlobalSensitivity.jl and QuasiMonteCarlo.jl.

The lower-level `CriSTool.forwardsensitivity` helper currently covers batch scalar
growth and independent scalar dissolution with MoM or FiniteVol, using the same
numeric SI parameters as `runsimulation(parameters, problem)`. It is qualified because it is not
exported. It rejects length-dependent kinetics and binary aggregation/breakage
explicitly because those source terms need a separate sensitivity
implementation.

[Tutorial 3 — Sensitivity Analysis](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%203%20Sensitivity%20Analysis.jl)
is the full runnable example. It uses the terminal concentration as its
scalar output and computes Sobol and DGSM measures over parameter bounds.

## Build a parameter-to-output map

Configure the problem once from a model and run conditions. The parameter
vector replaces its kinetic blocks in the composite `paramaxis(problem)` order
`[nucleation; growth; dissolution; aggregation; breakage]`; here only
nucleation and growth carry parameters. A sensitivity tool can then call the
forward map with each sampled parameter vector.

```julia
using CriSTool
using GlobalSensitivity
using QuasiMonteCarlo
using Distributions

baseline_model = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0007),
    growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
problem = CrystallisationProblem(baseline_model,
    RunConditions(temperature = 293.15, concentration = 18.0); solver = MoM())
save_grid = 0.0:360.0:21600.0

simulate_terminal_concentration = function (parameters)
    _, solution = runsimulation(parameters, problem; save_idx = save_grid)
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
