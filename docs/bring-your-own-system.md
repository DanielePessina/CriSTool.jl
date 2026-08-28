# Bringing your own system

CriSTool's v1 API is system-agnostic: the lysozyme assumptions live in the
defaults (`lysozyme_saturation()`, the default density/shape-factor
constants), not in the machinery. This guide walks a minimal non-lysozyme
system end-to-end — custom solubility, custom kinetics, a custom observable —
using the same pattern as `test/test_generalization.jl`.

## 1. Solubility / supersaturation

Define a `AbstractSaturationModel` and a `saturation_concentration` method:

```julia
struct LinearSaturation <: CriSTool.AbstractSaturationModel
    slope::Float64   # kg/m³ per °C
    intercept::Float64
end
function CriSTool.saturation_concentration(sm::LinearSaturation, temp_profile, t)
    return sm.slope * (temperature(temp_profile, t) - 273.15) + sm.intercept
end

problem = CrystallisationProblem(; saturation_model = LinearSaturation(0.25, 2.0), ...)
```

Built-ins: `ConstantSaturation(c)`, `PolynomialSaturation(coeffs, Tref)`,
`CallableSaturation(f)` (see [Saturation models](saturation-models.md)).

## 2. Custom kinetics (Tutorial-4 pattern)

Three pieces: a struct subtyping the right `Abstract*` family, a `paramaxis`
method, and a rate method with the standard signature
`(fn, params, prob, state, t)`:

```julia
using CriSTool: AbstractFPScalarGrowthFunction, _named_params
import CriSTool: paramaxis, growthrate

struct growth_custom <: AbstractFPScalarGrowthFunction
    nparams::Int64
    string::String
    symbols::Vector{Symbol}
end
growth_custom() = growth_custom(2, "Custom Gr", [:Ag, :g])

paramaxis(::growth_custom) = ComponentArrays.Axis(Ag = 1, g = 2)

function growthrate(gf::growth_custom, parameters,
                    prob::CrystallisationProblem, state, t)
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)   # state[end] / saturation_concentration(prob, t)
    return S > 1.001 ? p.Ag * 1e-9 * (S - 1)^p.g : 0.0
end
```

Nucleation, aggregation and breakage follow the same shape
(`nucleationrate`, `aggregationrate`, `breakagerate`).

## 3. Measurements with a custom observable

The `Observable` container carries any observable by shape: a time series has
vector `time`/`mean`/`variance`, a final-state scalar has scalar `time`/`mean`.
Adding an observable (pH, mass, PSD, ...) means adding a field to the
experiment's `NamedTuple` — nothing else:

```julia
expt = CrystallisationExperiment(;
    observables = (;
        concentration = Observable(; time = [0.0, 30.0, 60.0],
                                   mean = [25.0, 20.0, 16.0],
                                   variance = [0.5, 0.5, 0.5]),
        mass = Observable(; time = [0.0, 30.0, 60.0],     # custom observable
                          mean = [0.0, 5.0, 9.0]),
        d43 = Observable(; mean = 8.0, variance = 1.0),   # size observables
        d50q = Observable(; mean = 8.0, variance = 1.0)), # used by the losses
    temperature = 293.15, loading = 0.0, exp_id = 1)
```

The losses read `concentration` (series) and `d43`/`d50q` (scalars); extra
observables ride along untouched. Loaders for the Excel long format:
`load_experiments(path, sheet, loading)`.

## 4. Simulation, loss, estimation

```julia
_, sol = runsimulation([8.0, 2.0, 1.0, 2.0];
                       nucl = nucl_empirical(), gr = growth_custom(),
                       agg = noaggregation(), br = nobreakage(),
                       solver = MoM(), save_idx = [0.0, 30.0, 60.0, 120.0],
                       saturation_model = LinearSaturation(0.25, 2.0))

# one-off loss
L = loss(logMLE(), problem, [8.0, 2.0, 1.0, 2.0], [expt])

# optimisation loop: prepare once, evaluate via remake
setup = prepare_loss(problem, [expt])
L  = loss(logMLE(), setup, [8.0, 2.0, 1.0, 2.0])
```

`PE_Routine`, `run_abc` and the MCMC tutorials consume the same types — swap
in your system without touching the package.