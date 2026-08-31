# Bringing your own system

CriSTool's v1 API is system-agnostic: the lysozyme assumptions live in the
defaults (`lysozyme_solubility()`, the default density/shape-factor
constants), not in the machinery. This guide walks a minimal non-lysozyme
system end-to-end — custom solubility, custom kinetics, a custom observable —
using the same pattern as `test/test_generalization.jl`.

## 1. Solubility / supersaturation

Define a `AbstractSolubilityModel` and a `saturation_concentration` method:

```julia
using ComponentArrays

struct LinearSolubility <: CriSTool.AbstractSolubilityModel
    slope::Float64   # kg/m³ per °C
    intercept::Float64
end
function CriSTool.saturation_concentration(sm::LinearSaturation, temp_profile, t)
    return sm.slope * (CriSTool.temperature(temp_profile, t) - 273.15) + sm.intercept
end

problem = CrystallisationProblem(; saturation_model = LinearSaturation(0.25, 2.0), ...)
```

Built-ins: `ConstantSolubility(c)`, `PolynomialSolubility(coeffs, Tref)`,
`CallableSolubility(f)` (see [Solubility models](saturation-models.md)).

Coupled solvent variables are named in `initial_solvent_state`. The default
`solvent_dynamics` updates solute concentration from crystal growth; a custom
callable can return rates for concentration and additional variables:

```julia
solvent_dynamics = (problem, state, t, growth) ->
    (concentration = default_solvent_dynamics(problem, state, t, growth)[1],
     pH = -0.01 * growth)

problem = CrystallisationProblem(; initial_concentration = 25.0,
    initial_solvent_state = (; concentration = 25.0, pH = 7.0),
    solvent_dynamics = solvent_dynamics, ...)
```

The resulting solution exposes `solvent_state`, and `solvent_state(prob, state)`
returns the named values from a numerical state.

For reversible systems, use a signed scalar growth rate in the existing growth
slot: `G > 0` grows crystals, `G < 0` dissolves them, and the built-in
`growth_dissolution()` / `growth_energy_dissolution()` models apply the
equilibrium deadband. Scalar signed rates work with MoM, QMOM, FiniteVol, and
WENO. Length-dependent `growth_dissolution_length()` is a discretised-solver
model; its `growthrate!` method fills a caller-owned mesh buffer.

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

The `Observable` container carries any observable as a time series with vector
`time`/`mean`/`variance`. A single measurement is represented by a one-point
series. Adding an observable (pH, mass, PSD, ...) means adding a field to the
experiment's `NamedTuple` — nothing else:

```julia
expt = CrystallisationExperiment(;
    observables = (;
        concentration = Observable(; time = [0.0, 30.0, 60.0],
                                   mean = [25.0, 20.0, 16.0],
                                   variance = [0.5, 0.5, 0.5]),
        mass = Observable(; time = [0.0, 30.0, 60.0],     # custom observable
                          mean = [0.0, 5.0, 9.0]),
        d43 = Observable(; time = [30.0, 60.0], mean = [6.0, 8.0],
                         variance = [1.0, 1.0]))
    temperature = 293.15, exp_id = 1)
```

The losses read every simulated observable exposed by `observable_values`.
Built-in concentration and size metrics are available automatically; define an
`observable_values` method for a custom solution observable such as pH or mass.
Loaders for the standard CSV long format use
`load_measurements(path)`, while custom column layouts use
`load_measurements(path; observables=...)` / `experiments_from_table(table)` and the
`initial_crystals_cols` mapping.

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
