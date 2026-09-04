# Kinetics: nucleation and growth

CriSTool uses small structs to represent kinetic models. Each struct
implements a rate function by multiple dispatch in the corresponding
`src/physics/*_rates.jl` file, and
declares a `paramaxis` so its parameters can be accessed by name.

For the shortest working example, see
[Tutorial 1](<../examples/Tutorial 1 Running Simulations.jl>). To add a model
from a user script, follow [Tutorial 4](<../examples/Tutorial 4 Defining a Custom Kinetic.jl>)
and the custom-family section below.

Key ideas:
- Each kinetic struct has `nparams`, `string`, and often `symbols`.
- Each struct also declares a `paramaxis(::T)` method returning a
  ComponentArrays `Axis` — this is what powers named fields such as
  `p.ln_nucleation_prefactor` and `p.surface_energy` in rate functions.
- `runsimulation` builds a `ComponentArray` view over the flat
  parameter vector (`[nucl; gr; agg; br]`) so each rate function sees
  its own named slice.
- To add a new model: subtype the right `Abstract*Function`, declare
  `paramaxis`, and implement the corresponding rate method.

## Built-in examples

```julia
nucl     = nucl_CNT()        # nparams = 2, axis Axis(ln_nucleation_prefactor=1, surface_energy=2)
nucl_emp = nucl_empirical()  # nparams = 2, axis Axis(log10_nucleation_prefactor=1, nucleation_order=2)
growth   = growth_empirical()# nparams = 2, axis Axis(growth_coefficient=1, growth_order=2)
growth_L = growth_empirical_length() # adds a length multiplier; nparams = 4
growth_E = growth_energy()   # log10_growth_coefficient, growth_order
```

Common parameter blocks are:

| Family | Model | Parameter order |
| --- | --- | --- |
| Nucleation | `nucl_CNT()` | `ln_nucleation_prefactor`, `surface_energy` |
| Nucleation | `nucl_empirical()` | `log10_nucleation_prefactor`, `nucleation_order` |
| Nucleation | `nucl_empirical_energy()` | `ln_nucleation_prefactor`, `activation_energy`, `nucleation_order` |
| Growth | `growth_empirical()` | `growth_coefficient` (m/s), `growth_order` |
| Growth | `growth_empirical_length()` | `growth_coefficient` (m/s), `growth_order`, `size_dependence_coefficient`, `size_dependence_exponent` |
| Growth | `growth_energy()` | `log10_growth_coefficient`, `growth_order`; activation energy is stored on the model |
| Growth | `growth_energy_est()` | `log10_growth_coefficient`, `activation_energy`, `growth_order` |
| Growth | `growth_BCF()` | `growth_coefficient`, `activation_temperature` |
| Growth | `growth_BpS()` | `growth_coefficient`, `energy_barrier_temperature_squared` |

Use `nparams` and `paramaxis(model)` rather than hard-coding a block length
when building a general fitting or sampling workflow. Fixed variants such as
`growth_empirical_fixed` and `nucl_empirical_fixed` embed their parameters in
the model and contribute no free parameter slots.

## Signed growth and dissolution

The built-in dissolution models use the same `growthrate` interface as growth,
but return a signed crystal growth rate: positive values grow crystals and
negative values dissolve them. The default equilibrium deadband is
`|S - 1| ≤ 0.001`, where `S = c / saturation_concentration(problem, t)`.

```julia
dissolution = growth_dissolution()          # coefficient, activation energy, order
combined    = growth_energy_dissolution()   # growth block + dissolution block
length_dissolution = growth_dissolution_length() # adds size-dependence terms
length_growth = growth_empirical_length()        # empirical growth with size factor
```

`growth_dissolution` uses the dimensionless undersaturation driving force
`(1 - S)^d` and returns a scalar rate. `growth_energy_dissolution` uses the
growth law above the deadband and the dissolution law below it. The
length-dependent dissolution model evaluates `(1 + κ * L / Lref)^p` at each
mesh length, with `Lref = 1e-6` m by default; it is supported by `FiniteVol`
and `WENO`. The analogous `growth_empirical_length()` model uses

```math
G(L,S) = k_g(S-1)^{g}\left[1+\kappa_g\frac{L}{L_{ref}}\right]^{p_g},
\qquad S > 1.001,
```

and returns zero in the equilibrium deadband. Its `Lref` is a fixed model
field (1 µm by default), not a fifth fit parameter. It is available to
discretised solvers and to seeded DQMOM, which evaluates the rate at each
quadrature node. QMOM and MoM retain scalar-rate restrictions.

The scalar models can be used by all moment and discretised solvers. For a
length-dependent model, call `growthrate!` from a solver-owned scratch buffer
when writing a custom discretised RHS; the non-mutating `growthrate` convenience
method allocates a vector for standalone inspection.

## The `paramaxis` API

`paramaxis` has three overloads (all exported):

```julia
paramaxis(model)              # single-family axis, e.g. Axis(ln_nucleation_prefactor=1, surface_energy=2)
paramaxis(nucl, gr, agg, br)  # four-family axis without independent dissolution
paramaxis(nucl, gr, diss, agg, br)  # canonical axis with dissolution
paramaxis(prob)               # composite axis read off a CrystallisationProblem
```

Use the composite form to wrap a flat parameter vector with named slices:

```julia
using ComponentArrays
flat = [38.0, 0.0007, 1e-9 / 60, 3.0]
p    = ComponentArray(flat, paramaxis(nucl_CNT(), growth_empirical(),
                                       noaggregation(), nobreakage()))
p.nucl.ln_nucleation_prefactor    # 38.0
p.gr.growth_order                 # 3.0
```

User-defined variants that have not declared a custom `paramaxis` (or a
rate-function-less placeholder) fall back to a generic `Axis(θ1=1, θ2=2, ...)`
with one entry per `nparams` slot. All built-in families, including
`growth_energy_dissolution`, declare descriptive axes.

## Add a new kinetic family (three pieces)

Three pieces, all in your own user script:

1. A struct subtyping the right `Abstract*Function`.
2. A `paramaxis` method returning a ComponentArrays `Axis`.
3. The rate function, which reads named parameters via
   `_named_params(model, parameters)`.

Worked example in
[Tutorial 4](<../examples/Tutorial 4 Defining a Custom Kinetic.jl>). Sketched
here for nucleation and growth:

```julia
using CriSTool
using CriSTool: AbstractFPNucleationFunction, AbstractFPScalarGrowthFunction,
                _named_params
import CriSTool: paramaxis, nucleationrate, growthrate
using ComponentArrays

# --- Custom nucleation ---
struct nucl_custom <: AbstractFPNucleationFunction
    nparams::Int64
    string::String
    symbols::Vector{Symbol}
end
nucl_custom() = nucl_custom(2, "Custom Nu", [:ln_prefactor, :nucleation_order])

paramaxis(::nucl_custom) = ComponentArrays.Axis(ln_prefactor = 1, nucleation_order = 2)

function nucleationrate(nf::nucl_custom, parameters,
                        prob::CrystallisationProblem, state, t)
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? exp(p.ln_prefactor) * (S - 1)^p.nucleation_order : 0.0
end

# --- Custom growth ---
struct growth_custom <: AbstractFPScalarGrowthFunction
    nparams::Int64
    string::String
    symbols::Vector{Symbol}
end
growth_custom() = growth_custom(2, "Custom Gr", [:growth_coefficient, :growth_order])

paramaxis(::growth_custom) = ComponentArrays.Axis(growth_coefficient = 1, growth_order = 2)

function growthrate(gf::growth_custom, parameters,
                    prob::CrystallisationProblem, state, t)
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? p.growth_coefficient * (S - 1)^p.growth_order : 0.0
end
```

Then use them as usual:

```julia
params = [38.0, 2.0, 1e-9 / 60, 3.0]
problem, solution = runsimulation(
    params;
    nucl = nucl_custom(),
    gr   = growth_custom(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)
```

## Aggregation and breakage

Aggregation and breakage follow the same pattern using
`AbstractAggregationFunction` and `AbstractBreakageFunction` with
`aggregationrate` and `breakagerate` methods. Declare a `paramaxis` on
each new struct. For most runs you can use the provided no-op models:

```julia
agg = noaggregation()  # paramaxis = Axis() (no parameters)
br  = nobreakage()     # paramaxis = Axis()
```

### Built-in aggregation kernels

The finite-volume population balance uses crystal length `L` as its grid
coordinate, but aggregation adds crystal volume. A collision between lengths
`L1` and `L2` therefore produces a particle with representative volume
`L1^3 + L2^3`. The aggregation operator uses the standard binary
Smoluchowski convention: unequal-size pairs are counted once, and equal-size
pairs carry a factor of one half.

The available kernels are:

| Type | Kernel | `log10_aggregation_coefficient` convention |
| --- | --- | --- |
| `aggr_scalar` | `β` | `β = 10^log10_aggregation_coefficient` |
| `aggr_linear` | `β * (L1 + L2)` | `L` is in metres |
| `aggr_linearvol` | `β * volume_shape_factor * (L1^3 + L2^3)` | `volume_shape_factor` is the problem volume shape factor |
| `aggr_avg` | `β * (L1 + L2) / 2` | arithmetic mean of the two lengths |

The aggregation prefactor is dimensional. Its units depend on the selected
kernel, so fitted values must not be moved between kernels without converting
their units. `aggr_linearvol` uses physical crystal volume
`v = volume_shape_factor * L^3`.

### Built-in breakage kernels

`breakage_empirical` and `breakage_uniform` use binary breakage with a
daughter-number distribution uniform in crystal volume. For a parent of
length `λ`, the equivalent distribution on the length coordinate is
`b(L | λ) = 6L^2 / λ^3` for `0 ≤ L ≤ λ`. It produces two daughters and
conserves the third length moment, which represents crystal volume.

The selection-rate conventions are different for compatibility with existing
fitted parameters:

| Type | Parent selection rate |
| --- | --- |
| `breakage_empirical` | `Γ(L) = b * L^(3n)` with `L` in metres |
| `breakage_uniform` | `Γ(L) = exp(ln_breakage_coefficient) * (L / 1e-6 m)^(3n)` |

Both models use `breakagerate` to return daughter birth minus parent death.
Fragments outside the finite length mesh are not represented; mesh-refinement
tests are therefore required when checking crystal-volume conservation. The
coordinate and moment conventions follow the size-based crystallisation PBE
formulation in [Zhang et al. (2025)](https://doi.org/10.1016/j.compchemeng.2024.108860).
