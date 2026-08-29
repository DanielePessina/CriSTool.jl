# Kinetics: nucleation and growth

CriSTool uses small structs to represent kinetic models. Each struct
implements a rate function by multiple dispatch in `Models.jl`, and
declares a `paramaxis` so its parameters can be accessed by name.

Key ideas:
- Each kinetic struct has `nparams`, `string`, and often `symbols`.
- Each struct also declares a `paramaxis(::T)` method returning a
  ComponentArrays `Axis` — this is what powers `p.Aj`, `p.γ`, etc. in
  rate functions.
- `runsimulation` builds a `ComponentArray` view over the flat
  parameter vector (`[nucl; gr; agg; br]`) so each rate function sees
  its own named slice.
- To add a new model: subtype the right `Abstract*Function`, declare
  `paramaxis`, and implement the corresponding rate method.

## Built-in examples

```julia
nucl     = nucl_CNT()        # nparams = 2, axis Axis(Aj=1, γ=2)
nucl_emp = nucl_empirical()  # nparams = 2, axis Axis(Aj=1, j=2)
growth   = growth_empirical()# nparams = 2, axis Axis(Ag=1, g=2)
growth_E = growth_energy()   # nparams = 2, axis Axis(Ag=1, g=2) (with activation energy)
```

## The `paramaxis` API

`paramaxis` has three overloads (all exported):

```julia
paramaxis(model)              # single-family axis, e.g. Axis(Aj=1, γ=2)
paramaxis(nucl, gr, agg, br)  # composite top-level axis spanning all four families
paramaxis(prob)               # composite axis read off a CrystallisationProblem
```

Use the composite form to wrap a flat parameter vector with named slices:

```julia
using ComponentArrays
flat = [38.0, 0.7, 1.0, 3.0]
p    = ComponentArray(flat, paramaxis(nucl_CNT(), growth_empirical(),
                                       noaggregation(), nobreakage()))
p.nucl.Aj    # 38.0
p.gr.g       # 3.0
```

Variants that have not declared a custom `paramaxis` (the multi-loading
nucleation/growth families, the delegating `growth_energy_dissolution`,
and any rate-function-less placeholder) fall back to a generic
`Axis(θ1=1, θ2=2, ...)` with one entry per `nparams` slot.

## Add a new kinetic family (three pieces)

Three pieces, all in your own user script:

1. A struct subtyping the right `Abstract*Function`.
2. A `paramaxis` method returning a ComponentArrays `Axis`.
3. The rate function, which reads named parameters via
   `_named_params(model, parameters)`.

Worked example in
`CriSTool/examples/Tutorial 4 Defining a Custom Kinetic.jl`. Sketched
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
nucl_custom() = nucl_custom(2, "Custom Nu", [:A, :b])

paramaxis(::nucl_custom) = ComponentArrays.Axis(A = 1, b = 2)

function nucleationrate(nf::nucl_custom, parameters,
                        prob::CrystallisationProblem, state, t)
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? (60 * exp(p.A)) * (S - 1)^p.b : 0.0
end

# --- Custom growth ---
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
    S = supersaturation(prob, state, t)
    return S > 1.001 ? p.Ag * (S - 1)^p.g : 0.0
end
```

Then use them as usual:

```julia
params = [38.0, 1.5, 1.0, 3.0]
problem, solution = runsimulation(
    params;
    nucl = nucl_custom(),
    gr   = growth_custom(),
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:60.0:480.0,
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

| Type | Kernel | `logβ` prefactor convention |
| --- | --- | --- |
| `aggr_scalar` | `β` | `β = 10^logβ` |
| `aggr_linear` | `β * (L1 + L2)` | `L` is in metres |
| `aggr_linearvol` | `β * kv * (L1^3 + L2^3)` | `kv` is the problem volume shape factor |
| `aggr_avg` | `β * (L1 + L2) / 2` | arithmetic mean of the two lengths |

The aggregation prefactor is dimensional. Its units depend on the selected
kernel, so fitted values must not be moved between kernels without converting
their units. `aggr_linearvol` uses physical crystal volume
`v = kv * L^3`.

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
| `breakage_uniform` | `Γ(L) = exp(logb) * (L / 1μm)^(3n)` |

Both models use `breakagerate` to return daughter birth minus parent death.
Fragments outside the finite length mesh are not represented; mesh-refinement
tests are therefore required when checking crystal-volume conservation. The
coordinate and moment conventions follow the size-based crystallisation PBE
formulation in [Zhang et al. (2025)](https://doi.org/10.1016/j.compchemeng.2024.108860).
