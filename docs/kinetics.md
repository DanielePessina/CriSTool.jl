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
