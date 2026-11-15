# Kinetics: nucleation, growth, aggregation, and breakage

CriSTool uses small structs to represent kinetic models. Each struct
implements a rate function by multiple dispatch in the corresponding
`src/physics/*_rates.jl` file, and
declares a `paramaxis` so its parameters can be accessed by name.

For the shortest working example, see
[Tutorial 1](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%201%20Running%20Simulations.jl). To add a model
from a user script, follow [Tutorial 4](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%204%20Defining%20a%20Custom%20Kinetic.jl)
and the custom-family section below.

Key ideas:
- Each kinetic struct declares a `paramaxis(::T)` method returning a
  ComponentArrays `Axis`. It is the single declaration of the law's
  parameters: their names, order and count. It powers named fields such as
  `p.ln_nucleation_prefactor` and `p.surface_energy` in rate functions.
- Model users bind values by name and never assemble vectors. The low-level
  `remake(problem; p = values)` interface (for automatic differentiation and
  sensitivity work) builds a `ComponentArray` view over a flat parameter vector
  in the order `[nucl; gr; diss; agg; br]`.
- To add a new model: subtype the right `Abstract*Function`, declare
  `paramaxis`, and implement the corresponding rate method.

## Bind a law to named values

```julia
bound_growth = EmpiricalGrowth(growth_coefficient = 1e-9, growth_order = 2.0)
same_growth  = KineticModel(EmpiricalGrowth(); parameters = (
    growth_coefficient = 1e-9, growth_order = 2.0))
energy_growth = ArrheniusGrowth(log10_growth_coefficient = -8.0,
    activation_energy = 60e3, growth_order = 2.0)
```

A built-in law called with no keywords, such as `EmpiricalGrowth()`, is the
unbound law. Called with all of its parameter names it returns a bound
`KineticModel`; a partial set of names is an error. Keywords naming a struct
field, such as `Lref` on `EmpiricalLengthGrowth`, configure the law rather
than bind a parameter. Laws without parameters (`NoNucleation()`,
`NoDissolution()`, `NoAggregation()`, `NoBreakage()`) are bound automatically
when placed in a `CrystallisationModel`, and are its defaults: only `growth`
is required, so a growth-only model omits `nucleation`.

Unlisted parameters in a fitting specification remain fixed, so no separate
fixed kinetic variants exist. For a simple custom
law use `CallableGrowth`, `CallableNucleation` or `CallableLengthGrowth`; see
[Bringing your own system](bring-your-own-system.md) and Tutorial 4.
The dispatch-based interface below remains available for advanced extensions.

## Built-in examples

```julia
using CriSTool

nucl     = CNTNucleation()        # Axis(ln_nucleation_prefactor=1, surface_energy=2)
nucl_emp = EmpiricalNucleation()  # Axis(log10_nucleation_prefactor=1, nucleation_order=2)
growth   = EmpiricalGrowth()# Axis(growth_coefficient=1, growth_order=2)
growth_L = EmpiricalLengthGrowth() # adds a length multiplier; four parameters
growth_E = ArrheniusGrowth()   # log10_growth_coefficient, activation_energy, growth_order
```

Common parameter blocks are:

| Family | Model | Parameter order |
| --- | --- | --- |
| Nucleation | `CNTNucleation()` | `ln_nucleation_prefactor`, `surface_energy` |
| Nucleation | `EmpiricalNucleation()` | `log10_nucleation_prefactor`, `nucleation_order` |
| Nucleation | `ArrheniusNucleation()` | `ln_nucleation_prefactor`, `activation_energy`, `nucleation_order` |
| Growth | `EmpiricalGrowth()` | `growth_coefficient` (m/s), `growth_order` |
| Growth | `EmpiricalLengthGrowth()` | `growth_coefficient` (m/s), `growth_order`, `size_dependence_coefficient`, `size_dependence_exponent` |
| Nucleation | `SecondaryNucleation()` | `ln_nucleation_prefactor`, `activation_energy`, `nucleation_order` |
| Nucleation | `CombinedNucleation(primary, secondary)` | the two component axes nested under `primary` and `secondary` |
| Growth | `ArrheniusGrowth()` | `log10_growth_coefficient`, `activation_energy`, `growth_order` |
| Growth | `BCFGrowth()` | `growth_coefficient`, `activation_temperature` |
| Growth | `BirthSpreadGrowth()` | `growth_coefficient`, `energy_barrier_temperature_squared` |

Use `paramaxis(model)` rather than hard-coding a block length when building a
general fitting or sampling workflow. To fix a parameter, such as the
activation energy of `ArrheniusGrowth`, bind its value and leave it out of the
fitted parameter selection.

`CombinedNucleation` sums any two nucleation laws. Called with two bound laws
it returns the bound combination:

```julia
primary_plus_secondary = CombinedNucleation(
    CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4),
    SecondaryNucleation(ln_nucleation_prefactor = 20.0, activation_energy = 0.0, nucleation_order = 2.0))
```

## Growth and dissolution

Dissolution laws form their own family, `AbstractDissolutionFunction`, and sit
in the model's `dissolution` slot. Growth laws return a nonnegative rate and
dissolution laws a nonpositive one; the solvers add them to obtain the signed
net crystal growth rate. Positive values grow crystals and negative values
dissolve them. The default equilibrium deadband is `|S - 1| ≤ 0.001`, where
`S = c / saturation_concentration(problem, t)`.

```julia
dissolving_model = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4),
    growth = ArrheniusGrowth(log10_growth_coefficient = -8.0, activation_energy = 53e3, growth_order = 2.0),
    dissolution = ArrheniusDissolution(dissolution_coefficient = 1e-8, activation_energy = 40e3,
        dissolution_order = 1.0))
length_dissolution = ArrheniusLengthDissolution() # adds size-dependence terms
length_growth = EmpiricalLengthGrowth()        # empirical growth with size factor
```

`ArrheniusDissolution` uses the dimensionless undersaturation driving force
`(1 - S)^d` and returns a scalar rate. The
length-dependent dissolution model evaluates `(1 + κ * L / Lref)^p` at each
mesh length, with `Lref = 1e-6` m by default; it is supported by `FiniteVol`
and `WENO`. The analogous `EmpiricalLengthGrowth()` model uses

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

`paramaxis` has three public forms (all exported):

```julia
using CriSTool, ComponentArrays

paramaxis(model)                    # single-law axis, e.g. Axis(ln_nucleation_prefactor=1, surface_energy=2)
paramaxis(nucl, gr, diss, agg, br)  # composite axis with one block per slot
paramaxis(prob)                     # composite axis of a CrystallisationProblem (the layout of prob.p)
```

Use the composite form to wrap a flat parameter vector with named slices:

```julia
flat = [38.0, 0.0007, 1e-9 / 60, 3.0]
p    = ComponentArray(flat, paramaxis(CNTNucleation(), EmpiricalGrowth(), NoDissolution(),
                                       NoAggregation(), NoBreakage()))
p.nucl.ln_nucleation_prefactor    # 38.0
p.gr.growth_order                 # 3.0
```

Every law, including a custom one, must declare `paramaxis`; there is no
generic fallback.

## Add a new kinetic family (three pieces)

Three pieces, all in your own user script:

1. A struct subtyping the right `Abstract*Function`.
2. A `paramaxis` method returning a ComponentArrays `Axis`.
3. The rate function, which reads named parameters via
   `named_parameters(model, parameters)`.

Worked example in
[Tutorial 4](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%204%20Defining%20a%20Custom%20Kinetic.jl). Sketched
here for nucleation and growth:

```julia
using CriSTool
using CriSTool: AbstractNucleationFunction, AbstractScalarGrowthFunction,
                named_parameters
import CriSTool: paramaxis, nucleationrate, growthrate
using ComponentArrays

# --- Custom nucleation ---
struct CustomNucleation <: AbstractNucleationFunction end

paramaxis(::CustomNucleation) = ComponentArrays.Axis(ln_prefactor = 1, nucleation_order = 2)

function nucleationrate(nf::CustomNucleation, parameters, prob, state, t)
    p = named_parameters(nf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? exp(p.ln_prefactor) * (S - 1)^p.nucleation_order : 0.0
end

# --- Custom growth ---
struct CustomGrowth <: AbstractScalarGrowthFunction end

paramaxis(::CustomGrowth) = ComponentArrays.Axis(growth_coefficient = 1, growth_order = 2)

function growthrate(gf::CustomGrowth, parameters, prob, state, t)
    p = named_parameters(gf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? p.growth_coefficient * (S - 1)^p.growth_order : 0.0
end
```

Then bind them by name like any built-in law:

```julia
custom_model = CrystallisationModel(LysozymeSystem();
    nucleation = KineticModel(CustomNucleation(); parameters = (
        ln_prefactor = 38.0, nucleation_order = 2.0)),
    growth = KineticModel(CustomGrowth(); parameters = (
        growth_coefficient = 1e-9 / 60, growth_order = 3.0)))
custom_problem = CrystallisationProblem(custom_model,
    RunConditions(temperature = 293.15, concentration = 18.0), (0.0, 28_800.0))
solution = solve(custom_problem, MoM(); saveat = 3600.0)
```

Rate methods receive the solve-time problem, whose state layout depends on the
solver, so leave its argument untyped. Read it through
`supersaturation(prob, state, t)`, `solvent_state(prob, state)`,
`temperature(prob.temp_profile, t)` and `prob.model`.

## Aggregation and breakage

Aggregation and breakage follow the same pattern using
`AbstractAggregationFunction` and `AbstractBreakageFunction` with
`aggregationrate` and `breakagerate` methods. Declare a `paramaxis` on
each new struct. For most runs you can use the provided no-op models:

```julia
agg = NoAggregation()  # paramaxis = Axis() (no parameters)
br  = NoBreakage()     # paramaxis = Axis()
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
| `ConstantAggregation` | `β` | `β = 10^log10_aggregation_coefficient` |
| `SizeSumAggregation` | `β * (L1 + L2)` | `L` is in metres |
| `VolumeSumAggregation` | `β * volume_shape_factor * (L1^3 + L2^3)` | `volume_shape_factor` is the problem volume shape factor |
| `AverageAggregation` | `β * (L1 + L2) / 2` | arithmetic mean of the two lengths |

The aggregation prefactor is dimensional. Its units depend on the selected
kernel, so fitted values must not be moved between kernels without converting
their units. `VolumeSumAggregation` uses physical crystal volume
`v = volume_shape_factor * L^3`.

### Built-in breakage kernels

`EmpiricalBreakage` and `UniformBreakage` use binary breakage with a
daughter-number distribution uniform in crystal volume. For a parent of
length `λ`, the equivalent distribution on the length coordinate is
`b(L | λ) = 6L^2 / λ^3` for `0 ≤ L ≤ λ`. It produces two daughters and
conserves the third length moment, which represents crystal volume.

The selection-rate conventions are different for compatibility with existing
fitted parameters:

| Type | Parent selection rate |
| --- | --- |
| `EmpiricalBreakage` | `Γ(L) = b * L^(3n)` with `L` in metres |
| `UniformBreakage` | `Γ(L) = exp(ln_breakage_coefficient) * (L / 1e-6 m)^(3n)` |

Both models use `breakagerate` to return daughter birth minus parent death.
Fragments outside the finite length mesh are not represented; mesh-refinement
tests are therefore required when checking crystal-volume conservation. The
coordinate and moment conventions follow the size-based crystallisation PBE
formulation in [Zhang et al. (2025)](https://doi.org/10.1016/j.compchemeng.2024.108860).
