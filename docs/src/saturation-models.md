# Solubility models and supersaturation

The solubility of the solute in the solvent is a property of the *system*,
not the solver. `CrystallisationProblem` therefore carries a
`saturation_model::AbstractSolubilityModel` field instead of a hardcoded
solubility curve. All ODE right-hand sides compute supersaturation through the
single dispatch point `supersaturation(prob, state, t)`, which reads the named
`:concentration` solvent variable and divides it by the solubility at `t`:

```julia
supersaturation(prob, state, t)
```

## Built-in models

```julia
using CriSTool

# Constant solubility (kg/m³)
prob = CrystallisationProblem(; saturation_model = ConstantSolubility(2.47))

# Polynomial in (T_K - Tref): coeffs[1] + coeffs[2] x + coeffs[3] x² + ...
# (default Tref = 273.15 K; temperature differences are equivalent to Celsius
# differences), Horner evaluation
prob = CrystallisationProblem(;
    saturation_model = PolynomialSolubility(coeffs = [1.0, -0.1, 0.002]))

# Arbitrary function f(T_K, t)
prob = CrystallisationProblem(;
    saturation_model = CallableSolubility((T, t) -> 1.0 + 0.01 * (T - 273.15)))
```

`lysozyme_solubility()` returns the legacy lysozyme solubility polynomial
(`0.3705 + 7.171e-2 ΔT - 1.924e-3 ΔT² + 17.97e-5 ΔT³`, with ΔT in K) and is the
default. The gold fixture records the converted SI trajectories.

For a reversible seeded run using a constant saturation value, see
[Tutorial 6 — Dissolution](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%206%20Dissolution.jl).

## Querying

```julia
saturation_concentration(prob, t)                       # kg/m³ at time t
saturation_concentration(sm, temp_profile, t)           # model-level
supersaturation(prob, state, t)                         # named concentration / solubility
```

## Notes

- The solvent-phase concentration is the named `:concentration` component of
  both the MoM and
  discretised solver states, so `supersaturation(prob, state, t)` is
  solver-agnostic.
- Additional named solvent variables do not change the saturation API; the
  concentration component is located by name rather than by a fixed state
  index.
