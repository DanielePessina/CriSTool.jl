# Saturation models and supersaturation

The solubility of the solute in the solvent is a property of the *system*,
not the solver. `CrystallisationProblem` therefore carries a
`saturation_model::AbstractSaturationModel` field instead of a hardcoded
solubility curve. All ODE right-hand sides compute the supersaturation through
a single dispatch point:

```julia
supersaturation(prob, state, t) = state[end] / saturation_concentration(prob, t)
```

## Built-in models

```julia
# Constant solubility (kg/m³)
prob = CrystallisationProblem(; saturation_model = ConstantSaturation(2.47))

# Polynomial in (T_K - Tref): coeffs[1] + coeffs[2] x + coeffs[3] x² + ...
# (default Tref = 273.15, i.e. temperature in Celsius), Horner evaluation
prob = CrystallisationProblem(;
    saturation_model = PolynomialSaturation(coeffs = [1.0, -0.1, 0.002]))

# Arbitrary function f(T_K, t)
prob = CrystallisationProblem(;
    saturation_model = CallableSaturation((T, t) -> 1.0 + 0.01 * (T - 273.15)))
```

`lysozyme_saturation()` returns the legacy lysozyme solubility polynomial
(`0.3705 + 7.171e-2 ΔT - 1.924e-3 ΔT² + 17.97e-5 ΔT³`, ΔT in °C) and is the
default, so existing models and the gold fixture reproduce identically.

## Querying

```julia
saturation_concentration(prob, t)                       # kg/m³ at time t
saturation_concentration(sm, temp_profile, t)           # model-level
supersaturation(prob, state, t)                         # state[end] / solubility
```

## Notes

- The liquid-phase concentration is the last component of both the MoM and
  discretised solver states, so `supersaturation(prob, state, t)` is
  solver-agnostic.
- The legacy `_get_saturationconcentration` helper has been removed; use the
  model API instead.