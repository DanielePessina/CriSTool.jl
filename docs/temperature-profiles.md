# Temperature profiles

Crystallisation problems store temperature as a `temp_profile`, which is
an `AbstractTemperature`. The default is a constant 20 °C (`293.15` K).

You can pass a custom profile to `runsimulation` via `temp_profile`. For
side-by-side runs across constant, ramp, and arbitrary `T(t)` profiles,
see [Tutorial 1 — Running Simulations](<../examples/Tutorial 1 Running Simulations.jl>).

## Built-in profiles

`AbstractTemperature` subtypes (defined in `src/core/temperature_types.jl`):

- `ConstantTemperature(T)` — fixed temperature (Kelvin).
- `LinearTemperature(T0, slope)` — `T(t) = T0 + slope * t`.
- `RampTemperature(...)` — piecewise-linear ramp/hold profile.
- `CallableTemperature(f)` — wraps any `f(t)` so you can use an
  arbitrary closure as the temperature profile.

```julia
using CriSTool

# Constant temperature (Kelvin)
const_T = ConstantTemperature(293.15)

# Linear cooling profile (Kelvin, K per second)
linear_T = LinearTemperature(293.15, -0.01)

# Arbitrary T(t) via a closure
periodic_T = CallableTemperature(t -> 293.15 + 2.0 * sin(2π * t / 600))

problem, solution = runsimulation(
    [38.0, 0.7, 1.0, 3.0];
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    solver = MoM(),
    initial_concentration = 18.0,
    temp_profile = linear_T,
    save_idx = 0.0:60.0:480.0,
)
```

## Custom profile

Define a new type and implement `temperature(profile, t)`:

```julia
struct StepTemperature <: CriSTool.AbstractTemperature
    T1::Float64
    T2::Float64
    t_switch::Float64
end

CriSTool.temperature(p::StepTemperature, t) = t < p.t_switch ? p.T1 : p.T2

profile = StepTemperature(293.15, 288.15, 120.0)

problem, solution = runsimulation(
    [38.0, 0.7, 1.0, 3.0];
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    solver = MoM(),
    initial_concentration = 18.0,
    temp_profile = profile,
    save_idx = 0.0:60.0:480.0,
)
```
