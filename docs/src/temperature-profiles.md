# Temperature profiles

Every run's temperature is set by `RunConditions(temperature = ...)`. A number
is Kelvin and becomes a `ConstantTemperature`; any `AbstractTemperature` is
used as given. There is no default temperature. For side-by-side runs across
constant and cooling profiles,
see [Tutorial 1 — Running Simulations](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%201%20Running%20Simulations.jl).

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
linear_T = LinearTemperature(293.15, -0.01 / 60)

# Arbitrary T(t) via a closure
periodic_T = CallableTemperature(t -> 293.15 + 2.0 * sin(2π * t / 36000))

crystal_model = CrystallisationModel(LysozymeSystem();
    nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0007),
    growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
cooling_run = RunConditions(temperature = linear_T, concentration = 18.0)
cooling_problem = CrystallisationProblem(crystal_model, cooling_run, (0.0, 28_800.0))
solution = solve(cooling_problem, MoM(); saveat = 3600.0)
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

profile = StepTemperature(293.15, 288.15, 7200.0) # seconds

step_run = RunConditions(temperature = profile, concentration = 18.0)
solution = solve(remake(cooling_problem; conditions = step_run), MoM(); saveat = 3600.0)
```
