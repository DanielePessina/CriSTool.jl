# Reactor operations and steady state

CriSTool v1 models **one well-mixed vessel with at most one inlet** and a
one-dimensional crystal population. Three hydraulic operations are supported
for *dynamic* simulation; a dedicated steady-state runner is described below.

| Operation | Volume | Inlet | Outlet | Hydraulic state |
| --- | --- | --- | --- | --- |
| `BatchOperation(; volume=1.0)` | fixed | none | none | none (unit reference) |
| `MSMPROperation(; volume, inflow, feed)` | fixed | one slurry feed | mixed slurry at `inflow` | none |
| `FedBatchOperation(; initial_volume, inflow, feed)` | evolving `dV/dt = inflow` | one slurry feed | none | `:volume` reactor state |

All three are configured through the `operation` field of a
`CrystallisationProblem`; with the keyword `runsimulation` form, pass
`operation = ...` and it is forwarded to the problem constructor:

```julia
using CriSTool

nucl = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                            nucleation_order = 1.0)   # no nucleation
gr   = growth_empirical()

feed   = CrystallisationFeed(concentration = 2.0)     # kg/m³, clear feed
msmpr  = MSMPROperation(volume = 2.0, inflow = 0.5, feed = feed)  # m³, m³/s

problem, solution = runsimulation(
    [1e-9 / 60, 1.0];        # [growth_coefficient, growth_order]
    nucl = nucl, gr = gr, agg = noaggregation(), br = nobreakage(),
    solver = MoM(),
    operation = msmpr,
    initial_concentration = 8.0,
    saturation_model = ConstantSolubility(1.0),
    save_idx = 0.0:3600.0:21600.0,
)
```

A `CrystallisationProblem` also accepts `operation = ...` directly. The batch
default `BatchOperation()` keeps the existing concentration-based batch
equations; its `volume` is a unit reference that does not enter the balances.

## The feed

`CrystallisationFeed` describes the single slurry inlet:

- `concentration` is the feed solute concentration in kg/m³. It may be constant
  or a callable `f(time)` in seconds. Callable (time-dependent) inflow and
  feed concentration work for **transient** runs; the steady-state runner
  requires constant (autonomous) conditions.
- `crystals` is the fixed inlet population *per m³ of feed*, described by the
  same `AbstractInitialCrystals` types as the tank seed. It is independent of
  the tank's initial crystals.
- `solvent_state` carries additional feed solvent values (never `:concentration`,
  which is supplied by the `concentration` keyword).
- `transport` is an explicit hook for extra solvent variables:
  `transport(problem, state, time, feed_values, inflow, volume)` returns a
  `NamedTuple` of named extra transport rates. Concentration transport is always
  built in; the hook must not return a concentration rate. Derived variables
  such as **pH are not linearly mixed** without such an explicit hook.

```julia
seeded_feed = CrystallisationFeed(
    concentration = 2.0,
    crystals = LogNormalInitialCrystals(mass_concentration = 0.5,
                                        d43 = 20e-6, geometric_std = 1.05))
```

The tank's own initial population is set independently with
`initial_crystals` (or `initial_state`); `initial_concentration(problem)`
exposes the configured tank input, distinct from the feed concentration.

## Volume, flows and dilution

- Volume is a **separate reactor state**, not a solvent variable. Read it with
  `reactor_vars(solution)` (a `(; volume = ...)` NamedTuple for MSMPR and
  fed-batch, empty for batch) or `observable_values(solution, :volume)`.
  `reactor_volume(problem, state)` returns the current volume from a numerical
  state; `operation_flows(problem, time)` returns `(; inflow, outflow)` in m³/s.
- A fed-batch dilutes: every intensive density (moments, number density,
  concentration) satisfies `density = internal rate + (Qin/V)(feed − density)`
  with `dV/dt = Qin`. In an MSMPR the outlet equals the inlet and the
  exchange rate is `Qin/V`; the residence time is `V/Qin`.
- Fed-batch **includes volume and dilution** by construction: run
  `FedBatchOperation` and read `observable_values(solution, :volume)`.

## Steady state

Dynamic simulation is the default for every operation. The **final point of a
transient run is not automatically a steady state** — even when the trajectory
visually flattens, the numerical integration stops at the requested horizon, not
at equilibrium.

For an **autonomous fixed-flow MSMPR** (constant `volume`, constant `inflow`,
constant feed concentration), the release backend provides an explicit
steady-state runner:

```julia
steady = solve_steadystate(problem)    # autonomous fixed-flow MSMPR only
```

`solve_steadystate` relaxes the dynamic equations to equilibrium and returns
**only the final state**, together with a full scaled residual and
flow/conservation diagnostics. The relaxation horizon and residual tolerance
are solve options; transient profiles are not accepted because the steady
contract requires autonomous conditions. Until that backend is integrated,
verify a steady operating point by relaxing a long dynamic run and checking
the scaled residual yourself (see
[Tutorial 9](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%209%20MSMPR%20Dynamic%20versus%20Steady%20State.jl)).

At steady state the tank condition is the feed condition: concentration tends
to the feed concentration and the population tends to the feed crystal load.
With a **clear feed** (`crystals = nothing`) the steady population is empty,
so moment-derived sizes such as `d43` are undefined at equilibrium; use a
crystal-bearing feed when the steady size distribution is the quantity of
interest.

## Scope of this release

Dynamic and steady modelling cover a batch, an ideal MSMPR, and a fed-batch,
each with one well-mixed vessel and one inlet, and a one-dimensional crystal
population. This release does **not** include two-dimensional population
balances, multiple vessels/streams, recycle, classified product withdrawal,
energy balances, or anti-solvent composition models. DQMOM retains its seeded
positive-node limitations (see [Solvers](solvers.md)).