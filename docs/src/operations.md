# Reactor operations and steady state

Steady relaxation uses solve precision consistent with the requested residual
and a hydraulic step bound. For mesh solvers, population residuals use a common
density scale across cells; tiny tail cells do not define a separate relative
precision requirement. Every cell still contributes to the full-state residual.
Supply `residual_scales` to select explicit component scales. Size-domain boundary
flows are reported separately from the physical outlet and included in the
steady total-inventory balance. `balance_reltol` controls the accepted relative
balance error; mesh/domain refinement remains necessary for resolved solvers.

CriSTool models **one well-mixed vessel with at most one inlet** and a
one-dimensional crystal population. Three hydraulic operations are supported
for *dynamic* simulation; a dedicated steady-state runner is described below.

| Operation | Volume | Inlet | Outlet | Hydraulic state |
| --- | --- | --- | --- | --- |
| `BatchOperation(; volume=1.0)` | fixed | none | none | none (unit reference) |
| `MSMPROperation(; volume, inflow, feed)` | fixed | one slurry feed | mixed slurry at `inflow` | none |
| `FedBatchOperation(; initial_volume, inflow, feed)` | evolving `dV/dt = inflow` | one slurry feed | none | `:volume` reactor state |

The operation is part of a run's `RunConditions`:

```julia
using CriSTool

msmpr_model = CrystallisationModel(LysozymeSystem(saturation_model = ConstantSolubility(1.0));
    nucleation = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf, nucleation_order = 1.0),
    growth = growth_empirical(growth_coefficient = 1e-9 / 60, growth_order = 1.0))

feed  = CrystallisationFeed(concentration = 2.0)                  # kg/m³, clear feed
msmpr = MSMPROperation(volume = 2.0, inflow = 0.5, feed = feed)   # m³, m³/s
msmpr_run = RunConditions(temperature = 293.15, concentration = 8.0, operation = msmpr)

trajectory = simulate(msmpr_model, msmpr_run; solver = MoM(), saveat = 0.0:3600.0:21600.0)
```

The default `BatchOperation()` keeps the concentration-based batch equations;
its `volume` is a unit reference that does not enter the balances.

## The feed

`CrystallisationFeed` describes the single slurry inlet:

- `concentration` is the feed solute concentration in kg/m³. It may be constant
  or a callable `f(time)` in seconds. Callable (time-dependent) inflow and
  feed concentration work for **transient** runs; the steady-state runner
  requires constant (autonomous) conditions.
- `crystals` is the fixed inlet population *per m³ of feed*, described by the
  same `AbstractInitialCrystals` types as the tank seed. It is independent of
  the tank's initial crystals.
- `solvent_state` carries additional feed solvent species only; the feed
  concentration has its own `concentration` field.
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
`RunConditions(crystals = ...)`; the run's `concentration` is the tank's
starting value, distinct from the feed concentration.

## Volume, flows and dilution

- Volume is a **separate reactor state**, not a solvent variable. Read it with
  `reactor_vars(solution)` (including `volume` for MSMPR and fed-batch) or
  `observable_values(solution, :volume)`. Mesh results also report size-domain
  boundary flows, including for batch operation.
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
constant feed concentration, and time-invariant kinetics), use the explicit
steady-state runner on a problem built from the model and run:

```julia
msmpr_problem = CrystallisationProblem(msmpr_model, msmpr_run; solver = MoM())
steady = solve_steadystate(msmpr_problem;
    options = SteadyStateOptions(relaxation_horizon = 400.0))
steady.concentration                  # scalar kg/m³
steady.d43                            # scalar m
steady.product.dissolved_solute_flow  # kg/s leaving in the liquid
steady.diagnostics.converged
```

`solve_steadystate` relaxes the generated dynamics until the scaled full-state
RHS residual is within tolerance. Scalar observables are plain values
(`steady.concentration`, `steady.d43`); `steady.moments` is one vector. Product
flows sit in `steady.product` (`dissolved_solute_flow`, `solid_mass_flow`), the
hydraulics in `steady.hydraulics`, and convergence, residual norms and balance
checks in `steady.diagnostics`; `steady.solution` is the endpoint as a
one-point solution. If a finite relaxation horizon ends before convergence or a
physical/balance check fails, the call throws `SteadyStateConvergenceError`,
which carries the unsuccessful result in its `solution` field
(`success = false`). `SteadyStateOptions` holds `relaxation_horizon`,
`minimum_relaxation`, `residual_reltol`, `residual_abstol`, `residual_scales`,
`autonomous` and `balance_reltol`; `initial_guess` is a keyword of
`solve_steadystate`. The default relaxation horizon is `100 × residence time`
for direct solving and prepared steady inference; residual tolerances default
to `1e-10`/`1e-12`.

Mesh density checks allow undershoots within 128 machine epsilons of the
population peak. The raw values are preserved; accepted undershoots have status
`:roundoff_negative_number_density`, with their minimum, peak, tolerance and
count in `diagnostics.population_density_diagnostics`. Larger negatives fail
the physical check. Quadrature states that cannot be reconstructed raise a
`DomainError` directly; prepared inference records these as failed candidates
and retains the reason in prediction diagnostics.

Opaque time profiles and user-defined kinetics require
`SteadyStateOptions(autonomous = true)` as an explicit declaration that they are
time invariant; the declaration is the caller's responsibility. Batch and
fed-batch runs are transient: use `simulate`. See
[Tutorial 9](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%209%20MSMPR%20Dynamic%20versus%20Steady%20State.jl)
for a dynamic-vs-steady comparison, and the [API reference](api/simulation.md)
for the result type.

Without internal kinetics, pure mixing approaches the feed concentration and
feed population. With nucleation, growth, dissolution or binary sources, the
steady tank condition follows their balance with feed and withdrawal.
A **clear feed** (`crystals = nothing`) can therefore produce a nonempty steady
population through nucleation. With neither crystal feed nor nucleation, complete
washout has no meaningful equilibrium size. DQMOM still requires a positive
seeded startup; the other solvers support unseeded MSMPR nucleation.

## Scope of this release

Dynamic modelling covers batch, ideal MSMPR and fed-batch; the explicit steady
runner covers autonomous MSMPR. Each uses one well-mixed vessel, at most one
inlet, and a one-dimensional crystal population. This release does **not** include two-dimensional population
balances, multiple vessels/streams, recycle, classified product withdrawal,
energy balances, or anti-solvent composition models. DQMOM retains its seeded
positive-node limitations (see [Solvers](solvers.md)).
