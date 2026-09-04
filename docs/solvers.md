# Solvers

CriSTool provides five solvers for the population balance equation:

- `MoM()` (Method of Moments): fast, returns moments only.
- `QMOM(nquadrature=...)`: evolves raw moments and reconstructs a Gaussian
  quadrature for moment-source closures.
- `DQMOM(nquadrature=...)`: evolves seeded quadrature weights and nodes directly
  while enforcing the first `2N` moment equations.
- `FiniteVol(meshsize=..., lmax=...)`: full PSD, moderate cost.
- `WENO(meshsize=..., lmax=...)`: higher-order finite volume, more accurate for sharp fronts.

The full runnable examples are [Tutorial 6](<../examples/Tutorial 6 Dissolution.jl>)
for signed dissolution across the three solver families and
[Tutorial 7](<../examples/Tutorial 7 QMOM.jl>) for moment inversion and
quadrature output.

## Outputs by solver

- **MoM**: `concentration`, `d10`, `d32`, `d43`, `moment2`.
- **QMOM**: `concentration`, raw `moments`, reconstructed
  `quadrature_nodes`/`quadrature_weights`, and the moment-derived
  `d10`, `d32`, `d43`, `moment2` metrics. A three-node QMOM evolves `M₀:M₅`.
- **DQMOM**: `concentration`, physical quadrature `weights`/`nodes`, the
  corresponding raw `moments`, projection diagnostics, and the same
  `d10`, `d32`, `d43`, `moment2` metrics. A three-node DQMOM evolves six
  direct population variables and represents `M₀:M₅`.
- **FiniteVol/WENO**: `concentration`, `numberdensity`, `voldensity`,
  volume-density quantiles `d10q`, `d50q`, `d90q`, **and** the
  moment-derived sizes `d10`, `d32`, `d43`, `moment2` (computed once at
  solve time so `sol.d43` works the same way as on a MoM solution).

## Mesh storage

For `FiniteVol` and `WENO`, the mesh data (`cell_face`, `cell_centre`,
`cell_dL`, `meshsize`, `lmin`, and `lmax`) is stored on the solver. The
length-dependent dissolution rate accepts an explicit mesh argument when it
is evaluated outside the solver; use the solver-owned scratch buffer with
`growthrate!` in custom discretised code.

## QMOM, DQMOM, and signed rates

QMOM stores raw physical moments in the state and uses `coordinate_scale` only
while inverting those moments. The public quadrature values are crystal
lengths in metres and particle-number weights. Use `quadrature(solution, i)`
to reconstruct the active rule at saved time index `i`.

QMOM accepts scalar growth or dissolution kinetics and the supported
volume-additive aggregation and uniform-in-volume breakage closures. DQMOM
accepts the same validated scalar closures, plus a length-dependent growth
law. Both moment solvers reject `growth_dissolution_length`; use FiniteVol or
WENO for length-dependent dissolution.

## DQMOM state and contract

DQMOM is a subtype of `AbstractMomentSolver`, but it does not evolve the raw
moment vector used by `MoM` and `QMOM`. Its public population block is

```text
[w₁, …, w_N, L₁, …, L_N, solvent_state…]
```

where `wᵢ` is a particle-number weight in `m⁻³` and `Lᵢ` is a physical crystal
length in metres. The initial weights must be strictly positive and the nodes
must be distinct and above `minimum_size`. DQMOM is seeded-only in this first
implementation: provide `initial_crystals` or a complete direct `initial_state`.
An empty initial population is rejected instead of being activated by an
artificial epsilon seed.

For `N` nodes, the internal dimensionless state uses `qᵢ = wᵢ/W` and
`xᵢ = Lᵢ/Lscale`, and stores `(qᵢ, qᵢ xᵢ)`. The direct equations are the
standard DQMOM projection system

```math
\sum_i \left[(1-k)x_i^k\dot q_i
  + kx_i^{k-1}\frac{d}{dt}(q_i x_i)\right] = S_k,
\qquad k=0,\ldots,2N-1,
```

where `Sₖ` is the dimensionless source for the corresponding physical moment.
This is the direct analogue of the raw-moment source equations, not a second
moment inversion at every time step. `DQMOMProjectionDiagnostics` records node
separation, minimum weights, and the projection-matrix condition estimate.

The v1 lower-boundary policy is explicit: scalar dissolution is allowed while
all nodes remain above `minimum_size`, then the solve fails with a
`DomainError`. Node deletion and active-set changes are not silently applied.

### Example: seeded DQMOM

```julia
using CriSTool

initial_crystals = LogNormalInitialCrystals(
    mass_concentration = 0.25,
    d43 = 12e-6,
    geometric_std = 1.25)

parameters = [3e-9, 1.2]  # growth coefficient (m/s), supersaturation order
problem, solution = runsimulation(
    parameters;
    nucl = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                                nucleation_order = 1.0),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = DQMOM(nquadrature = 3),
    initial_crystals = initial_crystals,
    initial_concentration = 20.0,
    saturation_model = ConstantSolubility(10.0),
    save_idx = 0.0:3600.0:14400.0)

solution.nodes[:, end]
solution.weights[:, end]
solution.d43[end]
```

## Example: QMOM

```julia
using CriSTool

params = [38.0, 0.0007, 1e-9 / 60, 3.0]
problem, solution = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = QMOM(nquadrature = 3),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

@show solution.moments[:, end]
@show quadrature(solution, length(solution.time)).nodes
@show solution.d43[end]
```

## Example: choosing solvers

```julia
using CriSTool

params = [38.0, 0.0007, 1e-9 / 60, 3.0]
nucl, gr = nucl_CNT(), growth_empirical()
agg, br  = noaggregation(), nobreakage()

# MoM
_, sol_mom = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = MoM(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

# Finite volume
_, sol_fv = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = FiniteVol(meshsize = 100, lmax = 50e-6),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)

# WENO
_, sol_weno = runsimulation(
    params; nucl = nucl, gr = gr, agg = agg, br = br,
    solver = WENO(meshsize = 100, lmax = 50e-6),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)
```

## Overriding the time-stepper

You can override the ODE time-stepping algorithm via the
`timestepping_solver` keyword in `runsimulation` (or by setting the
solver field). This is used in the benchmarking scripts.

```julia
using CriSTool
import OrdinaryDiffEqSSPRK

params = [38.0, 0.0007, 1e-9 / 60, 3.0]

_, sol = runsimulation(
    params;
    nucl = nucl_CNT(),
    gr = growth_empirical(),
    agg = noaggregation(),
    br = nobreakage(),
    solver = FiniteVol(meshsize = 200, lmax = 50e-6),
    timestepping_solver = OrdinaryDiffEqSSPRK.SSPRK43(),
    initial_concentration = 18.0,
    save_idx = 0.0:3600.0:28800.0,
)
```
