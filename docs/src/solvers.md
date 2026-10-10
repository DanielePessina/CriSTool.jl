# Solvers

CriSTool provides five solvers for the population balance equation:

- `MoM()` (Method of Moments): fast, returns moments only.
- `QMOM(nquadrature=...)`: evolves raw moments and reconstructs a Gaussian
  quadrature for moment-source closures.
- `DQMOM(nquadrature=...)`: evolves seeded quadrature weights and nodes directly
  while enforcing the first `2N` moment equations.
- `FiniteVol(meshsize=..., lmax=...)`: full PSD, moderate cost.
- `WENO(meshsize=..., lmax=...)`: higher-order finite volume, more accurate for sharp fronts.

The full runnable examples are [Tutorial 6](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%206%20Dissolution.jl)
for signed dissolution across the three solver families and
[Tutorial 7](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%207%20QMOM.jl) for moment inversion and
quadrature output.

## Outputs by solver

Every solver returns a `CrystallisationSolution`. The shared fields are
`time`, `concentration`, `solvent_state`, `reactor_state`, the moment-derived
sizes `d10`, `d32`, `d43`, `moment2`, `final_state`, `ode_stats` and
`success`. The solver-specific data live in `solution.population`:

- **MoM**: `MomentPopulation` with the raw `moments` (`µ0..µ_nmoments`).
- **QMOM**: `QuadraturePopulation` with the raw `moments`, the reconstructed
  quadrature `nodes`/`weights` and the inversion diagnostics. A three-node
  QMOM evolves `M₀:M₅`.
- **DQMOM**: `QuadraturePopulation` with the physical quadrature
  `weights`/`nodes`, the corresponding raw `moments` and projection
  diagnostics. A three-node DQMOM evolves six direct population variables and
  represents `M₀:M₅`.
- **FiniteVol/WENO**: `MeshPopulation` with `numberdensity`, `voldensity`
  and the volume-density quantiles `d10q`, `d50q`, `d90q`.

`state_vars(solution)` and `size_metrics(solution)` return the same data as
named tuples, independently of the solver; `quadrature(solution, index)`
returns one saved QMOM or DQMOM rule.

## Mesh storage

For `FiniteVol` and `WENO`, the mesh data (`cell_face`, `cell_centre`,
`cell_dL`, `meshsize`, `lmin`, and `lmax`) is stored on the solver. The
length-dependent dissolution rate accepts an explicit mesh argument when it
is evaluated outside the solver; use the solver-owned scratch buffer with
`growthrate!` in custom discretised code.

WENO limits each proposed time step to `0.9 * cell_dL / maximum(abs, G)`
using the net growth or dissolution rate at the current accepted state. It
applies the limit at initialization and after accepted steps, and preserves
any smaller `solve_options.dtmax` supplied by the caller. For scalar rates,
the maximum is simply `abs(G)`. This local estimate does not guarantee a CFL
bound at every intermediate stage when rates vary with state or time.
The timestep controller uses ordinary numeric rate values; ForwardDiff
derivatives still propagate through the population and solvent equations.

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
implementation: provide a seed through `RunConditions(crystals = ...)`.
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

growth_only_model = CrystallisationModel(
    LysozymeSystem(saturation_model = ConstantSolubility(10.0));
    nucleation = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                                      nucleation_order = 1.0),
    growth = growth_empirical(growth_coefficient = 3e-9, growth_order = 1.2))

seeded_run = RunConditions(temperature = 293.15, concentration = 20.0,
    crystals = LogNormalInitialCrystals(
        mass_concentration = 0.25, d43 = 12e-6, geometric_std = 1.25))

solution = simulate(growth_only_model, seeded_run;
    solver = DQMOM(nquadrature = 3), saveat = 0.0:3600.0:14400.0)

solution.population.nodes[:, end]
solution.population.weights[:, end]
solution.d43[end]
```

## Example: QMOM

```julia
using CriSTool

crystal_model = CrystallisationModel(LysozymeSystem();
    nucleation = nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 0.0007),
    growth = growth_empirical(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
batch_run = RunConditions(temperature = 293.15, concentration = 18.0)

solution = simulate(crystal_model, batch_run;
    solver = QMOM(nquadrature = 3), saveat = 0.0:3600.0:28800.0)

@show solution.population.moments[:, end]
@show quadrature(solution, length(solution.time)).nodes
@show solution.d43[end]
```

## Example: choosing solvers

The model and conditions stay the same; only the `solver` keyword changes.

```julia
save_grid = 0.0:3600.0:28800.0

sol_mom  = simulate(crystal_model, batch_run; solver = MoM(), saveat = save_grid)
sol_fv   = simulate(crystal_model, batch_run;
    solver = FiniteVol(meshsize = 100, lmax = 50e-6), saveat = save_grid)
sol_weno = simulate(crystal_model, batch_run;
    solver = WENO(meshsize = 100, lmax = 50e-6), saveat = save_grid)
```

## Overriding the time-stepper

Override the ODE time-stepping algorithm by setting the solver's
`timestepping_algorithm` field to an OrdinaryDiffEq algorithm instance, such
as `Tsit5()`, `SSPRK43()`, `Kvaerno5()` or `KenCarp4()`. The default
`nothing` selects the solver-specific algorithm. A per-call `algorithm`
keyword of `simulate` takes precedence over both.

```julia
using OrdinaryDiffEq: SSPRK43

sol = simulate(crystal_model, batch_run;
    solver = FiniteVol(meshsize = 200, lmax = 50e-6,
                       timestepping_algorithm = SSPRK43()),
    saveat = 0.0:3600.0:28800.0)
```
