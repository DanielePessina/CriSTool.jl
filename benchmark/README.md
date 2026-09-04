# CriSTool benchmarks + allocation analysis

Self-contained benchmarking setup for the CriSTool package. Lives entirely in
`benchmark/`; it never touches `src/`, `test/`, `docs/` or the repo root.

## How to run

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'   # once, after fresh checkout
julia --project=benchmark benchmark/run_benchmarks.jl
```

The script:

1. Loads the gold fixture (`test/fixtures/real-experimental-dataset.csv`,
   7 experiments).
2. Benchmarks three solvers with Chairmarks (`@be`, `evals=1`, `seconds=5`,
   like the original `Thesis - Benchmarks/benchmarks.jl` script):
   - **MoM** — the oracle: canonical SI params
     `[38.0, 0.0006, 1e-9 / 60, 3.0]` (nucl_CNT + growth_empirical +
     noaggregation + nobreakage), per-experiment time grids and `temp_profile`
     from each experiment.
   - **FiniteVol(200)** and **WENO(200)** — same 7 experiments and params on a
     fixed common grid `0:1800:16200` seconds (the former `0:30:270` minutes).
3. Prints per-solver median time, allocation count and bytes (Chairmarks
   `Sample.allocs`/`Sample.bytes`), plus ODE stats aggregated over the 7
   experiments: `nf`, `naccept`, `nreject`, `nsolve`, `nsave`.
   It also prints a separate seeded compact-solver comparison for
   `QMOM(nquadrature=3)` and `DQMOM(nquadrature=3)` using the same
   `LogNormalInitialCrystals` workload. DQMOM is kept out of the unseeded
   table because the v1 implementation requires a nonempty seed population.
4. Runs **AllocCheck** (`check_allocs`) on a representative warm MoM call and
   lists every flagged allocation/dispatch site with its frame.
5. Prints the benchmarked package's git hash (`git rev-parse --short HEAD`) for
   attribution.

Expected wall time ≈ 1–2 minutes (mostly precompile + the 5 s sampling windows).

## Metrics

| Metric | Meaning |
|---|---|
| `med time` | Median wall time of one full fit (7 `runsimulation` calls), from Chairmarks `Sample.time` |
| `med allocs` | Median number of heap allocations per fit (`Sample.allocs`) |
| `med bytes` | Median bytes allocated per fit (`Sample.bytes`) |
| `nf` | Cumulative RHS evaluations (`sol.ode_stats.nf`) across the 7 experiments |
| `nacc` / `nrej` | Accepted / rejected solver steps |
| `nsave` | Sum of `length(sol.time)` (saved points) |
| `nsteps` | Always `missing`: SciMLBase's current `DEStats` no longer has an `nsteps` field (it was removed); the old benchmark script read `stats.nsteps` and would silently get `missing` too |

The seeded comparison uses `LogNormalInitialCrystals(mass_concentration =
0.25, d43 = 12e-6, geometric_std = 1.25)` and each experiment's original
temperature, initial concentration, and observation-time grid. It is a
performance comparison of the compact moment/quadrature backends under an
identical seeded workload, not a claim that DQMOM can reproduce an unseeded
nucleation transient yet.

## Tooling decisions

- **Chairmarks 1.3.1** (the same package family as the original script, which
  used `@be ... evals=1 seconds=10`). Allocation numbers come from the
  per-sample `allocs` (count) and `bytes` fields, aggregated with the median.
  `@be` kwargs do **not** support `$` interpolation — pass plain symbols or
  literals (`seconds=SECONDS`).
- **AllocCheck 0.2.6 — WORKS on Julia 1.12.4** (aarch64-apple-darwin). Verified
  by actually running `check_allocs` on a real warm `runsimulation` call:
  it returns results in ~5 s. Use the function form
  `AllocCheck.check_allocs(f, argtypes)` (the macro form only wraps new
  definitions). Important caveat: AllocCheck only sees statically resolvable
  code — the ODE solve core sits behind kwcall boundaries, so it reports the
  entry wrapper's allocations (`paramaxis`/ComponentArray axis construction in
  `src/physics/model_interfaces.jl` and the flat-vector forward in
  `src/solvers/runsimulation.jl`) and
  cannot see inside the solve.
- **Fallback** (if AllocCheck breaks on a future Julia): Chairmarks alloc
  counts are already reported; add `InteractiveUtils.@code_warntype` on the hot
  closure as a manual type-stability check.
- `stats.nsteps` handling: defensive `hasproperty` check, matching the original
  script's `_get_ode_stat`.

## Historical baseline (legacy units; 2026-08-28, git `b01b5b8`)

```
solver     med time   med allocs      med bytes       nf     nacc     nrej   nsolve      nsave
MoM        19.818 ms       357758     29417208.0     7863     1307        0        0         57
FV200      19.363 ms      1494015     35501472.0     3447      571        0        0         70
WENO200     17.233 ms      1064149     26652816.0     2391      395        0        0         70
```

- Per experiment: MoM ≈ 2.8 ms / 51k allocs / 4.2 MB; FV200 ≈ 2.8 ms / 213k
  allocs / 5.1 MB; WENO200 ≈ 2.5 ms / 152k allocs / 3.8 MB.
- AllocCheck: **flags allocations** — 16 sites, all in the flat-vector wrapper
  (`paramaxis` → `ViewAxis`/`UnitRange`/`NamedTuple` in
  `src/physics/model_interfaces.jl`, `ComponentArray` construction + kwarg
  splat in `src/solvers/runsimulation.jl`).
  The measured runtime cost of that wrapper is tiny (~70 allocs / 3.5 KB per
  full 7-experiment fit vs. the ComponentArray-direct path).
- The overwhelming allocation volume comes from the per-call ODE machinery
  (every `runsimulation` call constructs an `ODEProblem` + solver cache inside

## Post type-stability fix (legacy units; 2026-08-29, uncommitted on `4e10db6`)

```
solver     med time   med allocs      med bytes       nf     nacc     nrej   nsolve      nsave
MoM         0.990 ms         1519       126400.0     7863     1307        0        0         57
FV200       2.677 ms         2058      1010464.0     3447      571        0        0         70
WENO200      4.632 ms         2478      1829856.0     2403      397        0        0         70
```

- What was fixed (in `src/solvers/mom.jl` and the discretised solver files):
  1. **MoM RHS type instability**: `n_mom`/`n_states` were boxed closure captures
     (`code_warntype`: `Body::ANY`, 64 B allocated per RHS call). The
     `if n_states == 6` value-branch became a runtime branch on a dynamic value,
     deoptimising the whole solve. Replaced by static dispatch
     `_mom_rhs(..., u::SVector{6})` / `_mom_rhs(..., u::SVector{N}) where N`
     (N is a type parameter, so the `ntuple(Val(N))` fallback is static too);
     the `@assert n_mom >= 2` moved out of the RHS into
     `crystallisation_odeproblem` (construction-time).
  2. **FV/WENO RHS broadcast temps**: `sum(cell_dL .* numberdensity .* g .*
     cell_centre.^2)` allocated one Vector per RHS call (~1.6 KB at mesh 200).
     Replaced by the allocation-free `_concentration_depletion` accumulator at
     all three sites (FV, FV-length, WENO).
- MoM per-experiment (thesis sweep, tsit5, save_idx 0:4:400): **0.158 ms / 241
  allocs** — matches the original thesis code's 0.161 ms class with ~15x fewer
  allocs (was 0.94–1.35 ms / 23.3k allocs).
- All three RHS closures are now fully type-stable (`Body::Nothing` for FV/WENO
  iip, `Body::SVector{6,Float64}` for MoM) and allocation-free (0 B/call).
- AllocCheck: same 16 wrapper-only sites as before (flat-vector
  `paramaxis`/`ComponentArray`/kwarg-splat), ~70 allocs / 3.5 KB per 7-exp fit.
- Full test suite: 492 pass / 3 pre-existing broken; gold-fixture and
  nf/naccept identical to pre-fix runs (same physics, pure overhead removal).
- Historical "env effect" (identical code+manifest in two dirs → 0.16 vs
  0.95 ms): **no longer reproduces**; the original code measures 0.164 ms in
  this same env. See `research/benchmark-perf-log.md` for the full story.
  `_simulatecrystallisation`); this is opaque to AllocCheck.

## Numeric SI migration reruns (2026-09-03)

The harness was rerun after converting the parameters and fixed grids to SI
(`CANONICAL_θ = [38.0, 0.0006, 1e-9 / 60, 3.0]`; FV/WENO grid
`0:1800:16200` s). The first table is the SI migration commit with the
pre-existing solver tolerances; the second is the final main checkout, which
preserves the tighter FV/WENO tolerances from the original dirty checkout.

```
solver     med time   med allocs      med bytes       nf     nacc   nrej   nsolve   nsave
MoM         1.062 ms         1414       107984.0     7845     1304      0        0      57
FV200       3.131 ms         2275      1192688.0     3531      585      0        0      70
WENO200     4.888 ms         2688      2011632.0     2487      411      0        0      70
QMOM3       2.036 ms        10034       647248.0    14499     2413      0        0      70
```

The final main checkout (`edf7de8` plus the preserved uncommitted tolerance
edits) measured:

```
solver     med time   med allocs      med bytes       nf     nacc   nrej   nsolve   nsave
MoM         1.032 ms         1414       107984.0     7845     1304      0        0      57
FV200       6.913 ms         2275      1192688.0     8877     1476      0        0      70
WENO200    16.916 ms         2688      2011632.0     9747     1621      0        0      70
QMOM3       1.990 ms        10034       647248.0    14499     2413      0        0      70
```

Relative to the immediately preceding type-stable legacy-unit run, the final
main timings are approximately +2% (MoM), +155% (FV200), and +262% (WENO200).
That FV/WENO increase is from the tighter tolerances, not a type-instability
regression: accepted steps rise from `571→1476` and `397→1621`, respectively.
The trajectory oracle comparison still shows concentration agreement better
than 5×10⁻⁵ relative and discretised size-metric agreement better than
1.5×10⁻⁴ relative. Early MoM d43 differs by up to 1.5% because the migration
intentionally removed the old additive, dimensionally-invalid moment-ratio
floor; final MoM d43 agrees within 10⁻⁴ relative.

The isolated SI kernel benchmark (`kernel_benchmarks.jl`) measured:

```
mesh   aggregation μs   aggregation bytes   breakage μs   breakage bytes
50             12.54                 480           3.33             480
100            46.50                 928          13.38             928
250           266.38                2064          78.38            2064
500          1025.44                4160         305.92            4160
```

## Seeded DQMOM comparison (2026-09-03)

The benchmark harness was extended with a seeded compact-solver lane. On
Julia 1.12.4, Apple Silicon, with the current working-tree DQMOM changes and
the same seven fixture experiments, the measured table was:

```
solver             med time   med allocs      med bytes       nf     nacc   nrej   nsave
QMOM3-seeded        0.732 ms         6755       515456.0     2517      416      0      57
DQMOM3              4.478 ms         4853       614784.0     7581     1245     15      57
```

Both rows use `LogNormalInitialCrystals(mass_concentration = 0.25,
d43 = 12e-6, geometric_std = 1.25)`, `nucl_CNT`, scalar empirical growth,
and no aggregation or breakage. The DQMOM implementation uses an exact
three-node static projection path for this workload; binary-source and other
node-count paths remain on the generic dense projection implementation.
