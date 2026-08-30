# CriSTool benchmarks + allocation analysis

Self-contained benchmarking setup for the CriSTool package. Lives entirely in
`benchmark/`; it never touches `src/`, `test/`, `docs/` or the repo root.

## How to run

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'   # once, after fresh checkout
julia --project=benchmark benchmark/run_benchmarks.jl
```

The script:

1. Loads the gold fixture (`test/fixtures/real-experimental-dataset.xlsx`,
   sheet `Unseeded_PE` → 7 experiments).
2. Benchmarks three solvers with Chairmarks (`@be`, `evals=1`, `seconds=5`,
   like the original `Thesis - Benchmarks/benchmarks.jl` script):
   - **MoM** — the oracle: canonical params `[38.0, 0.6, 1.0, 3.0]`
     (nucl_CNT + growth_empirical + noaggregation + nobreakage), per-experiment
     time grids and `temp_profile` from each experiment.
   - **FiniteVol(200)** and **WENO(200)** — same 7 experiments and params on a
     fixed common grid `0:30:270`.
3. Prints per-solver median time, allocation count and bytes (Chairmarks
   `Sample.allocs`/`Sample.bytes`), plus ODE stats aggregated over the 7
   experiments: `nf`, `naccept`, `nreject`, `nsolve`, `nsave`.
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

## Baseline (2026-08-28, git `b01b5b8`, Julia 1.12.4, Chairmarks 1.3.1)

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

## Post type-stability fix (2026-08-29, uncommitted on `4e10db6`, Julia 1.12.4, Chairmarks 1.3.1)

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
