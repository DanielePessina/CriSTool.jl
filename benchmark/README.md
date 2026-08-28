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
   sheet `Unseeded_PE`, loading 0.0 → 7 experiments).
2. Benchmarks three solvers with Chairmarks (`@be`, `evals=1`, `seconds=5`,
   like the original `Thesis - Benchmarks/benchmarks.jl` script):
   - **MoM** — the oracle: canonical params `[38.0, 0.6, 1.0, 3.0]`
     (nucl_CNT + growth_empirical + noaggregation + nobreakage), per-experiment
     time grids, `loading`/`temp_profile` from each experiment.
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
  entry wrapper's allocations (`paramaxis`/ComponentArray axis construction at
  `src/Models.jl:34` and the flat-vector forward at `src/Models.jl:2045`) and
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
  (`paramaxis` → `ViewAxis`/`UnitRange`/`NamedTuple` at `src/Models.jl:34`,
  `ComponentArray` construction + kwarg splat at `src/Models.jl:2045-2046`).
  The measured runtime cost of that wrapper is tiny (~70 allocs / 3.5 KB per
  full 7-experiment fit vs. the ComponentArray-direct path).
- The overwhelming allocation volume comes from the per-call ODE machinery
  (every `runsimulation` call constructs an `ODEProblem` + solver cache inside
  `_simulatecrystallisation`); this is opaque to AllocCheck.