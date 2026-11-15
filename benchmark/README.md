# CriSTool benchmarks + allocation analysis

This directory contains a standalone benchmarking environment for CriSTool.
It does not modify `src/`, `test/`, `docs/`, or the repository root.

## How to run

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/run_benchmarks.jl
```

The script:

1. Loads the gold fixture (`test/fixtures/real-experimental-dataset.csv`,
   seven experiments).
2. Benchmarks MoM, `FiniteVol(200)`, and `WENO(200)` with Chairmarks on the
   canonical SI parameter vector.
3. Reports median time, allocations, allocated bytes, and ODE statistics.
4. Runs a representative AllocCheck analysis and prints its flagged frames.
5. Prints the package git hash used for the benchmark.

It also includes a seeded comparison of `QMOM(nquadrature = 3)` and
`DQMOM(nquadrature = 3)`. DQMOM is kept out of the unseeded table because the
current implementation requires a nonempty seed population.

Expected wall time is approximately 1–2 minutes after dependencies have been
precompiled.

## Metrics

| Metric | Meaning |
|---|---|
| `med time` | Median wall time of one full fit from Chairmarks |
| `med allocs` | Median heap-allocation count per fit |
| `med bytes` | Median bytes allocated per fit |
| `nf` | Cumulative right-hand-side evaluations |
| `nacc` / `nrej` | Accepted / rejected solver steps |
| `nsave` | Sum of saved solution points |

The seeded comparison uses `LogNormalInitialCrystals(mass_concentration =
0.25, d43 = 12e-6, geometric_std = 1.25)` and each experiment's original
temperature, initial concentration, and observation-time grid. It compares
compact solver backends under the same seeded workload; it is not a claim that
DQMOM reproduces an unseeded nucleation transient.

## Tooling decisions

- Chairmarks provides timing, allocation counts, and allocated bytes.
- AllocCheck uses `check_allocs(f, argtypes)` on a warmed representative call.
  It reports statically resolvable wrapper allocations; the ODE solve core is
  behind keyword-call boundaries.
- If AllocCheck changes in a future Julia release, use Chairmarks allocation
  counts and `InteractiveUtils.@code_warntype` as the fallback checks.
- ODE statistics are read defensively because current SciMLBase versions no
  longer expose an `nsteps` field.

Benchmark output is machine-dependent and is intentionally ignored by Git.
Record reproducible results outside the package tree when a benchmark run is
needed for a paper or release note.

## Named model interface

`model_interface.jl` compares built-in/callable growth rates, a plain
`solve(problem, MoM())` against `solve(remake(problem; p), MoM())`, and
full/selected prepared losses. It verifies matched
physical trajectories and objectives before timing, using BenchmarkTools medians
and warmed inputs. BenchmarkTools is a benchmark-only dependency.

Run `julia --project=benchmark benchmark/model_interface.jl`. To compare against
a package checkout's existing resolved runtime environment while loading the
benchmark instrumentation separately:

```sh
julia --project=. -e 'push!(LOAD_PATH, joinpath(pwd(), "benchmark")); include("benchmark/model_interface.jl")'
```

Observed on Julia 1.12.4, one thread, Apple M1, using the root resolved environment
and the second command (2026-10-07, before the `solve(problem, solver)` interface;
the two solve rows then measured `runsimulation(problem)` and `simulate(problem)`):

| Workload | Median | Bytes | Allocations |
| --- | ---: | ---: | ---: |
| built-in growth rate | 28.39 ns | 0 | 0 |
| callable growth rate | 28.48 ns | 0 | 0 |
| configured MoM solve | 168.25 µs | 14,096 | 212 |
| model simulation | 168.88 µs | 14,096 | 212 |
| full prepared loss | 152.38 µs | 10,608 | 182 |
| selected prepared loss | 153.13 µs | 10,624 | 182 |

The small timing differences are inconclusive. These measurements support
accepting the concrete law/named-value representation; they are not a claim of
an improvement in solver performance or first-call compilation latency. The
workload uses CNT/empirical growth, MoM, five saved times over four hours, and a
one-parameter selection. Mesh solves and large inference studies need their own
representative benchmarks.

## WENO time integrator (`weno_integrator.jl`)

`julia --project=. benchmark/weno_integrator.jl` compares Tsit5 and SSPRK43 on
WENO solves at the default tolerances. Errors are the maximum relative error at
the saved times against the same discretisation solved with Tsit5 at
`reltol = 1e-11`; for the translation seed, also against the closed form (which
includes the mesh error). Times are the minimum of seven solves on one Apple
M-series core, measured 2026-11-15.

| Case | Integrator | nf | Time (ms) | C error | d43 error |
| --- | --- | ---: | ---: | ---: | ---: |
| translation seed, WENO(200) | Tsit5 | 823 | 0.72 | 7.5e-13 | 4.4e-11 |
| translation seed, WENO(200) | SSPRK43 | 2588 | 2.19 | 2.0e-12 | 1.1e-10 |
| lysozyme CNT + growth, WENO(200) | Tsit5 | 1303 | 0.87 | 1.6e-6 | 7.2e-8 |
| lysozyme CNT + growth, WENO(200) | SSPRK43 | 3676 | 2.42 | 2.4e-5 | 1.3e-6 |
| dissolution to extinction, WENO(40) | Tsit5 | 1531 | 0.40 | 6.5e-9 | 2.0e-5 |
| dissolution to extinction, WENO(40) | SSPRK43 | 3796 | 1.01 | 2.6e-9 | 1.1e-4 |
| partial dissolution, WENO(200) | Tsit5 | 535 | 0.55 | 2.7e-14 | 1.0e-12 |
| partial dissolution, WENO(200) | SSPRK43 | 1776 | 1.75 | 1.8e-13 | 6.8e-12 |

Tsit5 stays the default for growth-only WENO: it is about three times cheaper
and more accurate. With dissolution WENO keeps SSPRK43, whose strong-stability
property guards signed transport near extinction; Tsit5 is cheaper there too
but not uniformly more accurate (concentration at extinction). Pass
`alg = Tsit5()` to `solve` to choose it.
