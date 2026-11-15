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

`model_interface.jl` compares built-in/callable growth rates, configured/model
simulation entry points, and full/selected prepared losses. It verifies matched
physical trajectories and objectives before timing, using BenchmarkTools medians
and warmed inputs. BenchmarkTools is a benchmark-only dependency.

Run `julia --project=benchmark benchmark/model_interface.jl`. To compare against
a package checkout's existing resolved runtime environment while loading the
benchmark instrumentation separately:

```sh
julia --project=. -e 'push!(LOAD_PATH, joinpath(pwd(), "benchmark")); include("benchmark/model_interface.jl")'
```

Observed on Julia 1.12.4, one thread, Apple M1, using the root resolved environment
and the second command (2026-10-07):

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
