# Contributing to CriSTool

CriSTool is a Julia package (1.10 and later) for batch and continuous
crystallisation population-balance simulation, parameter estimation and
uncertainty quantification. This guide covers development setup, the testing
and code standards, and the release checklist.

## Development setup

```sh
git clone https://github.com/DanielePessina/CriSTool.jl
cd CriSTool.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Manifest.toml` is not tracked. Every checkout resolves its own dependencies
from `Project.toml`, and CI does the same on Julia 1.10 to 1.13.

### Persistent REPL with Revise

Package load and precompilation take a while, so iterate in one long-lived
REPL rather than starting Julia per change. One option is tmux:

```sh
tmux new -d -s cristool-dev
tmux send-keys -t cristool-dev:0.0 -- 'julia --project=.' Enter
tmux send-keys -t cristool-dev:0.0 -- 'using Revise, CriSTool' Enter
tmux attach -t cristool-dev          # watch it live
tmux capture-pane -p -J -t cristool-dev:0.0 -S -200   # read output from a script
```

`Revise` is a development tool. Install it in your global Julia environment
(or load it from `startup.jl`) and never add it to `[deps]` in `Project.toml`.

## Tests

`test/runtests.jl` owns the shared imports and the test groups. Each group is
one `test/test_<group>.jl` file, and `test/fixtures.jl` holds shared models
(constant-rate laws with closed-form answers, a uniform seed, a lysozyme
regression case, small operations).

| Group | Covers |
| --- | --- |
| `numerics` | Hand-valued unit checks: reconstruction, QMOM inversion, moment sources, kernels, rates |
| `simulate` | Every solver against the translation closed form, seeds, aggregation, breakage, dissolution |
| `operations` | Fed-batch and MSMPR mixing and washout laws, time-varying flows |
| `steady` | Steady MSMPR contract, failure modes, closed form across solvers |
| `data` | Loaders, conditions rules, variance balancing, bootstrap |
| `fit_predict` | Loss accounting, fits (LBFGS, DE, NSGA2, NUTS, ABC), predictions |
| `gradients` | Finite differences against autodiff through loss, solvers and operations |
| `plotting`, `aqua` | Optional. Plot extensions, Aqua package-quality checks and public-name checks |

```sh
# core groups
julia --project=. -e 'using Pkg; Pkg.test()'
# selected groups
julia --project=. -e 'using Pkg; Pkg.test(test_args=["simulate", "fit_predict"])'
# everything, including plotting and Aqua
julia --project=. -e 'using Pkg; Pkg.test(test_args=["all"])'
```

To run a single file with fast iteration, load the shared imports first. Test
files have no `using` preamble of their own:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); using CriSTool; using Test; include("test/fixtures.jl"); include("test/test_simulate.jl")'
```

(`test_data.jl` and a few others need the extra imports at the top of
`runtests.jl`; copy them if a file reports an undefined name.)

### Testing standard

Tests must assert against independent ground truth. A test that re-derives the
code's own formula, or only asserts `isfinite`, `!= 0` or `>= 0`, stays green
after a behaviour-breaking refactor and gives false confidence. Use:

- hand-computed values;
- physical identities such as mass balance or known solubility;
- closed forms (the translation fixture, ideal MSMPR);
- finite-difference against autodiff agreement via `DifferentiationInterface`
  (the cases in `test/test_gradients.jl` are the model to copy);
- frozen fixtures such as `test/fixtures/real-experimental-dataset.csv`.

Keep each group fast.

## Code style

- **Method-oriented Julia.** Use dispatch or small trait functions
  (`_is_length_law`, `_is_time_invariant`) instead of `if x isa A ... elseif x
  isa B` chains. Use values or singleton types instead of `Symbol` or `Bool`
  mode flags. No `getproperty` delegation, no structs that duplicate another
  type's role, concretely typed fields on hot paths. `isa` is fine for
  argument validation.
- **Greppable names.** Every identifier should be a distinctive substring:
  `rg name` must return a clean hit list. Avoid single letters and overloaded
  generic names (`f`, `p`, `x`, `sol`, `problem`, `state`) and never duplicate
  an existing name.
- **Structs.** Public structs use `Base.@kwdef @concrete struct`
  (ConcreteStructs) and keyword construction.
- **Kinetic laws.** Parameters are named ASCII fields declared once by
  `paramaxis(law)`; laws are CamelCase types. See
  `examples/Tutorial 4 Defining a Custom Kinetic.jl`.
- **Public API.** Exported names and names marked
  `Compat.@compat public` in `src/CriSTool.jl` are supported. Underscore
  prefixed names are internal. When you add a documented but non-exported
  name, add it to the `public` list and to the check in `test/test_aqua.jl`.
- **Docs and tutorials** (`docs/src/*.md`, `examples/Tutorial *.jl`,
  `CHANGELOG.md`) are the user-facing surface. Update them whenever types or
  signatures change.
- Nothing writes to disk unless the caller asks for it.

## Documentation

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The tutorials run in their own environment:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --project=examples "examples/Tutorial 1 Running Simulations.jl"
```

## Dependency compatibility

`CompatHelper` (`.github/workflows/CompatHelper.yml`) opens pull requests when
a dependency releases a version outside a `[compat]` range. Prefer open-ended
ranges such as `"0.25.88"` (which means `0.25.88` up to, not including,
`0.26`) over upper-capped ranges, and cap a dependency only for a specific,
documented reason. Record the reason in `CHANGELOG.md` and in the pull
request.

## Release checklist

1. Bump `version` in `Project.toml`.
2. In `CHANGELOG.md`, replace "unreleased" in the version heading with the release date (or open a new unreleased section).
3. Run the full test suite, including Aqua and plotting:
   `julia --project=. -e 'using Pkg; Pkg.test(test_args=["all"])'`.
4. Build the documentation: `julia --project=docs docs/make.jl`.
5. Run the benchmarks and compare with the previous release:

   ```sh
   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
   julia --project=benchmark benchmark/solver_matrix.jl
   julia --project=benchmark benchmark/inference_overhead.jl
   ```

   `solver_matrix.jl` writes `benchmark/results/solver_matrix_<hash>.txt`.
   Compare it with the previous file in that directory (the directory is
   gitignored, so keep the last release's file locally). Investigate any
   slowdown above 10 percent before releasing. See `benchmark/README.md` for
   the metrics.
6. Commit on `main`, then comment `@JuliaRegistrator register` on the commit.
   After the General registry merges the pull request, TagBot
   (`.github/workflows/TagBot.yml`) creates the git tag and GitHub release.
