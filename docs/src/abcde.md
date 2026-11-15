# ABCDE routine

CriSTool implements Approximate Bayesian Computation via the ABCDE
algorithm. The unified entry point is `run_abc`, which owns the common
scaffolding (target computation, panels, persistence, plotting) and
selects an algorithm via a `<:AbstractABCSampler` keyword.

Two samplers ship today:

- `ABCDESampler(α=0)` — standard ABCDE (Differential Evolution).
- `ABCDETurnerSampler(K=8, p_migration=0.10, p_crossover=0.90, κ=1.0,
   γ2_burnin=0.5, kernel=:gaussian, burnin_frac=0.3)` — Turner &
  Sederberg variant with K-group migration and a kernel proposal.

`ABCDE_Routine` and `ABCDE_Turner_Routine` remain as thin
compatibility shims over `run_abc`. Both accept `target` and the new
loss-specific `test = :auto` default.

All forms require:
- measurement sets
- a reference parameter vector (usually the MLE from `PE_Routine`)
- a prior distribution (use `Distributions.product_distribution`)

For a worked end-to-end example see
[Tutorial 5 — ABCDE and MCMC](https://github.com/DanielePessina/CriSTool.jl/blob/main/examples/Tutorial%205%20ABCDE%20and%20MCMC.jl), which runs
`run_abc` and Turing NUTS on the same synthetic dataset and compares
posteriors side-by-side.

## Example: `run_abc` with the standard ABCDE sampler

```julia
using CriSTool
using Distributions
using Metaheuristics

path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.csv")
measurements = load_measurements(path)

nucl_f, growth_f = nucl_CNT(), growth_empirical()
PE_lb = [10.0, 0.00015, 0.3e-9 / 60, 1.0]
PE_ub = [65.0, 0.0025, 3.0e-9 / 60, 3.5]

solver = MoM()
lossfn = logMLE()

optres = PE_Routine(lossfn, measurements, PE_lb, PE_ub,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = solver)

optimal_params = minimizer(optres)

prior = Distributions.product_distribution([
    Uniform(PE_lb[1], PE_ub[1]),
    Uniform(PE_lb[2], PE_ub[2]),
    Uniform(PE_lb[3], PE_ub[3]),
    Uniform(PE_lb[4], PE_ub[4]),
])

res, meta = run_abc(lossfn, measurements, optimal_params, prior,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = solver,
                    sampler = ABCDESampler(),
                    nparticles = 512,
                    generations = 256,
                    confidenceinterval = 0.95,
                    test = :auto)
```

## Example: Turner variant via `run_abc`

```julia
res, meta = run_abc(lossfn, measurements, optimal_params, prior,
                    nucl_f, growth_f, noaggregation(), nobreakage();
                    solver = solver,
                    sampler = ABCDETurnerSampler(K = 8,
                                                 p_migration = 0.10,
                                                 p_crossover = 0.90,
                                                 kernel = :gaussian),
                    nparticles = 256,
                    generations = 1024,
                    confidenceinterval = 0.95,
                    test = :wilks)
```

## Example: legacy `ABCDE_Routine` shim

```julia
res, meta = ABCDE_Routine(lossfn, measurements, optimal_params, prior,
                          nucl_f, growth_f, noaggregation(), nobreakage();
                          solver = solver,
                          nparticles = 512,
                          generations = 256,
                          confidenceinterval = 0.95,
                          test = :auto)
```

## Adding a new sampler

`run_abc` is open to additional ABC algorithms. To add one:

1. Define a new `Base.@kwdef struct MySampler <: AbstractABCSampler ... end`
   holding the algorithm-specific knobs.
2. Implement `_runsampler(::MySampler, prior, lossfn, target,
   optimallossfunction; nparticles, generations, HPC, earlystop)`
   returning `(res, reached_ϵ)`.
3. Implement label and metadata helpers: `_routine_label`,
   `_panel_suffix`, `_save_suffix`, `_sampler_metadata`.

`run_abc` then dispatches into your sampler when callers pass
`sampler = MySampler(...)`.

## Notes

- The default `test = :auto` derives an additive Wilks target only for
  unit-weight `logMLE`: `reference_loss + quantile(Chisq(nparameters), confidenceinterval)/2`.
  Negative Gaussian NLL values are valid; multiplying them by an F factor is not.
- `:wilks`, `:chisq`, and `:chisqtest` select the same likelihood-ratio policy.
  It assumes a regular identifiable likelihood, an interior maximum-likelihood
  estimate, and sufficient data for the asymptotic approximation. Boundary
  parameters, non-identifiability, and small samples can invalidate its coverage.
- MAE, custom discrepancies, and weighted likelihoods require an explicit finite
  `target` at least as large as the loss at the reference parameters. For example,
  `run_abc(mae(), setup, optimal_params, prior; target = 0.2)` uses the caller's
  discrepancy tolerance without claiming confidence coverage.
- `:f`, `:fstat`, and `:ftest` are rejected for automatic target derivation:
  neither current loss is a residual-sum-of-squares discrepancy. Explicit `target`
  overrides derivation.
- `run_abc(lossfn, setup::LossSetup, optimal_params, prior; ...)` preserves the
  configured system, experiment preparation, and solve options. Included
  observation counts follow the loss's selection; excluded initial concentration
  points and zero-weight observables are not counted. The result metadata records
  `target_policy` and `included_observations`.
- Prefer `product_distribution` or
  `create_product_prior`.
- Use `validation = ...` to run posterior predictive checks on extra data.
  Supply a prepared `LossSetup` to preserve custom validation solve settings
  and observation selection, or supply experiments to prepare them with the
  configured system.
- Pass `outputdir = ...` to persist the posterior object (`.jld2`) and
  plots. With the default `outputdir = nothing` the routine performs no
  filesystem writes (it never writes into the current working directory).
