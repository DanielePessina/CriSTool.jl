# Optimisation routines

This guide focuses on optimization-based parameter estimation using
`PE_Routine_Optimisation`. It wraps Optimization.jl algorithms and uses
the same loss functions and measurement data as `PE_Routine`.

The adapter packages used in the example (`OptimizationBBO`,
`OptimizationOptimJL`, and `Optim`) must be available in the environment where
the example is run. They are not part of the package's core dependencies; add
the adapters you need to your project before using this workflow.

## Example: compare BBO vs LBFGS

```julia
using CriSTool
using OptimizationBBO
using OptimizationOptimJL
using Optim

path = joinpath(pkgdir(CriSTool), "examples", "fake-experimental-dataset.csv")
measurements = load_measurements(path)

nucl_f = nucl_CNT()
growth_f = growth_energy()
PE_lb = [10.0, 0.15, -10.0, 1.0]
PE_ub = [65.0, 2.5, 10.0, 3.5]

solver = MoM()
lossfn = logMLE(weighting = [1.0, 1.0])

# 1) BlackBoxOptim adaptive DE
res_bbo = PE_Routine_Optimisation(lossfn, measurements, PE_lb, PE_ub,
                                  nucl_f, growth_f, noaggregation(), nobreakage();
                                  solver = solver,
                                  searchalgo = BBO_adaptive_de_rand_1_bin_radiuslimited(),
                                  searchoptions = Dict(:PopulationSize => 64))

# 2) LBFGS with box constraints
res_lbfgs = PE_Routine_Optimisation(lossfn, measurements, PE_lb, PE_ub,
                                    nucl_f, growth_f, noaggregation(), nobreakage();
                                    solver = solver,
                                    searchalgo = Optim.Fminbox(Optim.LBFGS()))

@show res_bbo.u res_bbo.objective
@show res_lbfgs.u res_lbfgs.objective
```

## Notes

- `PE_Routine_Optimisation` internally builds an
  `OptimizationFunction` around `loss(lossfn, problem, x, experiments)`.
- Use the same bounds and parameter ordering as in `PE_Routine`.
- For large datasets, set `verbosity = 0` and consider `HPC = true` to
  reduce logging.
