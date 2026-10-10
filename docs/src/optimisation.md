# Optimisation algorithms

`OptimisationSpec(; bounds, algorithm, loss, adtype, searchoptions)` accepts
configured Metaheuristics algorithms or Optimization.jl algorithms; run it with
`fit(prepare_fit(model, experiments, spec); threaded=false)`.
The selection is resolved by named bounds; all other kinetic values are fixed.

For Metaheuristics, supply a configured `DE`, `PSO`, `SA` or `NSGA2`, with explicit
options and reproducibility seed. NSGA2 returns a vector of `FitResult`s, one per
Pareto candidate; explicitly choose a candidate before predictions.
`threaded=true` evaluates each Metaheuristics population in parallel.
For Optimization.jl, supply `OptimizationOptimJL.LBFGS()` as `algorithm`, with
`adtype` and `searchoptions` in the same specification.
The prepared objective is `loss(prepared_fit, selected_vector)` for advanced use.

Keep algorithm search options separate from forward `solve_options`. All native
backend diagnostics remain available in `backend_result`. See
[Parameter estimation](parameter-estimation.md) for the model workflow.
