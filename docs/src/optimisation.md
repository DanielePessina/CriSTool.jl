# Optimisation algorithms

`solve(EstimationProblem(model, experiments; bounds, loss), alg; parallel=EnsembleSerial())`
accepts configured Metaheuristics algorithms or Optimization.jl optimisers as
`alg`. The selection is resolved by named bounds; all other kinetic values are fixed.

For Metaheuristics, supply a configured `DE`, `PSO`, `SA` or `NSGA2`, with explicit
options and reproducibility seed. NSGA2 returns a vector of `FitResult`s, one per
Pareto candidate; explicitly choose a candidate before predictions.
`parallel=EnsembleThreads()` evaluates each Metaheuristics population in parallel;
for Optimization.jl algorithms it solves the experiments of each loss evaluation
on threads instead.
For Optimization.jl, pass for example `OptimizationOptimJL.LBFGS()` as `alg`:
`solve(problem, LBFGS(); adtype = AutoForwardDiff(), maxiters = 100)`. `adtype`
differentiates the loss and the remaining keywords go to the Optimization.jl
`solve`. The objective is `loss(problem, selected_vector)` for advanced use.

Keep algorithm search options separate from forward `solve_options`. All native
backend diagnostics remain available in `backend_result`. See
[Parameter estimation](parameter-estimation.md) for the model workflow.
