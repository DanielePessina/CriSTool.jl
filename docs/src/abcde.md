# ABC sampling

ABCDE solves the same `EstimationProblem` as optimisation and Bayesian
inference. Select parameters by explicit priors, choose the discrepancy as the
problem's `loss`, and give an explicit acceptance threshold to `solve`:

```julia
using CriSTool, Distributions
abc_problem = EstimationProblem(crystal_model, experiments;
    priors = (growth = (growth_coefficient = Uniform(1e-12, 1e-9),),),
    loss = MeanAbsoluteError(), solver = MoM())
abc_posterior = solve(abc_problem, ABCDE(); target = 0.1, n_particles = 128, n_generations = 100,
    parallel = EnsembleThreads())
abc_draws = abc_posterior.samples
abc_predictions = predict(abc_posterior)
```

`target` is required and must be a finite number. Thresholds have the
units/scale of the discrepancy and are a scientific choice; they are not
inferred from an optimum or from a confidence interval. `stopping = StopAtTarget()` stops
once every particle is within the target; the default `AllGenerations()` runs every generation. `parallel = EnsembleThreads()` evaluates particles in parallel; the experiments
of each discrepancy evaluation then run serially.
`ABCDETurner()` is the Turner and Sederberg variant, with its migration and
kernel settings as fields; `kernel` is `GaussianABCKernel()` (default), `LaplaceABCKernel()` or `FlatABCKernel()`. Other samplers subtype `AbstractABCSampler`. Native posterior output
is retained as `backend_result`; joint draws use the common prediction interface.
The sampler implementation is vendored in `CriSTool.KissABC`.
