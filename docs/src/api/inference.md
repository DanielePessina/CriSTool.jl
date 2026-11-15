# Inference and uncertainty

`EstimationProblem` follows the SciML problem/solve pattern: it holds the
model, experiments, selected parameters, loss and simulation settings, and
`solve(problem, alg)` dispatches on the optimiser, while
`sample([rng], problem, sampler, n)` draws from the posterior. The Turing
methods `sample([rng], problem, NUTS(), n_samples)` and
`sample([rng], problem, NUTS(), MCMCThreads(), n_samples, n_chains)` are
defined by the `CriSToolTuringExt` extension (`using Turing`).

```@docs
EstimationProblem
remake(::EstimationProblem)
solve(::EstimationProblem, ::Any)
solve(::EstimationProblem, ::CriSTool.Metaheuristics.Algorithm)
sample(::EstimationProblem, ::AbstractABCSampler, ::Integer)
solve(::EstimationProblem, ::Union{AbstractABCSampler, CriSTool.AbstractMCMC.AbstractSampler})
FitResult
is_likelihood
loss_terms
InferenceResult
ParameterSamples
parameter_samples
AbstractSimulationMode
TransientMode
SteadyMode
predict
PredictionResult
measurement_samples
AbstractABCSampler
ABCDE
ABCDETurner
AbstractABCStopping
AllGenerations
StopAtTarget
AbstractABCKernel
GaussianABCKernel
LaplaceABCKernel
FlatABCKernel
rename_chain
PredictionEnsemble
prediction_summary
```

Local parameter sensitivities are ForwardDiff derivatives of
`solve(remake(problem; p = values), solver)`; see [Sensitivity analysis](../sensitivity.md).
