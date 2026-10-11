# Inference and uncertainty

`EstimationProblem` follows the SciML problem/solve pattern: it holds the
model, experiments, selected parameters, loss and simulation settings, and
`solve(problem, alg)` dispatches on the estimation algorithm. The Turing method
`solve(problem, NUTS(); n_samples, n_chains, rng, parallel)` is defined by the
`CriSToolTuringExt` extension (`using Turing`).

```@docs
EstimationProblem
remake(::EstimationProblem)
solve(::EstimationProblem, ::Any)
solve(::EstimationProblem, ::CriSTool.Metaheuristics.Algorithm)
solve(::EstimationProblem, ::AbstractABCSampler)
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
rename_chain
PredictionEnsemble
prediction_summary
```

Local parameter sensitivities are ForwardDiff derivatives of
`solve(remake(problem; p = values), solver)`; see [Sensitivity analysis](../sensitivity.md).
