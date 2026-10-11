# Inference and uncertainty

```@docs
OptimisationSpec
BayesianSpec
ABCSpec
PreparedFit
prepare_fit
fit
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
ABCDESampler
ABCDETurnerSampler
rename_chain
PredictionEnsemble
prediction_summary
```

Local parameter sensitivities are ForwardDiff derivatives of
`solve(remake(problem; p = values), solver)`; see [Sensitivity analysis](../sensitivity.md).
