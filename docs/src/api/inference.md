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
predict
PredictionResult
measurement_samples
AbstractABCSampler
ABCDESampler
ABCDETurnerSampler
nuts_model
kinetic_parameter_symbols
rename_chain
chains_to_matrix
distribution_to_matrix
create_product_prior
prior_to_matrix
PredictionEnsemble
prediction_summary
```

`forwardsensitivity` is a lower-level qualified helper rather than an
exported binding. See [Sensitivity analysis](../sensitivity.md) for its
supported solver and kinetic subset.
