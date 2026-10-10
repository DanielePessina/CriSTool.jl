## Loss-function structs
"""
    AbstractPELossFunction

Abstract supertype for parameter estimation loss functions.
"""
abstract type AbstractPELossFunction end

"""
    AbstractVarianceModel

Strategy used by likelihood losses to obtain an observable variance when a
measurement does not provide one or when a relative-error policy is desired.
"""
abstract type AbstractVarianceModel end

"""
    MeasuredVariance()

Use supplied strictly positive measurement variances. Missing or zero variance
on a scored measurement is an error; noise is never fitted or invented.
"""
struct MeasuredVariance <: AbstractVarianceModel end

"""
    RelativeVariance(percent)

Use `percent` of the absolute measured value as the standard deviation.
"""
struct RelativeVariance{T <: Real} <: AbstractVarianceModel
    percent::T
end

"""
    logMLE(; weighting = (;), variance_model = MeasuredVariance(), relative_variance_floor = 1e-12)

Gaussian negative log likelihood summed over every scored measurement.

Fields:
- `weighting`: factor per observable, as a `NamedTuple` or dictionary keyed by
  observable name, or a vector in the prepared observable order. Unnamed
  observables get 1. Weights are resolved once when the loss is prepared.
- `variance_model::AbstractVarianceModel`: variance policy (default: measured variance)
- `relative_variance_floor::Float64`: dimensionless minimum variance fraction
  relative to the squared observable magnitude
"""
Base.@kwdef @concrete struct logMLE <: AbstractPELossFunction
    weighting = (;)
    variance_model::AbstractVarianceModel = MeasuredVariance()
    relative_variance_floor::Float64 = 1e-12
end

"""
    mae(; weighting = (;))

Mean absolute error, pooled over every scored measurement of each observable.

Fields:
- `weighting`: factor per observable, with the same forms as for `logMLE`.
"""
Base.@kwdef @concrete struct mae <: AbstractPELossFunction
    weighting = (;)
end
