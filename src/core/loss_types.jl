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

Use the measured variance, falling back to a relative 10% standard deviation
when the observable has no variance.
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
    logMLE <: AbstractPELossFunction

Log Maximum Likelihood Estimation loss function.

Fields:
- `weighting::Vector{Float64}`: Weighting factors by observable (default: concentration and size both 1.0)
- `variance_model::AbstractVarianceModel`: Variance policy (default: measured variance)
- `relative_variance_floor::Float64`: Dimensionless minimum variance fraction
  relative to the squared observable magnitude
- `string::String`: String identifier
- `symbols::Vector{Symbol}`: Parameter symbols [:logMLE]
"""
Base.@kwdef @concrete struct logMLE <: AbstractPELossFunction
    weighting::Vector{Float64} = [1.0, 1.0]
    variance_model::AbstractVarianceModel = MeasuredVariance()
    relative_variance_floor::Float64 = 1e-12
    string::String = weighting == [1.0, 1.0] ? "Log MLE" : "Log MLE wgted $(weighting)"
    symbols::Vector{Symbol} = [:logMLE]
end

"""
    mae <: AbstractPELossFunction

Mean Absolute Error loss function.

Fields:
- `weighting::Vector{Float64}`: Weighting factors by observable (default: concentration and size both 1.0)
- `string::String`: String identifier
"""
Base.@kwdef @concrete struct mae <: AbstractPELossFunction
    weighting::Vector{Float64} = [1.0, 1.0]
    string::String = weighting == [1.0, 1.0] ? "MAE" : "MAE wgted $(weighting)"
end
