## Saturation models
##############################################################

"""
    AbstractSaturationModel

Abstract supertype for solubility/saturation models (see
`ConstantSaturation`, `PolynomialSaturation`, `CallableSaturation`).
"""
abstract type AbstractSaturationModel end

"""
    lysozyme_saturation() -> PolynomialSaturation

The legacy lysozyme solubility polynomial (kg/m³, temperature in °C):
`0.3705 + 7.171e-2 ΔT - 1.924e-3 ΔT² + 17.97e-5 ΔT³`, with
`ΔT = T_K - 273.15`. This is the default `CrystallisationProblem`
saturation model (backward compatible with the original hardcoded curve).
"""
lysozyme_saturation() =
    PolynomialSaturation(; coeffs = [0.3705, 7.171e-2, -1.924e-3, 17.97e-5])

"""
    ConstantSaturation{T<:Real} <: AbstractSaturationModel

Constant solubility `value` (kg/m³), independent of temperature and time.
"""
Base.@kwdef @concrete struct ConstantSaturation{T <: Real} <: AbstractSaturationModel
    value::T
end

"""
    PolynomialSaturation{T<:Real} <: AbstractSaturationModel

Solubility as a polynomial in `T - Tref` (default `Tref = 273.15`, i.e.
temperature in Celsius): `coeffs[1] + coeffs[2] x + coeffs[3] x² + ...`,
evaluated with Horner's scheme.
"""
Base.@kwdef @concrete struct PolynomialSaturation{T <: Real} <: AbstractSaturationModel
    coeffs::Vector{T}
    Tref::T = 273.15
end

"""
    CallableSaturation{F} <: AbstractSaturationModel

Arbitrary solubility as a user function `f(T_K, t)` of temperature (K) and
time (minutes).
"""
Base.@kwdef @concrete struct CallableSaturation{F} <: AbstractSaturationModel
    f::F
end

"""
    saturation_concentration(sm::AbstractSaturationModel, temp_profile, t) -> Real

Solubility (kg/m³) at time `t` under the temperature profile `temp_profile`.
"""
saturation_concentration(sm::ConstantSaturation, temp_profile, t) = sm.value

function saturation_concentration(sm::PolynomialSaturation, temp_profile, t)
    x = temperature(temp_profile, t) - sm.Tref
    c = sm.coeffs
    acc = c[end]
    @inbounds for i in (length(c) - 1):-1:1
        acc = acc * x + c[i]
    end
    return acc
end

saturation_concentration(sm::CallableSaturation, temp_profile, t) =
    sm.f(temperature(temp_profile, t), t)

"""
    saturation_concentration(sm::AbstractSaturationModel, temp_profile) -> Real

Solubility at time `t = 0` under the temperature profile.
"""
saturation_concentration(sm::AbstractSaturationModel, temp_profile) =
    saturation_concentration(sm, temp_profile, 0.0)
