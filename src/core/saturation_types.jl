## Solubility models
##############################################################

"""
    AbstractSolubilityModel

Abstract supertype for solubility/saturation models (see
`ConstantSolubility`, `PolynomialSolubility`, `CallableSolubility`).
"""
abstract type AbstractSolubilityModel end

"""
    lysozyme_solubility() -> PolynomialSolubility

The legacy lysozyme solubility polynomial (kg/m³, temperature in °C):
`0.3705 + 7.171e-2 ΔT - 1.924e-3 ΔT² + 17.97e-5 ΔT³`, with
`ΔT = T_K - 273.15`. This is the default `CrystallisationProblem`
saturation model (backward compatible with the original hardcoded curve).
"""
lysozyme_solubility() =
    PolynomialSolubility(; coeffs = [0.3705, 7.171e-2, -1.924e-3, 17.97e-5])

"""
    ConstantSolubility{T<:Real} <: AbstractSolubilityModel

Constant solubility `value` (kg/m³), independent of temperature and time.
"""
Base.@kwdef @concrete struct ConstantSolubility{T <: Real} <: AbstractSolubilityModel
    value::T
end

"""
    PolynomialSolubility{T<:Real} <: AbstractSolubilityModel

Solubility as a polynomial in `T - Tref` (default `Tref = 273.15`, i.e.
temperature in Celsius): `coeffs[1] + coeffs[2] x + coeffs[3] x² + ...`,
evaluated with Horner's scheme.
"""
Base.@kwdef @concrete struct PolynomialSolubility{T <: Real} <: AbstractSolubilityModel
    coeffs::Vector{T}
    Tref::T = 273.15
end

"""
    CallableSolubility{F} <: AbstractSolubilityModel

Arbitrary solubility as a user function `f(T_K, t)` of temperature (K) and
time (minutes).
"""
Base.@kwdef @concrete struct CallableSolubility{F} <: AbstractSolubilityModel
    f::F
end

"""
    saturation_concentration(sm::AbstractSolubilityModel, temp_profile, t) -> Real

Solubility (kg/m³) at time `t` under the temperature profile `temp_profile`.
"""
saturation_concentration(sm::ConstantSolubility, temp_profile, t) = sm.value

function saturation_concentration(sm::PolynomialSolubility, temp_profile, t)
    x = temperature(temp_profile, t) - sm.Tref
    c = sm.coeffs
    acc = c[end]
    @inbounds for i in (length(c) - 1):-1:1
        acc = acc * x + c[i]
    end
    return acc
end

saturation_concentration(sm::CallableSolubility, temp_profile, t) =
    sm.f(temperature(temp_profile, t), t)

"""
    saturation_concentration(sm::AbstractSolubilityModel, temp_profile) -> Real

Solubility at time `t = 0` under the temperature profile.
"""
saturation_concentration(sm::AbstractSolubilityModel, temp_profile) =
    saturation_concentration(sm, temp_profile, 0.0)

# Compatibility names retained for existing callers.  The canonical public
# vocabulary is now Solubility because these models return c*(T), not S.
const AbstractSaturationModel = AbstractSolubilityModel
const ConstantSaturation = ConstantSolubility
const PolynomialSaturation = PolynomialSolubility
const CallableSaturation = CallableSolubility
const lysozyme_saturation = lysozyme_solubility
