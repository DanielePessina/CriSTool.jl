"""Initial crystal-population descriptions in canonical SI units."""
abstract type AbstractInitialCrystals end

"""
    LogNormalInitialCrystals(; mass_concentration, d43, geometric_std)

Lognormal initial crystal population. `mass_concentration` is kg/m^3, `d43`
is m, and `geometric_std` is dimensionless and greater than one.
"""
Base.@kwdef struct LogNormalInitialCrystals <: AbstractInitialCrystals
    mass_concentration::Float64
    d43::Float64
    geometric_std::Float64
end

"""
    GaussianInitialCrystals(; mass_concentration, d43, standard_deviation)

Positive-truncated Gaussian initial crystal population. `mass_concentration`
is kg/m^3 and both `d43` and `standard_deviation` are m.
"""
Base.@kwdef struct GaussianInitialCrystals <: AbstractInitialCrystals
    mass_concentration::Float64
    d43::Float64
    standard_deviation::Float64
end
