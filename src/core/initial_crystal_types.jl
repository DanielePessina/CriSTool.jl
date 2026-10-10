"""Initial crystal-population descriptions in canonical SI units."""
abstract type AbstractInitialCrystals end

"""
    FromObservation(observable_name)

Placeholder for a run condition read from an experiment's time-zero
observation of `observable_name`. It may stand for the initial concentration,
another solvent value, or any numeric seed field. Placeholders are resolved per
experiment, and the time-zero observation they consume is not scored.
"""
struct FromObservation
    observable_name::Symbol
end

_seed_field_value(field_value::Real) = Float64(field_value)
_seed_field_value(field_value::FromObservation) = field_value

"""
    LogNormalInitialCrystals(; mass_concentration, d43, geometric_std)

Lognormal initial crystal population. `mass_concentration` is kg/m^3, `d43`
is m, and `geometric_std` is dimensionless and greater than one. Any field may
be a `FromObservation` placeholder inside experiment conditions.
"""
struct LogNormalInitialCrystals{Mass, D43, Width} <: AbstractInitialCrystals
    mass_concentration::Mass
    d43::D43
    geometric_std::Width
end
LogNormalInitialCrystals(; mass_concentration, d43, geometric_std) = LogNormalInitialCrystals(
    _seed_field_value(mass_concentration), _seed_field_value(d43), _seed_field_value(geometric_std))

"""
    GaussianInitialCrystals(; mass_concentration, d43, standard_deviation)

Positive-truncated Gaussian initial crystal population. `mass_concentration`
is kg/m^3 and both `d43` and `standard_deviation` are m. Any field may be a
`FromObservation` placeholder inside experiment conditions.
"""
struct GaussianInitialCrystals{Mass, D43, Width} <: AbstractInitialCrystals
    mass_concentration::Mass
    d43::D43
    standard_deviation::Width
end
GaussianInitialCrystals(; mass_concentration, d43, standard_deviation) = GaussianInitialCrystals(
    _seed_field_value(mass_concentration), _seed_field_value(d43), _seed_field_value(standard_deviation))
