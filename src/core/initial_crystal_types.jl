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

"""
    DistributionInitialCrystals(distribution; mass_concentration, weighting=:number,
        quadrature_rtol=1e-8, mass_loss_tolerance=1e-4)

A continuous distribution of positive lengths in metres, weighted by number
(`weighting = :number`) or volume (`weighting = :volume`), and solid mass
concentration in kg/m³. No hidden truncation is applied.

The weighting is stored as `number_density_power`: the number density is
proportional to `pdf(distribution, L) * L^number_density_power` (0 for number
weighting, -3 for volume weighting). `d43` is resolved once on construction.
"""
struct DistributionInitialCrystals{Distribution, Mass} <: AbstractInitialCrystals
    distribution::Distribution
    mass_concentration::Mass
    number_density_power::Int
    quadrature_rtol::Float64
    mass_loss_tolerance::Float64
    d43::Float64
end

_number_density_power(weighting::Symbol) =
    weighting === :number ? 0 : weighting === :volume ? -3 :
    throw(ArgumentError("weighting must be :number or :volume."))

function DistributionInitialCrystals(length_distribution::Distributions.ContinuousUnivariateDistribution;
        mass_concentration::Union{Real, FromObservation}, weighting::Symbol = :number,
        quadrature_rtol::Real = 1e-8, mass_loss_tolerance::Real = 1e-4)
    number_density_power = _number_density_power(weighting)
    _validate_seed_mass(mass_concentration)
    minimum(length_distribution) >= 0 || throw(ArgumentError(
        "Length distribution has negative support; truncate it explicitly."))
    maximum(length_distribution) > 0 || throw(ArgumentError("Length distribution has no positive support."))
    0 < quadrature_rtol < 1 && 0 <= mass_loss_tolerance < 1 ||
        throw(ArgumentError("Invalid quadrature or mass-loss tolerance."))
    # The characteristic size is resolved once; it also verifies the required integrals.
    unsized_seed = DistributionInitialCrystals(length_distribution, _seed_field_value(mass_concentration),
        number_density_power, Float64(quadrature_rtol), Float64(mass_loss_tolerance), NaN)
    seed_d43 = _seed_raw_moment(unsized_seed, 4) / _seed_raw_moment(unsized_seed, 3)
    return DistributionInitialCrystals(length_distribution, unsized_seed.mass_concentration,
        number_density_power, Float64(quadrature_rtol), Float64(mass_loss_tolerance), seed_d43)
end

_validate_seed_mass(mass_concentration::FromObservation) = mass_concentration
_validate_seed_mass(mass_concentration::Real) =
    isfinite(mass_concentration) && mass_concentration > 0 ? mass_concentration :
    throw(ArgumentError("Distribution seeds require positive finite mass_concentration."))
