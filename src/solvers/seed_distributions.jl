"""
Seed-distribution integrals: number moments, weighting conversion, mesh cell
integrals and domain diagnostics for `DistributionInitialCrystals`.
"""

"""Analytic full-support raw moment of a source distribution, or `nothing`."""
function _analytic_full_support_moment(length_distribution::LogNormal, moment_power::Integer)
    log_location, log_width = Distributions.params(length_distribution)
    analytic_moment = exp(moment_power * log_location + 0.5 * moment_power^2 * log_width^2)
    isfinite(analytic_moment) && analytic_moment > 0 ||
        throw(ArgumentError("Seed moment $moment_power is unavailable."))
    return analytic_moment
end
_analytic_full_support_moment(length_distribution, moment_power::Integer) = nothing

"""
    _seed_integral(seed, moment_power[, lower_length, upper_length])

`∫ pdf(L) L^moment_power dL` of the seed's source distribution over the given
length interval (default: its whole support).
"""
function _seed_integral(seed_description::DistributionInitialCrystals, moment_power::Integer,
        lower_length = minimum(seed_description.distribution),
        upper_length = maximum(seed_description.distribution))
    length_distribution = seed_description.distribution
    lower_length = max(lower_length, minimum(length_distribution))
    upper_length = min(upper_length, maximum(length_distribution))
    lower_length >= upper_length && return 0.0
    if lower_length == minimum(length_distribution) && upper_length == maximum(length_distribution)
        analytic_moment = _analytic_full_support_moment(length_distribution, moment_power)
        analytic_moment === nothing || return analytic_moment
    end
    if moment_power < 0 && lower_length == 0 && pdf(length_distribution, 0.0) > 0
        throw(ArgumentError("Volume-to-number conversion diverges at zero; supply an explicit positive cutoff."))
    end
    # Integrate in dimensionless length coordinates to resolve micron-scale
    # distributions on an unbounded domain without an implicit lower cutoff.
    length_scale = quantile(length_distribution, 0.5)
    isfinite(length_scale) && length_scale > 0 ||
        throw(ArgumentError("Seed distribution must have a finite positive median."))
    # QuadGK cannot integrate a non-finite integrand value. Record any such
    # sample and integrate zero in its place, then reject the moment below.
    nonfinite_integrand = Ref(false)
    scaled_integrand = scaled_length -> begin
        physical_length = length_scale * scaled_length
        integrand_value = pdf(length_distribution, physical_length) * length_scale *
                          scaled_length^moment_power
        isfinite(integrand_value) && return integrand_value
        nonfinite_integrand[] = true
        return zero(integrand_value)
    end
    scaled_value, integral_error = quadgk(scaled_integrand, lower_length / length_scale,
        upper_length / length_scale;
        rtol = seed_description.quadrature_rtol, atol = 0.0, maxevals = 100_000)
    nonfinite_integrand[] &&
        throw(ArgumentError("Seed moment $moment_power is not finite/resolvable; supply valid support or explicit truncation."))
    isfinite(scaled_value) && scaled_value >= 0 &&
        integral_error <= seed_description.quadrature_rtol * abs(scaled_value) ||
        throw(ArgumentError("Seed moment $moment_power could not be resolved; check support and tail behaviour."))
    return scaled_value * length_scale^moment_power
end

"""
    _seed_raw_moment(seed, moment_power)

Normalised number moment `E_n[L^moment_power]` of the seed population (m^k).
"""
function _seed_raw_moment(seed_description::DistributionInitialCrystals, moment_power::Integer)
    density_power = _number_density_power(seed_description)
    return _seed_integral(seed_description, moment_power + density_power) /
           _seed_integral(seed_description, density_power)
end

# Normalised density adapter for general volume-weighted sources.
struct NumberWeightedDistribution{Source, Normalisation} <: Distributions.ContinuousUnivariateDistribution
    source::Source
    normalisation::Normalisation
end
Base.minimum(converted_density::NumberWeightedDistribution) = minimum(converted_density.source)
Base.maximum(converted_density::NumberWeightedDistribution) = maximum(converted_density.source)
Distributions.pdf(converted_density::NumberWeightedDistribution, crystal_length::Real) =
    crystal_length > 0 ? pdf(converted_density.source, crystal_length) /
        (crystal_length^3 * converted_density.normalisation) : 0.0

function Distributions.cdf(converted_density::NumberWeightedDistribution, crystal_length::Real)
    crystal_length <= minimum(converted_density) && return 0.0
    crystal_length >= maximum(converted_density) && return 1.0
    integral_value, integral_error = quadgk(length_value -> pdf(converted_density, length_value),
        minimum(converted_density), crystal_length; rtol = 1e-8)
    return integral_value
end
function Distributions.quantile(converted_density::NumberWeightedDistribution, probability_value::Real)
    0 <= probability_value <= 1 || throw(ArgumentError("Probability must be in [0,1]."))
    probability_value == 0 && return minimum(converted_density)
    probability_value == 1 && return maximum(converted_density)
    lower_length = minimum(converted_density)
    upper_length = quantile(converted_density.source, 0.99)
    for expansion_index in 1:100
        cdf(converted_density, upper_length) >= probability_value && break
        upper_length *= 2
    end
    cdf(converted_density, upper_length) >= probability_value ||
        throw(ArgumentError("Cannot bracket converted seed quantile."))
    for iteration_index in 1:80
        middle_length = (lower_length + upper_length) / 2
        if cdf(converted_density, middle_length) < probability_value
            lower_length = middle_length
        else
            upper_length = middle_length
        end
    end
    return (lower_length + upper_length) / 2
end

"""
    number_weighted(seed)

Return the same physical seed expressed as a number-weighted distribution.
For a volume-weighted lognormal the conversion is analytic; other distributions
use a normalised density adapter with checked improper integrals.
"""
number_weighted(seed_description::DistributionInitialCrystals{<:Any, <:Any, NumberWeighted}) =
    seed_description
function number_weighted(seed_description::DistributionInitialCrystals{<:Any, <:Any, VolumeWeighted})
    return DistributionInitialCrystals(
        _number_weighted_distribution(seed_description.distribution, seed_description);
        mass_concentration = seed_description.mass_concentration,
        quadrature_rtol = seed_description.quadrature_rtol,
        mass_loss_tolerance = seed_description.mass_loss_tolerance)
end

function _number_weighted_distribution(length_distribution::LogNormal, seed_description)
    log_location, log_width = Distributions.params(length_distribution)
    return LogNormal(log_location - 3 * log_width^2, log_width)
end
_number_weighted_distribution(length_distribution, seed_description) =
    NumberWeightedDistribution(length_distribution, _seed_integral(seed_description, -3))

_validate_initial_crystals(seed_description::DistributionInitialCrystals) = seed_description

"""`seed_domain_diagnostics(problem, seed)` reports omitted number and mass fractions."""
seed_domain_diagnostics(configured_run::CrystallisationProblem,
                        seed_description::DistributionInitialCrystals) =
    _seed_domain_diagnostics(configured_run.solver, seed_description)

_seed_domain_diagnostics(solver::AbstractMomentSolver, seed_description) =
    (; omitted_number_fraction = 0.0, omitted_mass_fraction = 0.0)
function _seed_domain_diagnostics(solver::AbstractDiscretisedSolver, seed_description)
    density_power = _number_density_power(seed_description)
    lower_length, upper_length = solver.lmin, solver.lmax
    number_fraction = _seed_integral(seed_description, density_power, lower_length, upper_length) /
        _seed_integral(seed_description, density_power)
    mass_fraction = _seed_integral(seed_description, 3 + density_power, lower_length, upper_length) /
        _seed_integral(seed_description, 3 + density_power)
    return (; omitted_number_fraction = max(0.0, 1 - number_fraction),
        omitted_mass_fraction = max(0.0, 1 - mass_fraction))
end

function _seed_population(solver::AbstractDiscretisedSolver, configured_run::CrystallisationProblem,
        seed_description::DistributionInitialCrystals)
    domain_report = _seed_domain_diagnostics(solver, seed_description)
    domain_report.omitted_mass_fraction <= seed_description.mass_loss_tolerance ||
        throw(ArgumentError("Seed outside mesh: $(domain_report.omitted_number_fraction) number fraction, " *
            "$(domain_report.omitted_mass_fraction) mass fraction; enlarge the mesh or explicitly truncate."))
    density_power = _number_density_power(seed_description)
    number_scale = seed_description.mass_concentration /
        (_solid_mass_factor(configured_run) *
            _seed_integral(seed_description, 3 + density_power))
    return [number_scale * _seed_integral(seed_description, density_power,
        solver.cell_face[cell_index], solver.cell_face[cell_index + 1]) /
        solver.cell_dL for cell_index in eachindex(solver.cell_centre)]
end

function _resolved_condition_value(seed_description::DistributionInitialCrystals, observation_lookup)
    seed_description.mass_concentration isa FromObservation || return seed_description
    resolved_mass = observation_lookup(seed_description.mass_concentration.observable_name)
    return DistributionInitialCrystals(seed_description.distribution,
        Float64(_validate_seed_mass(resolved_mass)), seed_description.weighting,
        seed_description.quadrature_rtol, seed_description.mass_loss_tolerance, seed_description.d43)
end
