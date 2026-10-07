"""
    DistributionInitialCrystals(distribution; mass_concentration, weighting=:number,
        quadrature_rtol=1e-8, mass_loss_tolerance=1e-4)

A continuous distribution of positive lengths in metres, weighted by number or
volume, and solid mass concentration in kg/m³. No hidden truncation is applied.
"""
struct DistributionInitialCrystals{Distribution} <: AbstractInitialCrystals
    distribution::Distribution
    mass_concentration::Float64
    weighting::Symbol
    quadrature_rtol::Float64
    mass_loss_tolerance::Float64
end

function DistributionInitialCrystals(length_distribution::Distributions.ContinuousUnivariateDistribution;
        mass_concentration::Real, weighting::Symbol = :number,
        quadrature_rtol::Real = 1e-8, mass_loss_tolerance::Real = 1e-4)
    weighting in (:number, :volume) || throw(ArgumentError("weighting must be :number or :volume."))
    isfinite(mass_concentration) && mass_concentration > 0 ||
        throw(ArgumentError("Distribution seeds require positive finite mass_concentration."))
    minimum(length_distribution) >= 0 || throw(ArgumentError(
        "Length distribution has negative support; truncate it explicitly."))
    maximum(length_distribution) > 0 || throw(ArgumentError("Length distribution has no positive support."))
    0 < quadrature_rtol < 1 && 0 <= mass_loss_tolerance < 1 ||
        throw(ArgumentError("Invalid quadrature or mass-loss tolerance."))
    seed_description = DistributionInitialCrystals(length_distribution, Float64(mass_concentration),
        weighting, Float64(quadrature_rtol), Float64(mass_loss_tolerance))
    seed_description.d43 # Verify required integrals at construction.
    return seed_description
end

function _seed_integral(seed_description::DistributionInitialCrystals, moment_power::Integer,
        lower_length = minimum(seed_description.distribution),
        upper_length = maximum(seed_description.distribution))
    length_distribution = seed_description.distribution
    lower_length = max(lower_length, minimum(length_distribution))
    upper_length = min(upper_length, maximum(length_distribution))
    lower_length >= upper_length && return 0.0
    if length_distribution isa LogNormal &&
       lower_length == minimum(length_distribution) && upper_length == maximum(length_distribution)
        log_location, log_width = Distributions.params(length_distribution)
        analytic_moment = exp(moment_power * log_location + 0.5 * moment_power^2 * log_width^2)
        isfinite(analytic_moment) && analytic_moment > 0 ||
            throw(ArgumentError("Seed moment $moment_power is unavailable."))
        return analytic_moment
    end
    if moment_power < 0 && lower_length == 0 && pdf(length_distribution, 0.0) > 0
        throw(ArgumentError("Volume-to-number conversion diverges at zero; supply an explicit positive cutoff."))
    end
    # Integrate in dimensionless length coordinates to resolve micron-scale
    # distributions on an unbounded domain without an implicit lower cutoff.
    length_scale = quantile(length_distribution, 0.5)
    isfinite(length_scale) && length_scale > 0 ||
        throw(ArgumentError("Seed distribution must have a finite positive median."))
    scaled_integrand = scaled_length -> begin
        physical_length = length_scale * scaled_length
        pdf(length_distribution, physical_length) * length_scale * scaled_length^moment_power
    end
    scaled_value, integral_error = try
        quadgk(scaled_integrand, lower_length / length_scale, upper_length / length_scale;
            rtol = seed_description.quadrature_rtol, atol = 0.0, maxevals = 100_000)
    catch integral_failure
        integral_failure isa DomainError || rethrow()
        throw(ArgumentError("Seed moment $moment_power is not finite/resolvable; supply valid support or explicit truncation."))
    end
    isfinite(scaled_value) && scaled_value >= 0 &&
        integral_error <= seed_description.quadrature_rtol * abs(scaled_value) ||
        throw(ArgumentError("Seed moment $moment_power could not be resolved; check support and tail behaviour."))
    return scaled_value * length_scale^moment_power
end

function _seed_number_moment(seed_description::DistributionInitialCrystals, moment_power::Integer)
    offset_power = seed_description.weighting === :volume ? -3 : 0
    return _seed_integral(seed_description, moment_power + offset_power) /
           _seed_integral(seed_description, offset_power)
end

function Base.getproperty(seed_description::DistributionInitialCrystals, field_name::Symbol)
    field_name === :d43 && return _seed_number_moment(seed_description, 4) /
        _seed_number_moment(seed_description, 3)
    return getfield(seed_description, field_name)
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
function number_weighted(seed_description::DistributionInitialCrystals)
    seed_description.weighting === :number && return seed_description
    length_distribution = seed_description.distribution
    converted_distribution = if length_distribution isa LogNormal
        log_location, log_width = Distributions.params(length_distribution)
        LogNormal(log_location - 3 * log_width^2, log_width)
    else
        NumberWeightedDistribution(length_distribution, _seed_integral(seed_description, -3))
    end
    return DistributionInitialCrystals(converted_distribution;
        mass_concentration = seed_description.mass_concentration,
        quadrature_rtol = seed_description.quadrature_rtol,
        mass_loss_tolerance = seed_description.mass_loss_tolerance)
end

_validate_initial_crystals(seed_description::DistributionInitialCrystals) = seed_description
function _initial_crystal_distribution_model(seed_description::DistributionInitialCrystals)
    offset_power = seed_description.weighting === :volume ? -3 : 0
    density_normalisation = _seed_integral(seed_description, offset_power)
    return (; raw_moment = moment_power -> _seed_number_moment(seed_description, moment_power),
        density = crystal_length -> crystal_length > 0 ?
            pdf(seed_description.distribution, crystal_length) * crystal_length^offset_power /
                density_normalisation : 0.0)
end

"""`seed_domain_diagnostics(problem, seed)` reports omitted number and mass fractions."""
function seed_domain_diagnostics(configured_run::CrystallisationProblem,
        seed_description::DistributionInitialCrystals)
    configured_run.solver isa AbstractDiscretisedSolver ||
        return (; omitted_number_fraction = 0.0, omitted_mass_fraction = 0.0)
    offset_power = seed_description.weighting === :volume ? -3 : 0
    lower_length, upper_length = configured_run.solver.lmin, configured_run.solver.lmax
    number_fraction = _seed_integral(seed_description, offset_power, lower_length, upper_length) /
        _seed_integral(seed_description, offset_power)
    mass_fraction = _seed_integral(seed_description, 3 + offset_power, lower_length, upper_length) /
        _seed_integral(seed_description, 3 + offset_power)
    return (; omitted_number_fraction = max(0.0, 1 - number_fraction),
        omitted_mass_fraction = max(0.0, 1 - mass_fraction))
end

function _initial_mesh_population(configured_run::CrystallisationProblem,
        seed_description::DistributionInitialCrystals)
    domain_report = seed_domain_diagnostics(configured_run, seed_description)
    domain_report.omitted_mass_fraction <= seed_description.mass_loss_tolerance ||
        throw(ArgumentError("Seed outside mesh: $(domain_report.omitted_number_fraction) number fraction, " *
            "$(domain_report.omitted_mass_fraction) mass fraction; enlarge the mesh or explicitly truncate."))
    offset_power = seed_description.weighting === :volume ? -3 : 0
    number_scale = seed_description.mass_concentration /
        (configured_run.crystal_density * configured_run.volume_shape_factor *
            _seed_integral(seed_description, 3 + offset_power))
    configured_solver = configured_run.solver
    return [number_scale * _seed_integral(seed_description, offset_power,
        configured_solver.cell_face[cell_index], configured_solver.cell_face[cell_index + 1]) /
        configured_solver.cell_dL for cell_index in eachindex(configured_solver.cell_centre)]
end

function _checked_seed_description(seed_description::LogNormalInitialCrystals)
    _validate_initial_crystals(seed_description)
    seed_description.d43 == 0 && return seed_description
    log_width = log(seed_description.geometric_std)
    return DistributionInitialCrystals(LogNormal(log(seed_description.d43) - 3.5 * log_width^2, log_width);
        mass_concentration = seed_description.mass_concentration)
end
function _checked_seed_description(seed_description::GaussianInitialCrystals)
    _validate_initial_crystals(seed_description)
    seed_description.d43 == 0 && return seed_description
    seed_location = _gaussian_mean_for_d43(seed_description.d43, seed_description.standard_deviation)
    return DistributionInitialCrystals(truncated(Normal(seed_location, seed_description.standard_deviation), 0, Inf);
        mass_concentration = seed_description.mass_concentration)
end
_checked_seed_description(seed_description::DistributionInitialCrystals) = seed_description
