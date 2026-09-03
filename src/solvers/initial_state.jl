"""
Initial crystal-population construction from measurable characteristics.

The public interface deliberately describes the population at a high level.
The solver-specific raw moments or mesh values remain an implementation detail.
"""

function _validate_initial_crystals(initial_crystals::AbstractInitialCrystals)
    mass_concentration = initial_crystals.mass_concentration
    d43 = initial_crystals.d43
    isfinite(mass_concentration) && mass_concentration >= 0.0 ||
        throw(ArgumentError("initial crystal mass_concentration must be finite and nonnegative."))
    isfinite(d43) && d43 >= 0.0 ||
        throw(ArgumentError("initial crystal d43 must be finite and nonnegative."))
    if d43 == 0.0
        mass_concentration == 0.0 ||
            throw(ArgumentError("initial crystal d43 = 0 requires mass_concentration = 0."))
        return initial_crystals
    end

    mass_concentration > 0.0 ||
        throw(ArgumentError("positive initial crystal d43 requires positive mass_concentration."))
    if initial_crystals isa LogNormalInitialCrystals
        isfinite(initial_crystals.geometric_std) && initial_crystals.geometric_std > 1.0 ||
            throw(ArgumentError("geometric_std must be finite and greater than 1."))
    else
        isfinite(initial_crystals.standard_deviation) &&
            initial_crystals.standard_deviation > 0.0 ||
            throw(ArgumentError("standard_deviation must be finite and strictly positive."))
    end
    return initial_crystals
end

function _positive_gaussian_raw_moment(mean_value, standard_deviation, order::Integer)
    normal = Normal()
    standardised_mean = mean_value / standard_deviation
    normalisation = cdf(normal, standardised_mean)
    normalisation > 0.0 ||
        throw(ArgumentError("Gaussian initial crystal profile has numerically zero positive support."))

    order == 0 && return one(promote_type(typeof(mean_value), typeof(standard_deviation)))

    previous_previous = normalisation
    previous = mean_value * normalisation +
               standard_deviation * pdf(normal, standardised_mean)
    order == 1 && return previous / normalisation

    current = previous
    for moment_order_value in 2:order
        current = mean_value * previous +
                  (moment_order_value - 1) * standard_deviation^2 * previous_previous
        previous_previous, previous = previous, current
    end
    return current / normalisation
end

function _positive_gaussian_d43(mean_value, standard_deviation)
    return _positive_gaussian_raw_moment(mean_value, standard_deviation, 4) /
           _positive_gaussian_raw_moment(mean_value, standard_deviation, 3)
end

function _gaussian_mean_for_d43(target_d43, standard_deviation)
    # Keep the lower bracket within the numerically stable tail of the
    # standard normal CDF while allowing the upper bracket to cover a target
    # much larger than the requested width.
    lower = -10.0 * standard_deviation
    upper = 10.0 * max(target_d43, standard_deviation)
    target_ratio = target_d43

    lower_ratio = _positive_gaussian_d43(lower, standard_deviation)
    upper_ratio = _positive_gaussian_d43(upper, standard_deviation)
    for _ in 1:12
        lower_ratio <= target_ratio <= upper_ratio && break
        lower *= 2.0
        upper *= 2.0
        lower_ratio = _positive_gaussian_d43(lower, standard_deviation)
        upper_ratio = _positive_gaussian_d43(upper, standard_deviation)
    end
    lower_ratio <= target_ratio <= upper_ratio ||
        throw(ArgumentError("Could not construct a positive Gaussian profile with d43=$target_d43."))

    for _ in 1:100
        midpoint = (lower + upper) / 2.0
        midpoint_ratio = _positive_gaussian_d43(midpoint, standard_deviation)
        if midpoint_ratio < target_ratio
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return (lower + upper) / 2.0
end

function _initial_crystal_distribution_model(initial_crystals::LogNormalInitialCrystals)
    log_spread = log(initial_crystals.geometric_std)
    log_median = log(initial_crystals.d43) - 3.5 * log_spread^2
    lognormal = LogNormal(log_median, log_spread)
    raw_moment = order -> exp(order * log_median + 0.5 * order^2 * log_spread^2)
    density = length_value -> length_value > 0.0 ? pdf(lognormal, length_value) : 0.0
    return (; raw_moment, density)
end

function _initial_crystal_distribution_model(initial_crystals::GaussianInitialCrystals)
    target_d43 = initial_crystals.d43
    standard_deviation = initial_crystals.standard_deviation
    mean_value = _gaussian_mean_for_d43(target_d43, standard_deviation)
    normal = Normal(mean_value, standard_deviation)
    positive_probability = cdf(Normal(), mean_value / standard_deviation)
    raw_moment = order -> _positive_gaussian_raw_moment(mean_value,
                                                        standard_deviation,
                                                        order)
    density = length_value -> length_value > 0.0 ?
                              pdf(normal, length_value) / positive_probability : 0.0
    return (; raw_moment, density)
end

function _validate_initial_crystal_domain(problem::CrystallisationProblem, d43_metres)
    solver = problem.solver
    if solver isa AbstractDiscretisedSolver
        solver.lmin <= d43_metres <= solver.lmax ||
            throw(ArgumentError("initial crystal d43 must lie within the solver size domain " *
                                "[$(solver.lmin), $(solver.lmax)] m."))
    elseif solver isa QMOM
        d43_metres >= solver.minimum_size ||
            throw(ArgumentError("initial crystal d43 is below QMOM.minimum_size."))
    end
    return nothing
end

function _initial_moment_population(problem::CrystallisationProblem, initial_crystals)
    model = _initial_crystal_distribution_model(initial_crystals)
    target_mu3 = initial_crystals.mass_concentration / (problem.crystal_density * problem.volume_shape_factor)
    population_count = moment_count(problem.solver)
    number_scale = target_mu3 / model.raw_moment(3)
    return [number_scale * model.raw_moment(moment_order_value)
            for moment_order_value in 0:(population_count - 1)]
end

function _initial_mesh_population(problem::CrystallisationProblem, initial_crystals)
    model = _initial_crystal_distribution_model(initial_crystals)
    solver = problem.solver
    mesh = solver.cell_centre
    target_mu3 = initial_crystals.mass_concentration / (problem.crystal_density * problem.volume_shape_factor)
    number_scale = target_mu3 / model.raw_moment(3)
    numberdensity = [number_scale * model.density(length_value) for length_value in mesh]

    discrete_mu3 = solver.cell_dL * sum(numberdensity[index] * mesh[index]^3
                                        for index in eachindex(mesh))
    discrete_mu3 > 0.0 && isfinite(discrete_mu3) ||
        throw(ArgumentError("initial crystal profile has no positive mass on the solver mesh."))
    numberdensity .*= target_mu3 / discrete_mu3
    return numberdensity
end

"""
    initial_state_from_characteristics(problem, initial_crystals) -> Vector

Construct the complete solver state from an initial crystal population
specification. `mass_concentration` is crystal solid mass per batch volume in
kg/m³, and `d43` is in metres. Use `LogNormalInitialCrystals` with a
dimensionless `geometric_std`, or `GaussianInitialCrystals` with a metre-valued
`standard_deviation`.

The returned vector contains population variables followed by the values from
`problem.initial_solvent_state`. For discretised solvers, the profile is
sampled on the configured mesh, so its d43 is subject to mesh and domain
approximation. A zero d43 represents an empty population and must be paired
with zero mass concentration.
"""
function initial_state_from_characteristics(problem::CrystallisationProblem,
                                             initial_crystals::AbstractInitialCrystals)
    characteristics = _validate_initial_crystals(initial_crystals)
    solvent_values = Float64.(collect(values(problem.initial_solvent_state)))
    population_count = _population_state_count(problem.solver)

    if characteristics.d43 == 0.0
        return vcat(zeros(Float64, population_count), solvent_values)
    end

    supports_d43 = problem.solver isa AbstractDiscretisedSolver ||
                   (problem.solver isa AbstractMomentSolver &&
                    moment_order(problem.solver) >= 4)
    supports_d43 ||
        throw(ArgumentError("The initial d43 characteristic requires a moment solver " *
                            "that tracks through the fourth raw moment."))
    _validate_initial_crystal_domain(problem, characteristics.d43)
    population = problem.solver isa AbstractMomentSolver ?
                 _initial_moment_population(problem, characteristics) :
                 _initial_mesh_population(problem, characteristics)
    return vcat(population, solvent_values)
end

function _problem_with_initial_state(problem::CrystallisationProblem, state)
    fields = NamedTuple{propertynames(problem)}(
        Tuple(getfield(problem, field) for field in propertynames(problem)))
    return CrystallisationProblem(; merge(fields, (; initial_state = state))...)
end
