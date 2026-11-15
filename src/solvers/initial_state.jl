"""
Initial crystal-population construction from measurable characteristics.

Every seed description, whether an initial reactor seed or a feed population,
is resolved once into a `DistributionInitialCrystals` (or `nothing` for an
empty population) by `_checked_seed_description`. Solver states are then built
from that single description: raw moments from its number moments, DQMOM
weights and nodes from the inverted moments, and mesh densities from exact
cell integrals.
"""

"""
    _seed_is_populated(mass_concentration, d43) -> Bool

Check the shared mass/size contract of a parametric seed. A zero `d43` is an
empty population and requires zero mass.
"""
function _seed_is_populated(mass_concentration, d43)
    isfinite(mass_concentration) && mass_concentration >= 0.0 ||
        throw(ArgumentError("initial crystal mass_concentration must be finite and nonnegative."))
    isfinite(d43) && d43 >= 0.0 ||
        throw(ArgumentError("initial crystal d43 must be finite and nonnegative."))
    if d43 == 0.0
        mass_concentration == 0.0 ||
            throw(ArgumentError("initial crystal d43 = 0 requires mass_concentration = 0."))
        return false
    end
    mass_concentration > 0.0 ||
        throw(ArgumentError("positive initial crystal d43 requires positive mass_concentration."))
    return true
end

function _validate_initial_crystals(seed_description::LogNormalInitialCrystals)
    _seed_is_populated(seed_description.mass_concentration, seed_description.d43) || return seed_description
    isfinite(seed_description.geometric_std) && seed_description.geometric_std > 1.0 ||
        throw(ArgumentError("geometric_std must be finite and greater than 1."))
    return seed_description
end

function _validate_initial_crystals(seed_description::GaussianInitialCrystals)
    _seed_is_populated(seed_description.mass_concentration, seed_description.d43) || return seed_description
    isfinite(seed_description.standard_deviation) && seed_description.standard_deviation > 0.0 ||
        throw(ArgumentError("standard_deviation must be finite and strictly positive."))
    return seed_description
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

"""
    _checked_seed_description(seed) -> Union{Nothing, DistributionInitialCrystals}

Validate a seed and resolve it into the one description every solver state is
built from. An empty parametric seed (`d43 = 0`, zero mass) becomes `nothing`.
"""
_checked_seed_description(seed_description::Nothing) = nothing
_checked_seed_description(seed_description::DistributionInitialCrystals) = seed_description
function _checked_seed_description(seed_description::LogNormalInitialCrystals)
    _validate_initial_crystals(seed_description)
    seed_description.d43 == 0 && return nothing
    log_width = log(seed_description.geometric_std)
    return DistributionInitialCrystals(LogNormal(log(seed_description.d43) - 3.5 * log_width^2, log_width);
        mass_concentration = seed_description.mass_concentration)
end
function _checked_seed_description(seed_description::GaussianInitialCrystals)
    _validate_initial_crystals(seed_description)
    seed_description.d43 == 0 && return nothing
    seed_location = _gaussian_mean_for_d43(seed_description.d43, seed_description.standard_deviation)
    return DistributionInitialCrystals(truncated(Normal(seed_location, seed_description.standard_deviation), 0, Inf);
        mass_concentration = seed_description.mass_concentration)
end

"""Solver-specific admissibility of a nonempty seed with characteristic size `d43` (m)."""
function _validate_seed_for_solver(solver::AbstractDiscretisedSolver, d43)
    solver.lmin <= d43 <= solver.lmax ||
        throw(ArgumentError("initial crystal d43 must lie within the solver size domain " *
                            "[$(solver.lmin), $(solver.lmax)] m."))
    return nothing
end
function _validate_seed_for_solver(solver::MoM, d43)
    _require_fourth_raw_moment(solver)
    return nothing
end
function _validate_seed_for_solver(solver::Union{QMOM, DQMOM}, d43)
    _require_fourth_raw_moment(solver)
    d43 >= solver.minimum_size ||
        throw(ArgumentError("initial crystal d43 is below the solver minimum_size."))
    return nothing
end
_require_fourth_raw_moment(solver::AbstractMomentSolver) =
    moment_order(solver) >= 4 ||
        throw(ArgumentError("The initial d43 characteristic requires a moment solver " *
                            "that tracks through the fourth raw moment."))

"""Raw moments `µ0..µ_{count-1}` (SI) of a seed scaled to its solid mass concentration."""
function _seed_raw_moments(problem::DiscretisedProblem, seed_description::DistributionInitialCrystals,
                           moment_total::Integer)
    target_mu3 = seed_description.mass_concentration / _solid_mass_factor(problem)
    number_scale = target_mu3 / _seed_raw_moment(seed_description, 3)
    return [number_scale * _seed_raw_moment(seed_description, moment_index)
            for moment_index in 0:(moment_total - 1)]
end

"""
    _seed_population(solver, problem, seed) -> Vector

Initial population block for `solver`: raw moments (MoM, QMOM), positive
weights and distinct physical nodes (DQMOM), or cell-averaged number density
(mesh solvers, exact cell integrals of the seed density).
"""
_seed_population(solver::Union{MoM, QMOM}, problem::DiscretisedProblem,
                 seed_description::DistributionInitialCrystals) =
    _seed_raw_moments(problem, seed_description, moment_count(solver))

function _seed_population(solver::DQMOM, problem::DiscretisedProblem,
                          seed_description::DistributionInitialCrystals)
    nquadrature = solver.nquadrature
    raw_moments = _seed_raw_moments(problem, seed_description, moment_count(solver))
    inversion_solver = QMOM(
        nquadrature = nquadrature,
        coordinate_scale = solver.coordinate_scale,
        realizability_tolerance = solver.realizability_tolerance,
        empty_population_tolerance = 0.0,
        node_coalescence_tolerance = solver.node_coalescence_tolerance,
        minimum_size = solver.minimum_size)
    rule = invert_moments(raw_moments, inversion_solver)
    rule.active_nodes == nquadrature ||
        throw(ArgumentError("DQMOM requires an initial quadrature with " *
                            "$nquadrature distinct active nodes."))
    all(weight -> weight > 0.0, rule.weights) ||
        throw(ArgumentError("DQMOM requires strictly positive initial quadrature weights."))
    return vcat(collect(rule.weights), collect(rule.nodes))
end

"""
    _feed_population(solver, problem, seed) -> Vector

Inlet population in the coordinates of the solver's transport terms: the
initial-state population block, except DQMOM, whose feed exchange is applied
to physical raw moments.
"""
_feed_population(solver::AbstractSolver, problem::DiscretisedProblem, seed_description::Nothing) =
    zeros(_population_state_count(solver))
function _feed_population(solver::AbstractSolver, problem::DiscretisedProblem,
                          seed_description::DistributionInitialCrystals)
    _validate_seed_for_solver(solver, seed_description.d43)
    return _seed_population(solver, problem, seed_description)
end
function _feed_population(solver::DQMOM, problem::DiscretisedProblem,
                          seed_description::DistributionInitialCrystals)
    _validate_seed_for_solver(solver, seed_description.d43)
    return _seed_raw_moments(problem, seed_description, moment_count(solver))
end

"""
    initial_state_from_characteristics(problem, solver, initial_crystals) -> Vector

Construct the complete `solver` state of `problem` from an initial crystal
population specification. `mass_concentration` is crystal solid mass per batch
volume in kg/m³, and `d43` is in metres. Use `LogNormalInitialCrystals` with a
dimensionless `geometric_std`, `GaussianInitialCrystals` with a metre-valued
`standard_deviation`, or a `DistributionInitialCrystals`.

The returned vector contains population variables followed by the operation
and initial solvent values. Every description is first resolved into a
`DistributionInitialCrystals`; mesh solvers receive exact cell integrals of its
density. A zero d43 represents an empty population and must be paired with
zero mass concentration.
"""
initial_state_from_characteristics(problem::CrystallisationProblem, solver::AbstractSolver,
                                   initial_crystals::AbstractInitialCrystals) =
    initial_state_from_characteristics(DiscretisedProblem(problem, solver), initial_crystals)
initial_state_from_characteristics(problem::DiscretisedProblem,
                                   initial_crystals::AbstractInitialCrystals) =
    vcat(_initial_population(problem, initial_crystals),
         _operation_initial_state(problem.operation),
         Float64.(collect(values(problem.initial_solvent_state))))

"""
    _initial_population(problem, initial_crystals) -> Vector{Float64}

Population block of the initial state for `initial_crystals`, after the
solver-specific seed check. An empty seed gives zeros.
"""
_initial_population(problem::DiscretisedProblem, initial_crystals::AbstractInitialCrystals) =
    _initial_population(problem, _checked_seed_description(initial_crystals))
_initial_population(problem::DiscretisedProblem, seed_description::Nothing) =
    zeros(Float64, _population_state_count(problem.solver))
function _initial_population(problem::DiscretisedProblem, seed_description::DistributionInitialCrystals)
    _validate_seed_for_solver(problem.solver, seed_description.d43)
    return Vector{Float64}(_seed_population(problem.solver, problem, seed_description))
end
