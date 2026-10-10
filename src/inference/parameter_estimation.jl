"""
Prepared loss setups and objective evaluation shared by `fit`, `predict` and
the Bayesian/ABC workflows.
"""

"""
    PreparedExperiment

A per-experiment simulation bundle: the experiment's `CrystallisationProblem`
(conditions applied: temperature, initial concentration, initial crystals), its
`ODEProblem` template, the solver algorithm, and the measurement time grid.
The ODEProblem is built once and reused across parameter vectors with
`remake` (SciML idiom) — the parameter-estimation hot loop never reconstructs
problems.
"""
struct PreparedExperiment
    problem::CrystallisationProblem
    odeproblem::ODEProblem
    algorithm::Any
    saveat::Vector{Float64}
    solve_options::NamedTuple
    callback_factory::Any
    mode::Symbol
    steady_options::NamedTuple
end

# A numerical return-code failure must remain a failed candidate even if its
# truncated population cannot be reconstructed into physical size metrics.
struct FailedPreparedSimulation{TS, TR} <: AbstractSolution
    time::Vector{Float64}
    concentration::Vector{Float64}
    d32::Vector{Float64}
    d43::Vector{Float64}
    d50q::Vector{Float64}
    ode_stats::TS
    reason::TR
    success::Bool
end
state_vars(failed_solution::FailedPreparedSimulation) = (; concentration = failed_solution.concentration)
size_metrics(failed_solution::FailedPreparedSimulation) =
    (; d32 = failed_solution.d32, d43 = failed_solution.d43, d50q = failed_solution.d50q)

function _failed_prepared_simulation(preparation, ode_solution, reason)
    missing_values = fill(NaN, length(preparation.saveat))
    return FailedPreparedSimulation(copy(preparation.saveat), missing_values,
        copy(missing_values), copy(missing_values), copy(missing_values), ode_solution.stats, reason, false)
end

PreparedExperiment(problem, odeproblem, algorithm, saveat) =
    PreparedExperiment(problem, odeproblem, algorithm, saveat, (;), nothing, :transient, (;))
PreparedExperiment(problem, odeproblem, algorithm, saveat, solve_options, callback_factory) =
    PreparedExperiment(problem, odeproblem, algorithm, saveat, solve_options, callback_factory, :transient, (;))

"""
    LossSetup

A `CrystallisationProblem` (kinetics + solver + saturation) bundled with
per-experiment `PreparedExperiment`s. Built with `prepare_loss` and evaluated
with `loss(lf, setup, params)`.
"""
struct LossSetup
    problem::CrystallisationProblem
    experiments::Vector{CrystallisationExperiment}
    prepared::Vector{PreparedExperiment}
    observable_names::Vector{Symbol}
    included_observations::Vector{Dict{Symbol,Vector{Int}}}
    heterogeneous_schema::Bool
    explicit_observable_order::Bool
    observable_projections::NamedTuple
end

"""
    prepare_loss(problem::CrystallisationProblem,
                 experiments::Vector{CrystallisationExperiment};
                 algorithm=nothing, solve_options=(;), callback_factory=nothing) -> LossSetup

Build per-experiment `ODEProblem` templates (u0 from each experiment's
initial concentration and initial crystals, tspan from its measurement grid,
constant temperature profile). Each loss evaluation then only remakes the parameter
vector — no ODEProblem construction in the optimisation loop.
Simulation options are distinct from optimizer options; `callback_factory(odeproblem)`
builds a fresh callback for each evaluation and composes with package callbacks.
"""
function prepare_loss(problem::CrystallisationProblem,
                      experiments::Vector{<:AbstractExperiment};
                      observable_order = nothing,
                      exclude_initial_concentration::Bool = true,
                      algorithm = nothing, solve_options::NamedTuple = (;),
                      callback_factory = nothing, observable_projections::NamedTuple = (;),
                      mode::Symbol = :transient, steady_options::NamedTuple = (;))
    mode in (:transient, :steady) || throw(ArgumentError("mode must be :transient or :steady."))
    if mode === :steady
        steady_problems = [_experiment_problem(problem, expt;
            use_measurement_concentration = false) for expt in experiments]
        return prepare_loss(steady_problems, experiments;
            observable_order, exclude_initial_concentration = false, algorithm,
            solve_options, callback_factory, observable_projections, mode, steady_options)
    end
    isempty(experiments) && throw(ArgumentError("At least one experiment is required."))
    _validate_crystallisation_solve_options(solve_options)
    prepared = map(experiments) do expt
        saveat = _loss_saveat(expt, problem.solver)
        concentration_start = Float64(first(expt.observables.concentration.time))
        concentration_start == first(saveat) || throw(ArgumentError(
            "Template-based transient preparation uses the first concentration measurement as C0, " *
            "so it must be at the simulation start. For later concentration measurements, " *
            "configure initial concentration independently and call " *
            "prepare_loss(configured_problems, experiments)."))
        per_exp_problem = _experiment_problem(problem, expt)
        odeprob, default_algorithm = crystallisation_odeproblem(per_exp_problem,
                                                        saveat)
        PreparedExperiment(per_exp_problem, odeprob,
                           algorithm === nothing ? default_algorithm : algorithm,
                           saveat, solve_options, callback_factory)
    end
    return _build_loss_setup(problem, experiments, prepared;
                             observable_order, exclude_initial_concentration, observable_projections)
end

const _BUILTIN_LOSS_PROJECTION_NAMES = (
    :concentration, :d10, :d32, :d43, :d10q, :d50q, :d90q, :moment2,
    :numberdensity, :voldensity, :moments, :quadrature_nodes,
    :quadrature_weights, :weights, :nodes, :volume,
    :size_boundary_lower_solid_mass_flow, :size_boundary_upper_solid_mass_flow)

function _validate_loss_projection_names(problem, prepared, observable_projections)
    isempty(observable_projections) && return nothing
    builtin_names = Set(_BUILTIN_LOSS_PROJECTION_NAMES)
    union!(builtin_names, propertynames(problem.initial_solvent_state))
    for prepared_experiment in prepared
        union!(builtin_names,
               propertynames(prepared_experiment.problem.initial_solvent_state))
    end
    collisions = sort!(collect(intersect(Set(propertynames(observable_projections)),
                                         builtin_names)))
    isempty(collisions) && return nothing
    collision_names = join((":$name" for name in collisions), ", ")
    throw(ArgumentError("Local projections must not replace built-in or configured " *
                        "solvent observables: $collision_names."))
end

function _build_loss_setup(problem, experiments, prepared;
                           observable_order = nothing,
                           exclude_initial_concentration::Bool = true,
                           observable_projections::NamedTuple = (;))
    isempty(experiments) && throw(ArgumentError("At least one experiment is required."))
    schemas = [collect(propertynames(expt.observables)) for expt in experiments]
    heterogeneous = any(Set(schema) != Set(first(schemas)) for schema in schemas)
    all_names = union(schemas...)
    names = isnothing(observable_order) ?
        (heterogeneous ? sort(all_names) : first(schemas)) : collect(Symbol, observable_order)
    length(unique(names)) == length(names) && Set(names) == Set(all_names) ||
        throw(ArgumentError("observable_order must list every measured observable exactly once."))
    included = [Dict(name => _included_observation_indices(expt, name, prep.saveat[1];
                        exclude_initial_concentration) for name in propertynames(expt.observables))
                for (prep, expt) in zip(prepared, experiments)]
    _validate_loss_projection_names(problem, prepared, observable_projections)
    return LossSetup(problem, experiments, prepared, names, included, heterogeneous,
                     !isnothing(observable_order), observable_projections)
end

"""
    prepare_loss(configured_problems, experiments; observable_order=nothing)

Prepare one explicitly configured problem per experiment. Initial conditions,
seeds, temperature and physical properties come from those problems; observations
supply targets and times. All concentration points are scored by default.
The problems must share the estimated kinetic parameter names and ordering.
"""
function prepare_loss(configured_problems::AbstractVector{<:CrystallisationProblem},
                      experiments::Vector{<:AbstractExperiment};
                      observable_order = nothing,
                      exclude_initial_concentration::Bool = false,
                      algorithm = nothing, solve_options::NamedTuple = (;),
                      callback_factory = nothing, initial_time::Real = 0.0,
                      observable_projections::NamedTuple = (;), mode::Symbol = :transient,
                      steady_options::NamedTuple = (;))
    mode in (:transient, :steady) || throw(ArgumentError("mode must be :transient or :steady."))
    mode === :steady && exclude_initial_concentration &&
        throw(ArgumentError("Steady concentration measurements cannot supply initial conditions."))
    _validate_crystallisation_solve_options(solve_options)
    length(configured_problems) == length(experiments) && !isempty(experiments) ||
        throw(ArgumentError("Supply one configured problem per experiment."))
    parameter_axis = paramaxis(first(configured_problems))
    all(paramaxis(configured) == parameter_axis for configured in configured_problems) ||
        throw(ArgumentError("Configured problems must share the same kinetic parameter axis."))
    prepared = map(zip(configured_problems, experiments)) do (configured, expt)
        observation_times = _loss_saveat(expt, configured.solver; require_multiple = false)
        saveat = if mode === :steady
            observation_times
        else
            isfinite(initial_time) && 0 <= initial_time <= first(observation_times) ||
                throw(ArgumentError("initial_time must be finite, nonnegative and no later than the observations."))
            sort!(unique!(vcat(Float64(initial_time), observation_times)))
        end
        _validate_crystallisation_problem(configured)
        _validate_save_times(saveat)
        integration_times = if mode === :steady
            declared_autonomous = _steady_option(steady_options, :autonomous, false)
            declared_autonomous isa Bool ||
                throw(ArgumentError("steady_options.autonomous must be Bool."))
            _validate_steadystate_problem(configured; autonomous = declared_autonomous)
            requested_horizon = _steady_option(steady_options, :relaxation_horizon, nothing)
            relaxation_horizon = isnothing(requested_horizon) ?
                _default_steady_relaxation_horizon(configured) : requested_horizon
            [0.0, Float64(relaxation_horizon)]
        else
            saveat
        end
        _validate_save_times(integration_times)
        odeproblem, default_algorithm = crystallisation_odeproblem(configured, integration_times)
        PreparedExperiment(configured, odeproblem,
            algorithm === nothing ? default_algorithm : algorithm, saveat,
            solve_options, callback_factory, mode, steady_options)
    end
    return _build_loss_setup(first(configured_problems), experiments, prepared;
                             observable_order, exclude_initial_concentration, observable_projections)
end



# Preserve caller configuration while isolating mutable status, parameters and RNG.
function _configured_mh_algorithm(supplied_algorithm;
                                  nparticles = nothing, generations = nothing,
                                  parallel_evaluation = nothing, verbosity = nothing,
                                  HPC = false)
    run_algorithm = if supplied_algorithm === nothing
        Metaheuristics.DE(; N = 128, strategy = :best1,
            options = Metaheuristics.Options(; iterations = 128,
                parallel_evaluation = true,
                f_calls_limit = CRISTOOL_MAX_OPTIMISER_CALLS))
    else
        deepcopy(supplied_algorithm)
    end
    run_algorithm.status = Metaheuristics.State(nothing, [])
    if nparticles !== nothing
        nparticles > 0 || throw(ArgumentError("nparticles must be positive"))
        run_algorithm.parameters.N = nparticles
    end
    if generations !== nothing
        generations > 0 || throw(ArgumentError("generations must be positive"))
        run_algorithm.options.iterations = generations
    end
    if parallel_evaluation !== nothing
        run_algorithm.options.parallel_evaluation = parallel_evaluation
    end
    if verbosity !== nothing
        run_algorithm.options.verbose = verbosity > 1
    end
    HPC && (run_algorithm.options.verbose = false)
    return run_algorithm
end

"""
    _experiment_problem(problem, expt) -> CrystallisationProblem

The problem with the experiment's conditions applied (temperature profile,
initial concentration, and initial crystal characteristics). Kinetics,
saturation model, physical constants and explicit initial population carry over
from the base problem. Explicit experiment crystal characteristics override that
population. The base problem is never mutated.
"""
function _experiment_problem(problem::CrystallisationProblem,
                             expt::CrystallisationExperiment;
                             use_measurement_concentration::Bool = true)
    template_solvent = isnothing(problem.initial_state) ?
                       problem.initial_solvent_state : solvent_state(problem, problem.initial_state)
    applied_concentration = use_measurement_concentration ?
        first(expt.observables.concentration.mean) : initial_concentration(problem)
    experiment_solvent = merge(template_solvent, (; concentration = applied_concentration))
    experiment_state = if isnothing(problem.initial_state)
        nothing
    else
        # Preserve public population coordinates, including DQMOM weights/nodes.
        # Solvent conditions belong to the experiment; never mutate its template.
        vcat(problem.initial_state[_population_state_range(problem)],
             problem.initial_state[_reactor_state_range(problem)],
             collect(values(experiment_solvent)))
    end
    experiment_problem = _copy_crystallisation_problem(problem;
        temp_profile = expt.conditions.temperature,
        initial_concentration = applied_concentration,
        initial_solvent_state = experiment_solvent,
        initial_state = experiment_state)
    experiment_crystals = _experiment_conditions(expt).crystals
    isnothing(experiment_crystals) && return experiment_problem
    experiment_seed_state = initial_state_from_characteristics(experiment_problem, experiment_crystals)
    if !isnothing(experiment_state)
        # A new seed changes the population, not an explicit reactor volume.
        reactor_indices = _reactor_state_range(experiment_problem)
        experiment_seed_state[reactor_indices] .= experiment_state[reactor_indices]
    end
    return _problem_with_initial_state(experiment_problem, experiment_seed_state)
end

_total_nparams(nucl, gr, agg, br; diss = nodissolution()) =
    _nparams_of(nucl) + _nparams_of(gr) + _nparams_of(diss) +
    _nparams_of(agg) + _nparams_of(br)

_nparams_of(kinetic_function) = hasproperty(kinetic_function, :nparams) ?
    getproperty(kinetic_function, :nparams) : 0

"""
    _params_to_p(prob, params) -> NamedTuple

Split a flat parameter vector into the named-tuple parameter container used by
the ODE right-hand sides. The compatibility order is
`[p_nucleation; p_growth; p_aggregation; p_breakage]` when the dissolution
block is empty; the canonical order with independent dissolution is
`[p_nucleation; p_growth; p_dissolution; p_aggregation; p_breakage]`. The field types match
the template's `p` exactly for the same parameter element type, so `remake`
keeps the problem type stable across evaluations.
"""
function _params_to_p(prob::CrystallisationProblem, params)
    params isa AbstractVector || throw(ArgumentError("Kinetic parameters must be a vector."))
    expected_parameters = _total_nparams(prob.kinetics_nucleationfunction,
        prob.kinetics_growthfunction, prob.kinetics_aggregationfunction,
        prob.kinetics_breakagefunction; diss = prob.kinetics_dissolutionfunction)
    length(params) == expected_parameters ||
        throw(ArgumentError("Expected $expected_parameters kinetic parameters, received $(length(params))."))
    # ComponentArray with the composite kinetic axis: `p.nucl`/`p.gr` are
    # views, so the rate functions' `_named_params` short-circuits without
    # rebuilding the parameter container on every ODE step.
    return ComponentArray(params, paramaxis(prob))
end

"""
    _solve_prepared(prep::PreparedExperiment, params) -> AbstractSolution

Solve a prepared experiment for a parameter vector: `remake` the template
ODEProblem with the named parameter container, then solve with the same
options as `_simulatecrystallisation`.
"""
function _solve_prepared(prep::PreparedExperiment, params)
    remade = remake(prep.odeproblem; p = _params_to_p(prep.problem, params))
    if prep.mode === :steady
        return _solve_steadystate_ode(prep.problem, remade, prep.algorithm;
            steady_options = prep.steady_options, solve_options = prep.solve_options,
            callback_factory = prep.callback_factory,
            reconstruction_failure_handler = (raw_solution, reconstruction_error) ->
                _failed_prepared_simulation(prep, raw_solution, reconstruction_error))
    end
    sol = _solve_crystallisation_ode(prep.problem, remade, prep.algorithm, prep.saveat;
                                     solve_options = prep.solve_options,
                                     callback_factory = prep.callback_factory)
    OrdinaryDiffEq.SciMLBase.successful_retcode(sol.retcode) ||
        return _failed_prepared_simulation(prep, sol, sol.retcode)
    physical_result, _ = _wrap_prepared_result(prep.problem, sol;
        reconstruction_failure_handler = (raw_solution, reconstruction_error) ->
            _failed_prepared_simulation(prep, raw_solution, reconstruction_error))
    return physical_result
end

"""
    _solve_experiment(problem, params, expt) -> AbstractSolution

Convenience wrapper for one-off evaluations (tests, scripting): build a
single-experiment `LossSetup` and solve. Use `prepare_loss` + `loss(lf,
setup, params)` for optimisation loops.
"""
function _solve_experiment(problem::CrystallisationProblem, params,
                           expt::CrystallisationExperiment)
    return _solve_prepared(prepare_loss(problem, [expt]).prepared[1], params)
end

"""
    _loss_observable_names(experiment, solver) -> Vector{Symbol}

Return every measured observable in the order declared by the experiment's
typed `NamedTuple`. Observable selection is data-driven: if two metrics such
as `d43` and `d50q` are supplied, both are objective terms.
"""
_loss_observable_names(expt::CrystallisationExperiment, ::AbstractSolver) =
    collect(propertynames(expt.observables))

_loss_observable_names(expt::CrystallisationExperiment, solution::AbstractSolution) =
    _loss_observable_names(expt, MoM())

function _loss_saveat(expt::CrystallisationExperiment, solver::AbstractSolver;
                     require_multiple::Bool = true)
    times = Float64[]
    for name in _loss_observable_names(expt, solver)
        measured = getproperty(expt.observables, name)
        append!(times, Float64.(measured.time))
    end
    saveat = sort!(unique!(times))
    (!require_multiple || length(saveat) >= 2) ||
        throw(ArgumentError("An experiment needs at least two distinct observation times."))
    return saveat
end

function _measurement_data(observable::Observable)
    return observable.time, observable.mean
end

function _variance_at(observable::Observable, index::Int)
    return observable.variance === nothing ? nothing : observable.variance[index]
end

function _scaled_variance_floor(mean_value, relative_variance_floor)
    value_scale = max(abs(mean_value), eps(float(one(mean_value))))
    return relative_variance_floor * value_scale^2
end

function _measurement_variance(observable, mean_value, index,
                               ::MeasuredVariance, relative_variance_floor)
    measured_variance = _variance_at(observable, index)
    measured_variance isa Real && isfinite(measured_variance) && measured_variance > 0 ||
        throw(ArgumentError("Gaussian likelihood requires supplied strictly positive measurement variance."))
    return measured_variance
end

function _measurement_variance(observable, mean_value, index,
                               model::RelativeVariance, relative_variance_floor)
    variance_floor = _scaled_variance_floor(mean_value, relative_variance_floor)
    return max(variance_floor, (0.01 * model.percent * abs(mean_value))^2)
end

function _linear_interpolate(xs::AbstractVector, ys::AbstractVector, x::Real)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    right = searchsortedfirst(xs, x)
    right == 1 && return ys[1]
    xs[right] == x && return ys[right]
    left = right - 1
    fraction = (x - xs[left]) / (xs[right] - xs[left])
    return ys[left] + fraction * (ys[right] - ys[left])
end

function _simulated_at(solution::AbstractSolution, name::Symbol,
                       target_time::AbstractVector)
    simulated = observable_values(solution, name)
    simulated isa AbstractVector ||
        throw(ArgumentError("Simulated observable :$name must be a trajectory."))
    all((solution.time[1] .<= target_time) .&
        (target_time .<= solution.time[end])) ||
        throw(ArgumentError("Observation times for :$name lie outside the simulated time span."))
    return [_linear_interpolate(solution.time, simulated, t) for t in target_time]
end

function _setup_observable_values(setup::LossSetup, solution::AbstractSolution, name::Symbol)
    hasproperty(setup.observable_projections, name) || return observable_values(solution, name)
    projected_values = getproperty(setup.observable_projections, name)(solution)
    projected_values isa AbstractVector && length(projected_values) == length(solution.time) ||
        throw(ArgumentError("Projection :$name must return a trajectory matching the solution times."))
    all(projected_value isa Real && isfinite(projected_value) for projected_value in projected_values) ||
        throw(ArgumentError("Projection :$name must contain finite real values."))
    return projected_values
end

function _simulated_at(setup::LossSetup, solution::AbstractSolution, name::Symbol,
                       target_time::AbstractVector)
    projected_values = _setup_observable_values(setup, solution, name)
    projected_values isa AbstractVector ||
        throw(ArgumentError("Simulated observable :$name must be a trajectory."))
    all(first(solution.time) <= measured_time <= last(solution.time) for measured_time in target_time) ||
        throw(ArgumentError("Observation times for :$name lie outside the simulated time span."))
    return [_linear_interpolate(solution.time, projected_values, measured_time) for measured_time in target_time]
end

function _included_observation_indices(expt::CrystallisationExperiment, name::Symbol,
                                       start_time; exclude_initial_concentration::Bool = true)
    measured = getproperty(expt.observables, name)
    if name === :concentration && exclude_initial_concentration
        measured.time[1] == start_time || throw(ArgumentError(
            "Concentration observations must begin at the simulation start when used as C0. " *
            "Supply independently configured initial conditions otherwise."))
        return collect(2:length(measured.time))
    end
    return collect(eachindex(measured.time))
end

_included_observation_indices(setup::LossSetup, experiment_index::Int, name::Symbol) =
    setup.included_observations[experiment_index][name]

function _observable_weight(lossfunction::AbstractPELossFunction, index::Int,
                            name::Symbol)
    hasproperty(lossfunction, :weighting) || return 1.0
    weights = lossfunction.weighting
    if weights isa NamedTuple
        return hasproperty(weights, name) ? getproperty(weights, name) : 1.0
    elseif weights isa AbstractDict
        return get(weights, name, 1.0)
    end
    return index <= length(weights) ? weights[index] : 1.0
end

function _validate_loss_weights(lossfunction, setup)
    hasproperty(lossfunction, :weighting) || return nothing
    weights = lossfunction.weighting
    if weights isa NamedTuple || weights isa AbstractDict
        extra_names = setdiff(collect(keys(weights)), setup.observable_names)
        isempty(extra_names) || throw(ArgumentError("Weights refer to unmeasured observables: $extra_names"))
    elseif setup.heterogeneous_schema && !setup.explicit_observable_order
        effective_weights = [_observable_weight(lossfunction, objective_index, name)
                             for (objective_index, name) in enumerate(setup.observable_names)]
        if !isempty(effective_weights) && any(weight != first(effective_weights) for weight in effective_weights)
            throw(ArgumentError("Nonuniform vector weights with different observable subsets require observable_order; use named weights instead."))
        end
    end
    all(weight -> weight isa Real && isfinite(weight) && weight >= 0, values(weights)) ||
        throw(ArgumentError("Observable weights must be finite nonnegative numbers."))
    return nothing
end

"""
    _experiment_objectives(lf, expt, solution) -> Vector

Return one weighted objective contribution per measured observable. The
observable order is the `NamedTuple` field order and every time point in each
observable contributes to its objective.
"""
function _experiment_objectives(lf::AbstractPELossFunction,
                                expt::CrystallisationExperiment, solution)
    names = _loss_observable_names(expt, solution)
    contributions = map(enumerate(names)) do (observable_index, name)
        measured = getproperty(expt.observables, name)
        measured_time, measured_mean = _measurement_data(measured)
        simulated_mean = _simulated_at(solution, name, measured_time)
        first_index = name === :concentration && measured_time[1] == solution.time[1] ? 2 : 1

        if lf isa logMLE
            objective = 0.0
            for index in first_index:length(measured_mean)
                variance = _measurement_variance(measured, measured_mean[index], index,
                                                 lf.variance_model,
                                                 lf.relative_variance_floor)
                residual = simulated_mean[index] - measured_mean[index]
                objective += log(2π * variance) + residual^2 / variance
            end
            return _observable_weight(lf, observable_index, name) * 0.5 * objective
        end

        if !solution.success
            return _observable_weight(lf, observable_index, name) * CRISTOOL_FAILED_SIMULATION_PENALTY
        end
        objective = first_index > length(measured_mean) ? zero(eltype(simulated_mean)) :
                    mean(abs.(simulated_mean[first_index:end] .-
                              measured_mean[first_index:end]))
        return _observable_weight(lf, observable_index, name) * objective
    end
    return contributions
end

"""
    loss(lf::logMLE, setup::LossSetup, params) -> Real

Log Maximum Likelihood Estimation loss over all prepared experiments. The
initial concentration point is excluded because it defines the experiment's
initial condition; all points of every other observable contribute.
"""
function _loss_objectives(lf::AbstractPELossFunction, setup::LossSetup, params)
    _validate_loss_weights(lf, setup)
    # ODE parameter containers differ by backend; discretised solvers keep
    # empty kinetic blocks in a fixed-width NamedTuple. Count the kinetic axis,
    # not the number of fields in the prepared ODE parameter container.
    expected_count = _setup_parameter_count(setup)
    length(params) == expected_count || throw(DimensionMismatch(
        "Expected $expected_count kinetic parameters, received $(length(params))."))
    objective_sums = [zero(eltype(params)) for _ in setup.observable_names]
    observation_counts = zeros(Int, length(setup.observable_names))
    for (experiment_index, (prep, expt)) in enumerate(zip(setup.prepared, setup.experiments))
        solution = _solve_prepared(prep, params)
        # Numerical failures have a definite penalty. Exceptions from data,
        # kinetics and custom observables propagate, including DomainError.
        solution.success || return [zero(eltype(params)) +
            _observable_weight(lf, objective_index, name) * CRISTOOL_FAILED_SIMULATION_PENALTY
            for (objective_index, name) in enumerate(setup.observable_names)]
        for (objective_index, name) in enumerate(setup.observable_names)
            hasproperty(expt.observables, name) || continue
            iszero(_observable_weight(lf, objective_index, name)) && continue
            measured = getproperty(expt.observables, name)
            included_indices = _included_observation_indices(setup, experiment_index, name)
            isempty(included_indices) && continue
            predicted = prep.mode === :steady ?
                fill(only(_setup_observable_values(setup, solution, name)), length(measured.time)) :
                _simulated_at(setup, solution, name, measured.time)
            for measurement_index in included_indices
                residual = predicted[measurement_index] - measured.mean[measurement_index]
                if lf isa logMLE
                    variance = _measurement_variance(measured, measured.mean[measurement_index],
                        measurement_index, lf.variance_model, lf.relative_variance_floor)
                    objective_sums[objective_index] += 0.5 * (log(2π * variance) + residual^2 / variance)
                else
                    objective_sums[objective_index] += abs(residual)
                end
                observation_counts[objective_index] += 1
            end
        end
    end
    return [ _observable_weight(lf, objective_index, name) *
             (lf isa mae && observation_counts[objective_index] > 0 ?
                objective_sums[objective_index] / observation_counts[objective_index] :
                objective_sums[objective_index])
             for (objective_index, name) in enumerate(setup.observable_names)]
end

"""
    loss(lf, setup::LossSetup, params)

Sum named observable objectives across experiments. MAE pools each observable's
included points before averaging; likelihood sums their NLL contributions.
Unsuccessful numerical solves receive a penalty. Invalid inputs and user-code
exceptions propagate to the caller.
"""
loss(lf::Union{mae,logMLE}, setup::LossSetup, params) = sum(_loss_objectives(lf, setup, params))

"""
    loss(lf::logMLE, problem::CrystallisationProblem, params,
         experiments::Vector{CrystallisationExperiment}) -> Real

Convenience form: builds a `LossSetup` via `prepare_loss` and evaluates.
For optimisation loops, build the setup once and call
`loss(lf, setup, params)`.
"""
loss(lf::AbstractPELossFunction, problem::CrystallisationProblem, params,
     experiments::Vector{<:AbstractExperiment}) =
    loss(lf, prepare_loss(problem, experiments), params)
