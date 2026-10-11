"""
Prepared solve templates and objective evaluation shared by `fit`, `predict` and
the Bayesian/ABC workflows.
"""

"""
    AbstractSimulationMode

How a prepared run is solved: `TransientMode()` integrates the trajectory over
the observation times; `SteadyMode(options)` relaxes an autonomous MSMPR to its
equilibrium and compares the steady values with every observation.
"""
abstract type AbstractSimulationMode end

"""
    TransientMode()

Integrate each run from time zero and compare the trajectory with observations
at their measured times. This is the default for `prepare_fit` and `predict`.
"""
struct TransientMode <: AbstractSimulationMode end

"""
    SteadyMode(options::SteadyStateOptions)
    SteadyMode(; options...)

Relax each run to its MSMPR steady state with `options` and compare the steady
values with every observation, regardless of observation time. Run conditions
must be numeric. Callable kinetic laws need `autonomous = true`.
"""
struct SteadyMode <: AbstractSimulationMode
    options::SteadyStateOptions
end
SteadyMode(; steady_settings...) = SteadyMode(SteadyStateOptions(; steady_settings...))

"""
    PreparedRun

One configured `CrystallisationProblem` with its `ODEProblem` template, resolved
ODE algorithm and save times. The template is built once and reused across
parameter vectors with `remake`, so the estimation loop never reconstructs it.
"""
struct PreparedRun{Problem <: CrystallisationProblem, Template, Algorithm}
    problem::Problem
    odeproblem::Template
    algorithm::Algorithm
    saveat::Vector{Float64}
end

"""
    PreparedSolves

Prepared runs that share one simulation mode and one set of numerical solve
settings. `requested_algorithm` is the caller's ODE algorithm (or `nothing`),
kept so predictions reuse the same request.
"""
struct PreparedSolves{Runs <: AbstractVector{<:PreparedRun}, Mode <: AbstractSimulationMode,
                      Algorithm, SolveOptions <: NamedTuple, CallbackFactory}
    runs::Runs
    mode::Mode
    requested_algorithm::Algorithm
    solve_options::SolveOptions
    callback_factory::CallbackFactory
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

function _failed_prepared_simulation(prepared_run::PreparedRun, ode_solution, reason)
    missing_values = fill(NaN, length(prepared_run.saveat))
    return FailedPreparedSimulation(copy(prepared_run.saveat), missing_values,
        copy(missing_values), copy(missing_values), copy(missing_values), ode_solution.stats, reason, false)
end

"""
    LossSetup

Prepared solves bundled with the measured experiments they are scored against:
the observable order, the scored sample indices per experiment, the observable
weights resolved from the preparation loss, and local observable projections.
Built with `prepare_loss` (normally through `prepare_fit`) and evaluated with
`loss(lf, setup, params)`.
"""
struct LossSetup{Solves <: PreparedSolves, Weights <: NamedTuple, Projections <: NamedTuple}
    solves::Solves
    experiments::Vector{CrystallisationExperiment}
    observable_names::Vector{Symbol}
    included_observations::Vector{Dict{Symbol, Vector{Int}}}
    observable_weights::Weights
    observable_projections::Projections
end

# Transient runs always start at time zero; steady runs need no start.
function _run_save_times(::TransientMode, observation_times::AbstractVector)
    first(observation_times) >= 0 ||
        throw(ArgumentError("Observation times must be nonnegative; runs start at time zero."))
    return sort!(unique!(vcat(0.0, observation_times)))
end
_run_save_times(::SteadyMode, observation_times::AbstractVector) = collect(Float64, observation_times)

_integration_times(::TransientMode, configured_problem, save_times) = save_times
function _integration_times(steady_mode::SteadyMode, configured_problem, save_times)
    _validate_steadystate_problem(configured_problem; autonomous = steady_mode.options.autonomous)
    relaxation_horizon = something(steady_mode.options.relaxation_horizon,
        _default_steady_relaxation_horizon(configured_problem))
    return [0.0, Float64(relaxation_horizon)]
end

"""
    _prepare_solves(configured_problems, observation_times, mode;
                    algorithm=nothing, solve_options=(;), callback_factory=nothing) -> PreparedSolves

Build one `ODEProblem` template per configured problem. `observation_times[i]`
are the times at which run `i` is compared or reported. The problems must share
the kinetic parameter axis. `callback_factory(odeproblem)` builds a fresh
callback for each evaluation and composes with package callbacks.
"""
function _prepare_solves(configured_problems::AbstractVector{<:CrystallisationProblem},
                         observation_times::AbstractVector, simulation_mode::AbstractSimulationMode;
                         algorithm = nothing, solve_options::NamedTuple = (;), callback_factory = nothing)
    _validate_crystallisation_solve_options(solve_options)
    isempty(configured_problems) && throw(ArgumentError("At least one run is required."))
    length(configured_problems) == length(observation_times) ||
        throw(ArgumentError("Supply one set of observation times per run."))
    parameter_axis = paramaxis(first(configured_problems))
    all(paramaxis(configured) == parameter_axis for configured in configured_problems) ||
        throw(ArgumentError("Configured problems must share the same kinetic parameter axis."))
    prepared_runs = map(configured_problems, observation_times) do configured, run_times
        _validate_crystallisation_problem(configured)
        save_times = _run_save_times(simulation_mode, run_times)
        _validate_save_times(save_times)
        integration_times = _integration_times(simulation_mode, configured, save_times)
        _validate_save_times(integration_times)
        odeproblem, default_algorithm = crystallisation_odeproblem(configured, integration_times)
        PreparedRun(configured, odeproblem, something(algorithm, default_algorithm), save_times)
    end
    return PreparedSolves(prepared_runs, simulation_mode, algorithm, solve_options, callback_factory)
end

"""
    prepare_loss(configured_problems, experiments; mode=TransientMode(), loss_function=nothing,
                 observable_order=nothing, observable_projections=(;), solve settings...) -> LossSetup

Prepare one explicitly configured problem per experiment. Initial conditions,
seeds, temperature and physical properties come from those problems; observations
supply targets and times. Time-zero observations that a run reads through
`FromObservation` conditions are its initial values and are not scored.
`loss_function` supplies the observable weights; without it every weight is 1.
"""
function prepare_loss(configured_problems::AbstractVector{<:CrystallisationProblem},
                      experiments::AbstractVector{<:CrystallisationExperiment};
                      mode::AbstractSimulationMode = TransientMode(), loss_function = nothing,
                      observable_order = nothing, observable_projections::NamedTuple = (;),
                      algorithm = nothing, solve_options::NamedTuple = (;), callback_factory = nothing)
    length(configured_problems) == length(experiments) && !isempty(experiments) ||
        throw(ArgumentError("Supply one configured problem per experiment."))
    prepared_solves = _prepare_solves(configured_problems, map(_observation_times, experiments), mode;
        algorithm, solve_options, callback_factory)
    _validate_loss_projection_names(prepared_solves, observable_projections)
    schemas = [collect(propertynames(expt.observables)) for expt in experiments]
    heterogeneous_schema = any(Set(schema) != Set(first(schemas)) for schema in schemas)
    all_names = union(schemas...)
    names = isnothing(observable_order) ?
        (heterogeneous_schema ? sort(all_names) : first(schemas)) : collect(Symbol, observable_order)
    length(unique(names)) == length(names) && Set(names) == Set(all_names) ||
        throw(ArgumentError("observable_order must list every measured observable exactly once."))
    included = [Dict(name => _scored_indices(expt, name) for name in propertynames(expt.observables))
                for expt in experiments]
    observable_weights = _observable_weights(loss_function, names,
        heterogeneous_schema && isnothing(observable_order))
    return LossSetup(prepared_solves, collect(CrystallisationExperiment, experiments), names, included,
                     observable_weights, observable_projections)
end

const _BUILTIN_LOSS_PROJECTION_NAMES = (
    :concentration, :d10, :d32, :d43, :d10q, :d50q, :d90q, :moment2,
    :numberdensity, :voldensity, :moments, :quadrature_nodes,
    :quadrature_weights, :weights, :nodes, :volume,
    :size_boundary_lower_solid_mass_flow, :size_boundary_upper_solid_mass_flow)

function _validate_loss_projection_names(prepared_solves::PreparedSolves, observable_projections)
    isempty(observable_projections) && return nothing
    builtin_names = Set(_BUILTIN_LOSS_PROJECTION_NAMES)
    for prepared_run in prepared_solves.runs
        union!(builtin_names, propertynames(prepared_run.problem.initial_solvent_state))
    end
    collisions = sort!(collect(intersect(Set(propertynames(observable_projections)),
                                         builtin_names)))
    isempty(collisions) && return nothing
    collision_names = join((":$name" for name in collisions), ", ")
    throw(ArgumentError("Local projections must not replace built-in or configured " *
                        "solvent observables: $collision_names."))
end

# Time-zero samples that a run reads as initial values are its conditions, not targets.
function _scored_indices(expt::CrystallisationExperiment, name::Symbol)
    measured_times = getproperty(expt.observables, name).time
    name in observation_sources(expt.conditions) || return collect(eachindex(measured_times))
    return [sample_index for sample_index in eachindex(measured_times) if !iszero(measured_times[sample_index])]
end

"""Observable weights by name, resolved once from the preparation loss."""
_observable_weights(loss_function, names, ambiguous_order::Bool) =
    NamedTuple{Tuple(names)}(ntuple(_ -> 1.0, length(names)))
_observable_weights(loss_function::Union{GaussianNLL, MeanAbsoluteError}, names, ambiguous_order::Bool) =
    NamedTuple{Tuple(names)}(Tuple(_checked_weight.(_weights_in_order(loss_function.weighting, names, ambiguous_order))))

function _weights_in_order(weighting::Union{NamedTuple, AbstractDict}, names, ambiguous_order::Bool)
    extra_names = setdiff(collect(keys(weighting)), names)
    isempty(extra_names) || throw(ArgumentError("Weights refer to unmeasured observables: $extra_names"))
    return [get(weighting, name, 1.0) for name in names]
end

function _weights_in_order(weighting::AbstractVector, names, ambiguous_order::Bool)
    ordered_weights = [objective_index <= length(weighting) ? weighting[objective_index] : 1.0
                       for objective_index in eachindex(names)]
    ambiguous_order && !allequal(ordered_weights) && throw(ArgumentError(
        "Nonuniform vector weights with different observable subsets require observable_order; use named weights instead."))
    return ordered_weights
end

function _checked_weight(weight)
    weight isa Real && isfinite(weight) && weight >= 0 ||
        throw(ArgumentError("Observable weights must be finite nonnegative numbers."))
    return Float64(weight)
end

_kinetic_parameter_count(configured_problem::CrystallisationProblem) = _parameter_count(configured_problem)
_kinetic_parameter_count(prepared_solves::PreparedSolves) =
    _kinetic_parameter_count(first(prepared_solves.runs).problem)
_kinetic_parameter_count(setup::LossSetup) = _kinetic_parameter_count(setup.solves)

"""
    _params_to_p(prob, params) -> ComponentArray

Split a flat parameter vector into the named parameter container used by the
ODE right-hand sides. The compatibility order is
`[p_nucleation; p_growth; p_aggregation; p_breakage]` when the dissolution
block is empty; the canonical order with independent dissolution is
`[p_nucleation; p_growth; p_dissolution; p_aggregation; p_breakage]`.
"""
function _params_to_p(prob::CrystallisationProblem, params)
    params isa AbstractVector || throw(ArgumentError("Kinetic parameters must be a vector."))
    expected_parameters = _kinetic_parameter_count(prob)
    length(params) == expected_parameters ||
        throw(ArgumentError("Expected $expected_parameters kinetic parameters, received $(length(params))."))
    # ComponentArray with the composite kinetic axis: `p.nucl`/`p.gr` are
    # views, so the rate functions' `_named_params` short-circuits without
    # rebuilding the parameter container on every ODE step.
    return ComponentArray(params isa ComponentArray ? getdata(params) : params, paramaxis(prob))
end

"""
    _solve_prepared(prepared_solves, prepared_run, params) -> AbstractSolution

Solve one prepared run for a parameter vector: `remake` its template with the
named parameter container and solve it in the prepared simulation mode.
"""
_solve_prepared(prepared_solves::PreparedSolves, prepared_run::PreparedRun, params) =
    _solve_prepared(prepared_solves.mode, prepared_run,
        remake(prepared_run.odeproblem; p = _params_to_p(prepared_run.problem, params)),
        prepared_solves.solve_options, prepared_solves.callback_factory)

function _solve_prepared(steady_mode::SteadyMode, prepared_run::PreparedRun, remade_problem,
                         solve_options, callback_factory)
    return _solve_steadystate_ode(prepared_run.problem, remade_problem, prepared_run.algorithm;
        steady_options = steady_mode.options, solve_options, callback_factory,
        reconstruction_failure_handler = (raw_solution, reconstruction_error) ->
            _failed_prepared_simulation(prepared_run, raw_solution, reconstruction_error))
end

function _solve_prepared(::TransientMode, prepared_run::PreparedRun, remade_problem,
                         solve_options, callback_factory)
    raw_solution = _solve_crystallisation_ode(prepared_run.problem, remade_problem,
        prepared_run.algorithm, prepared_run.saveat; solve_options, callback_factory)
    SciMLBase.successful_retcode(raw_solution.retcode) ||
        return _failed_prepared_simulation(prepared_run, raw_solution, raw_solution.retcode)
    physical_result, _ = _wrap_prepared_result(prepared_run.problem, raw_solution;
        reconstruction_failure_handler = (failed_solution, reconstruction_error) ->
            _failed_prepared_simulation(prepared_run, failed_solution, reconstruction_error))
    return physical_result
end

"""Sorted distinct times of every observable measured in `expt`."""
function _observation_times(expt::CrystallisationExperiment)
    times = Float64[]
    for measured in values(expt.observables)
        append!(times, measured.time)
    end
    return sort!(unique!(times))
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

# Projections always see a trajectory; a steady result exposes scalars, so they get its endpoint solution.
_projection_trajectory(solution::AbstractSolution) = solution
_projection_trajectory(steady_result::CrystallisationSteadyStateSolution) = steady_result.solution

function _projected_values(observable_projections::NamedTuple, solution::AbstractSolution, name::Symbol)
    hasproperty(observable_projections, name) || return observable_values(solution, name)
    solution = _projection_trajectory(solution)
    projected_values = getproperty(observable_projections, name)(solution)
    projected_values isa AbstractVector && length(projected_values) == length(solution.time) ||
        throw(ArgumentError("Projection :$name must return a trajectory matching the solution times."))
    all(projected_value isa Real && isfinite(projected_value) for projected_value in projected_values) ||
        throw(ArgumentError("Projection :$name must contain finite real values."))
    return projected_values
end

"""
    _predicted_series(mode, observable_projections, solution, name, target_times)

Simulated `name` at `target_times`: interpolated along a transient trajectory,
or the single steady value repeated for every target time.
"""
function _predicted_series(::TransientMode, observable_projections::NamedTuple,
                           solution::AbstractSolution, name::Symbol, target_times::AbstractVector)
    projected_values = _projected_values(observable_projections, solution, name)
    projected_values isa AbstractVector ||
        throw(ArgumentError("Simulated observable :$name must be a trajectory."))
    all(first(solution.time) <= measured_time <= last(solution.time) for measured_time in target_times) ||
        throw(ArgumentError("Observation times for :$name lie outside the simulated time span."))
    return [_linear_interpolate(solution.time, projected_values, measured_time) for measured_time in target_times]
end

_predicted_series(::SteadyMode, observable_projections::NamedTuple, solution::AbstractSolution,
                  name::Symbol, target_times::AbstractVector) =
    fill(only(_projected_values(observable_projections, solution, name)), length(target_times))

# One scored measurement's contribution, and how an observable's contributions pool.
function _residual_term(likelihood::GaussianNLL, measured::Observable, measurement_index::Int, residual)
    variance = _measurement_variance(measured, measured.mean[measurement_index], measurement_index,
        likelihood.variance_model, likelihood.relative_variance_floor)
    return 0.5 * (log(2π * variance) + residual^2 / variance)
end
_residual_term(::MeanAbsoluteError, measured::Observable, measurement_index::Int, residual) = abs(residual)

_pooled(::GaussianNLL, objective_sum, observation_count::Int) = objective_sum
_pooled(::MeanAbsoluteError, objective_sum, observation_count::Int) =
    observation_count > 0 ? objective_sum / observation_count : objective_sum

"""
    _loss_objectives(lf, setup, params) -> Vector

One weighted objective per observable in `setup.observable_names`, pooled over
every scored measurement of every experiment. A numerically failed solve gives
each observable its weighted penalty.
"""
function _loss_objectives(lossfunction::Union{GaussianNLL, MeanAbsoluteError}, setup::LossSetup, params)
    expected_count = _kinetic_parameter_count(setup)
    length(params) == expected_count || throw(DimensionMismatch(
        "Expected $expected_count kinetic parameters, received $(length(params))."))
    observable_weights = setup.observable_weights
    objective_sums = [zero(eltype(params)) for _ in setup.observable_names]
    observation_counts = zeros(Int, length(setup.observable_names))
    prepared_solves = setup.solves
    for (experiment_index, (prepared_run, expt)) in enumerate(zip(prepared_solves.runs, setup.experiments))
        solution = _solve_prepared(prepared_solves, prepared_run, params)
        # Numerical failures have a definite penalty. Exceptions from data,
        # kinetics and custom observables propagate, including DomainError.
        solution.success || return [zero(eltype(params)) + observable_weight * CRISTOOL_FAILED_SIMULATION_PENALTY
                                    for observable_weight in observable_weights]
        for (objective_index, name) in enumerate(setup.observable_names)
            hasproperty(expt.observables, name) || continue
            iszero(observable_weights[objective_index]) && continue
            included_indices = setup.included_observations[experiment_index][name]
            isempty(included_indices) && continue
            measured = getproperty(expt.observables, name)
            predicted = _predicted_series(prepared_solves.mode, setup.observable_projections,
                solution, name, measured.time)
            for measurement_index in included_indices
                residual = predicted[measurement_index] - measured.mean[measurement_index]
                objective_sums[objective_index] += _residual_term(lossfunction, measured, measurement_index, residual)
                observation_counts[objective_index] += 1
            end
        end
    end
    return [observable_weights[objective_index] *
            _pooled(lossfunction, objective_sums[objective_index], observation_counts[objective_index])
            for objective_index in eachindex(setup.observable_names)]
end

"""
    loss(lf, setup::LossSetup, params)

Sum named observable objectives across experiments. MAE pools each observable's
included points before averaging; likelihood sums their NLL contributions.
Observable weights are those resolved from the loss the setup was prepared with.
Unsuccessful numerical solves receive a penalty. Invalid inputs and user-code
exceptions propagate to the caller.
"""
loss(lossfunction::Union{MeanAbsoluteError, GaussianNLL}, setup::LossSetup, params) =
    sum(_loss_objectives(lossfunction, setup, params))

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
