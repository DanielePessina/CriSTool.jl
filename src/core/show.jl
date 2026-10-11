# REPL display for user-facing types.
#
# Every type gets a compact one-line `show(io, x)` (used inside containers and
# by `repr`) and, where a summary helps, a multi-line `show(io, MIME"text/plain"(), x)`.
# Neither form ends with a newline. Sizes stored in metres are displayed in µm and
# labelled; nothing here changes stored values. Dual numbers are displayed by value.

# --- value formatting -------------------------------------------------------

# Duals (and other wrappers exposing a primal `value` field) display as their primal value.
_display_real(plain_value::Union{AbstractFloat, Integer}) = plain_value
_display_real(wrapped_value::Real) =
    hasfield(typeof(wrapped_value), :value) ? _display_real(getfield(wrapped_value, :value)) : wrapped_value

_number_text(plain_value::Integer; digits = 4) = string(plain_value)
function _number_text(wrapped_value::Real; digits = 4)
    shown_value = _display_real(wrapped_value)
    shown_value isa Integer && return string(shown_value)
    return isfinite(shown_value) ? string(round(shown_value; sigdigits = digits)) : string(shown_value)
end

_quantity_text(quantity::Real, unit::AbstractString; scale = 1, digits = 4) =
    strip(string(_number_text(_display_real(quantity) * scale; digits), " ", unit))
_quantity_text(placeholder::FromObservation, unit::AbstractString; scale = 1, digits = 4) = _placeholder_text(placeholder)
_quantity_text(other_value, unit::AbstractString; scale = 1, digits = 4) = string(other_value)

_placeholder_text(placeholder::FromObservation) = string("FromObservation(:", placeholder.observable_name, ")")

# A profile that is either a constant or a callable of time.
_profile_text(profile::Real, unit::AbstractString) = _quantity_text(profile, unit)
_profile_text(profile, unit::AbstractString) = "function of time"

const _METRES_TO_MICROMETRES = 1.0e6

_size_text(length_metres) = _quantity_text(length_metres, "µm"; scale = _METRES_TO_MICROMETRES)

# Right-padded `label: text` rows under a heading.
function _labelled_summary(heading::AbstractString, label_rows)
    label_width = maximum(length ∘ first, label_rows)
    return join(vcat([heading],
        ["  " * rpad(row_label * ":", label_width + 2) * row_text for (row_label, row_text) in label_rows]), "\n")
end

_first_last_text(series::AbstractVector, to_text) =
    isempty(series) ? "no saves" : string(to_text(first(series)), " → ", to_text(last(series)))

# --- temperature profiles ---------------------------------------------------

_kelvin_text(temperature_value) = _quantity_text(temperature_value, "K"; digits = 6)

Base.show(display_io::IO, profile::ConstantTemperature) =
    print(display_io, "ConstantTemperature(", _kelvin_text(profile.value), ")")
Base.show(display_io::IO, profile::LinearTemperature) =
    print(display_io, "LinearTemperature(T0 = ", _kelvin_text(profile.T0),
        ", slope = ", _quantity_text(profile.slope, "K/s"), ")")
Base.show(display_io::IO, profile::RampTemperature) =
    print(display_io, "RampTemperature(T0 = ", _kelvin_text(profile.T0),
        ", t0 = ", _quantity_text(profile.t0, "s"), ", t1 = ", _quantity_text(profile.t1, "s"),
        ", slope = ", _quantity_text(profile.slope, "K/s"), ")")
Base.show(display_io::IO, ::CallableTemperature) = print(display_io, "CallableTemperature(function of time)")

# --- initial crystals -------------------------------------------------------

Base.show(display_io::IO, seed::LogNormalInitialCrystals) =
    print(display_io, "LogNormalInitialCrystals(mass_concentration = ", _quantity_text(seed.mass_concentration, "kg/m³"),
        ", d43 = ", _size_text(seed.d43), ", geometric_std = ", _quantity_text(seed.geometric_std, ""), ")")
Base.show(display_io::IO, seed::GaussianInitialCrystals) =
    print(display_io, "GaussianInitialCrystals(mass_concentration = ", _quantity_text(seed.mass_concentration, "kg/m³"),
        ", d43 = ", _size_text(seed.d43), ", standard_deviation = ", _size_text(seed.standard_deviation), ")")

Base.show(display_io::IO, seed::DistributionInitialCrystals) =
    print(display_io, "DistributionInitialCrystals(", seed.distribution, ", mass_concentration = ",
        _quantity_text(seed.mass_concentration, "kg/m³"), ", weighting = ", seed.weighting,
        ", d43 = ", _size_text(seed.d43), ")")

Base.show(display_io::IO, ::MIME"text/plain", seed::AbstractInitialCrystals) = show(display_io, seed)

# --- operations and feeds ---------------------------------------------------

_crystals_text(::Nothing) = "none"
_crystals_text(seed::AbstractInitialCrystals) = sprint(show, seed)

_batch_volume_text(volume::Real) = volume == 1 ? "" : "volume = " * _quantity_text(volume, "m³")
Base.show(display_io::IO, operation::BatchOperation) =
    print(display_io, "BatchOperation(", _batch_volume_text(operation.volume), ")")

function Base.show(display_io::IO, feed::CrystallisationFeed)
    print(display_io, "CrystallisationFeed(concentration = ", _profile_text(feed.concentration, "kg/m³"))
    isnothing(feed.crystals) || print(display_io, ", crystals = ", _crystals_text(feed.crystals))
    isempty(feed.solvent_state) || print(display_io, ", solvent_state = ", join(keys(feed.solvent_state), ", "))
    print(display_io, ")")
end
Base.show(display_io::IO, ::MIME"text/plain", feed::CrystallisationFeed) = show(display_io, feed)

Base.show(display_io::IO, operation::MSMPROperation) =
    print(display_io, "MSMPROperation(volume = ", _quantity_text(operation.volume, "m³"),
        ", inflow = ", _profile_text(operation.inflow, "m³/s"), ")")
Base.show(display_io::IO, operation::FedBatchOperation) =
    print(display_io, "FedBatchOperation(initial_volume = ", _quantity_text(operation.initial_volume, "m³"),
        ", inflow = ", _profile_text(operation.inflow, "m³/s"), ")")

Base.show(display_io::IO, ::MIME"text/plain", operation::BatchOperation) = show(display_io, operation)
Base.show(display_io::IO, ::MIME"text/plain", operation::Union{MSMPROperation, FedBatchOperation}) =
    print(display_io, _labelled_summary(string(nameof(typeof(operation))), _operation_rows(operation)))
_operation_rows(operation::MSMPROperation) = ["volume" => _quantity_text(operation.volume, "m³"),
    "inflow" => _profile_text(operation.inflow, "m³/s"), "feed" => sprint(show, operation.feed)]
_operation_rows(operation::FedBatchOperation) = ["initial_volume" => _quantity_text(operation.initial_volume, "m³"),
    "inflow" => _profile_text(operation.inflow, "m³/s"), "feed" => sprint(show, operation.feed)]

# --- run conditions ---------------------------------------------------------

function Base.show(display_io::IO, ::MIME"text/plain", conditions::RunConditions)
    condition_rows = ["temperature" => sprint(show, conditions.temperature),
        "concentration" => _quantity_text(conditions.solvent.concentration, "kg/m³")]
    for (solvent_name, solvent_value) in pairs(Base.structdiff(conditions.solvent, (; concentration = nothing)))
        push!(condition_rows, string(solvent_name) => _quantity_text(solvent_value, ""))
    end
    push!(condition_rows, "crystals" => _crystals_text(conditions.crystals))
    push!(condition_rows, "operation" => sprint(show, conditions.operation))
    print(display_io, _labelled_summary("RunConditions", condition_rows))
end

Base.show(display_io::IO, conditions::RunConditions) =
    print(display_io, "RunConditions(", conditions.temperature, ", ",
        _quantity_text(conditions.solvent.concentration, "kg/m³"),
        isnothing(conditions.crystals) ? "" : ", seeded", ", ", conditions.operation, ")")

# --- solvers ----------------------------------------------------------------

const _ReportedSolver = Union{FiniteVol, WENO, MoM, QMOM, DQMOM}

_solver_summary(solver::Union{FiniteVol, WENO}) = string(nameof(typeof(solver)), "(", solver.meshsize, " cells)")
_solver_summary(solver::MoM) = string("MoM(", solver.nmoments, " moments)")
_solver_summary(solver::Union{QMOM, DQMOM}) = string(nameof(typeof(solver)), "(", solver.nquadrature, " nodes)")

_solver_rows(solver::Union{FiniteVol, WENO}) = ["mesh" => string(solver.meshsize, " cells over ",
    _number_text(solver.lmin * _METRES_TO_MICROMETRES), " … ", _size_text(solver.lmax))]
_solver_rows(solver::MoM) = ["moments" => string("µ0 … µ", solver.nmoments)]
_solver_rows(solver::Union{QMOM, DQMOM}) = ["quadrature" => string(solver.nquadrature, " nodes, moments µ0 … µ",
    2 * solver.nquadrature - 1)]

_stepper_text(::Nothing) = "solver default"
_stepper_text(stepper) = string(nameof(typeof(stepper)))

Base.show(display_io::IO, solver::_ReportedSolver) = print(display_io, _solver_summary(solver))
function Base.show(display_io::IO, ::MIME"text/plain", solver::_ReportedSolver)
    solver_rows = vcat(_solver_rows(solver), ["tolerances" => string("reltol = ", _number_text(solver.reltol),
        ", abstol = ", _number_text(solver.abstol), solver.auto_abstol ? ", auto_abstol" : ""),
        "stepper" => _stepper_text(solver.timestepping_algorithm)])
    print(display_io, _labelled_summary(string(nameof(typeof(solver))), solver_rows))
end

# --- kinetic models and the crystallisation model ---------------------------

_parameters_text(parameters::NamedTuple) = join((string(parameter_name, " = ", _parameter_value_text(parameter_value))
    for (parameter_name, parameter_value) in pairs(parameters)), ", ")
# Nested parameter groups print as `group = (a = 1, b = 2)`.
_parameter_value_text(parameter_value::NamedTuple) = string("(", _parameters_text(parameter_value), ")")
_parameter_value_text(parameter_value::Real) = _number_text(parameter_value)
_parameter_value_text(parameter_values::AbstractVector) = string("[", join(_number_text.(parameter_values), ", "), "]")

Base.show(display_io::IO, bound_kinetic::KineticModel) =
    print(display_io, nameof(typeof(bound_kinetic.law)), "(", _parameters_text(bound_kinetic.parameters), ")")
Base.show(display_io::IO, ::MIME"text/plain", bound_kinetic::KineticModel) = show(display_io, bound_kinetic)

# Slots left at their `No...` default are omitted from the compact model form.
function Base.show(display_io::IO, ::MIME"text/plain", crystal_model::CrystallisationModel)
    material_system = crystal_model.system
    model_rows = ["density" => string(_number_text(material_system.crystal_density), " kg/m³; shape factor ",
            _number_text(material_system.volume_shape_factor)),
        "solubility" => string(nameof(typeof(material_system.saturation_model)))]
    for slot_name in _MODEL_KINETIC_SLOTS
        push!(model_rows, string(slot_name) => sprint(show, getproperty(crystal_model, slot_name)))
    end
    print(display_io, _labelled_summary("CrystallisationModel", model_rows))
end

function Base.show(display_io::IO, crystal_model::CrystallisationModel)
    active_slots = [slot_name for slot_name in _MODEL_KINETIC_SLOTS
        if !_is_null_kinetic(getproperty(crystal_model, slot_name).law)]
    print(display_io, "CrystallisationModel(", join((string(slot_name, " = ", getproperty(crystal_model, slot_name))
        for slot_name in active_slots), ", "), ")")
end

# --- problems ---------------------------------------------------------------

_tspan_text(tspan::Tuple{Float64, Float64}) =
    string(_number_text(first(tspan)), " … ", _number_text(last(tspan)), " s")

function Base.show(display_io::IO, ::MIME"text/plain", problem::CrystallisationProblem)
    problem_rows = ["model" => sprint(show, problem.model), "conditions" => sprint(show, problem.conditions),
        "tspan" => _tspan_text(problem.tspan), "p" => string(length(problem.p), "-entry kinetic vector")]
    print(display_io, _labelled_summary("CrystallisationProblem", problem_rows))
end
Base.show(display_io::IO, problem::CrystallisationProblem) =
    print(display_io, "CrystallisationProblem(", problem.conditions, ", ", _tspan_text(problem.tspan), ")")

Base.show(display_io::IO, steady_problem::CrystallisationSteadyStateProblem) =
    print(display_io, "CrystallisationSteadyStateProblem(", steady_problem.problem.conditions,
        isnothing(steady_problem.u0) ? "" : ", initial guess", ")")
Base.show(display_io::IO, ::MIME"text/plain", steady_problem::CrystallisationSteadyStateProblem) =
    show(display_io, steady_problem)

# --- solutions --------------------------------------------------------------

_population_description(population::MeshPopulation) = string(size(population.numberdensity, 1), "-cell mesh")
_population_description(population::MomentPopulation) = string("moments µ0 … µ", size(population.moments, 1) - 1)
_population_description(population::QuadraturePopulation) = string(size(population.nodes, 1), "-node quadrature")

_retcode_text(retcode::SciMLBase.ReturnCode.T) = string("retcode = ", retcode)

_ode_stats_text(stats::SciMLBase.DEStats) = string(stats.nf, " RHS evaluations, ", stats.naccept,
    " accepted / ", stats.nreject, " rejected steps")
_ode_stats_text(other_stats) = "not recorded"

function _time_range_text(saved_times::AbstractVector)
    isempty(saved_times) && return "no saves"
    return string(_number_text(first(saved_times)), " … ", _number_text(last(saved_times)), " s (",
        length(saved_times), " saves)")
end

_size_trajectory_text(size_series::AbstractVector) = any(isnan, size_series) ? "not tracked by this solver" :
    _first_last_text(size_series, size_value -> _number_text(_display_real(size_value) * _METRES_TO_MICROMETRES)) * " µm"

function Base.show(display_io::IO, ::MIME"text/plain", solution::CrystallisationSolution)
    solution_rows = ["t" => _time_range_text(solution.t),
        "concentration" => _first_last_text(solution.concentration, _number_text) * " kg/m³",
        "d43" => _size_trajectory_text(solution.d43),
        "ODE" => _ode_stats_text(solution.ode_stats)]
    heading = string("CrystallisationSolution{", nameof(typeof(solution.population)), "} — ",
        _population_description(solution.population), ", ", _retcode_text(solution.retcode))
    print(display_io, _labelled_summary(heading, solution_rows))
end

function Base.show(display_io::IO, solution::CrystallisationSolution)
    saved_times = solution.t
    time_text = isempty(saved_times) ? "no saves" :
        string(_number_text(first(saved_times)), "–", _number_text(last(saved_times)), " s")
    print(display_io, "CrystallisationSolution(", nameof(typeof(solution.population)), ", ",
        length(saved_times), " saves, ", time_text, ", ", _retcode_text(solution.retcode), ")")
end

_steady_status_text(steady::CrystallisationSteadyStateSolution) =
    SciMLBase.successful_retcode(steady) ? "converged" : "not converged"
_steady_scalar_text(scalar_value::Real, to_text) = to_text(scalar_value)
_steady_scalar_text(other_value, to_text) = "not a scalar"

function Base.show(display_io::IO, ::MIME"text/plain", steady::CrystallisationSteadyStateSolution)
    steady_rows = ["concentration" => _steady_scalar_text(get(steady.equilibrium, :concentration, NaN),
            concentration_value -> _quantity_text(concentration_value, "kg/m³")),
        "d43" => _steady_scalar_text(get(steady.equilibrium, :d43, NaN), _size_text),
        "residence time" => _steady_scalar_text(get(steady.hydraulics, :residence_time, NaN),
            residence_time -> _quantity_text(residence_time, "s")),
        "scaled residual" => _steady_scalar_text(get(steady.diagnostics, :scaled_residual_norm, NaN), _number_text),
        "retcode" => string(steady.retcode)]
    print(display_io, _labelled_summary(string("CrystallisationSteadyStateSolution — ", _steady_status_text(steady)),
        steady_rows))
end
Base.show(display_io::IO, steady::CrystallisationSteadyStateSolution) =
    print(display_io, "CrystallisationSteadyStateSolution(", _steady_status_text(steady), ")")

# --- fitting and prediction results -----------------------------------------

_algorithm_label(algorithm) = string(nameof(typeof(algorithm)))
_algorithm_label(::Nothing) = "default"
_algorithm_label(algorithm::Metaheuristics.Algorithm) = string(nameof(typeof(algorithm.parameters)))

_specification_label(specification::OptimisationSpec) =
    string("OptimisationSpec(", _algorithm_label(specification.algorithm), ")")
_specification_label(specification::BayesianSpec) = string("BayesianSpec(", _bayesian_sampler_label(specification.sampler), ")")
# `sampler = nothing` is resolved to `Turing.NUTS()` by the Turing extension.
_bayesian_sampler_label(::Nothing) = "NUTS"
_bayesian_sampler_label(sampler) = _algorithm_label(sampler)
_specification_label(specification::ABCSpec) = string("ABCSpec(", _algorithm_label(specification.sampler), ")")

# Only Optimization.jl results report a convergence status.
_fit_status_suffix(backend_result) = ""
_fit_status_suffix(backend_result::SciMLBase.AbstractOptimizationSolution) =
    SciMLBase.successful_retcode(backend_result.retcode) ? ", converged" :
    string(", not converged (", backend_result.retcode, ")")

_total_loss(fit_result::FitResult) = sum(values(fit_result.objectives))

function Base.show(display_io::IO, ::MIME"text/plain", fit_result::FitResult)
    print(display_io, "FitResult — ", _specification_label(fit_result.preparation.specification),
        ", loss = ", _number_text(_total_loss(fit_result)), _fit_status_suffix(fit_result.backend_result))
    parameter_leaves = _named_parameter_leaves(fit_result.parameters)
    isempty(parameter_leaves) && return nothing
    name_width = maximum(length ∘ first, parameter_leaves)
    for (parameter_path, parameter_value) in parameter_leaves
        print(display_io, "\n  ", rpad(parameter_path, name_width + 3), _number_text(parameter_value))
    end
end
Base.show(display_io::IO, fit_result::FitResult) =
    print(display_io, "FitResult(", _specification_label(fit_result.preparation.specification),
        ", loss = ", _number_text(_total_loss(fit_result)), ", ", length(_named_parameter_leaves(fit_result.parameters)),
        " parameters)")

_draw_count(samples::ParameterSamples) = size(samples.values, 2)

Base.show(display_io::IO, samples::ParameterSamples) =
    print(display_io, "ParameterSamples(", length(samples.names), " parameters, ", _draw_count(samples),
        " draws, source = :", samples.source, ")")
Base.show(display_io::IO, ::MIME"text/plain", samples::ParameterSamples) = show(display_io, samples)

function Base.show(display_io::IO, ::MIME"text/plain", inference_result::InferenceResult)
    posterior_samples = inference_result.samples
    print(display_io, "InferenceResult — ", _specification_label(inference_result.preparation.specification),
        ", ", _draw_count(posterior_samples), " draws")
    name_width = maximum(length, posterior_samples.names)
    for (row_index, parameter_path) in enumerate(posterior_samples.names)
        draws = view(posterior_samples.values, row_index, :)
        print(display_io, "\n  ", rpad(parameter_path, name_width + 3), _number_text(Statistics.mean(draws)),
            " ± ", _number_text(length(draws) > 1 ? Statistics.std(draws) : 0.0))
    end
end
Base.show(display_io::IO, inference_result::InferenceResult) =
    print(display_io, "InferenceResult(", _specification_label(inference_result.preparation.specification),
        ", ", length(inference_result.samples.names), " parameters, ", _draw_count(inference_result.samples), " draws)")

function Base.show(display_io::IO, ::MIME"text/plain", predictions::PredictionResult)
    print(display_io, "PredictionResult — ", length(predictions.experiments), " runs, ",
        _draw_count(predictions.samples), " draws")
    for (run_id, ensemble) in zip(predictions.experiments, predictions.ensembles)
        print(display_io, "\n  run ", run_id, ": ", length(ensemble.t), " saves, observables (",
            join(keys(ensemble.predictions), ", "), "), ", count(ensemble.success), "/", length(ensemble.success),
            " draws succeeded")
    end
end
Base.show(display_io::IO, predictions::PredictionResult) =
    print(display_io, "PredictionResult(", length(predictions.experiments), " runs, ",
        _draw_count(predictions.samples), " draws)")
