"""
    crystallisation_solution(problem, solver, ode_solution)

Convert a SciML solution of the ODE that `crystallisation_odeproblem(problem, solver)`
generates into its physical result representation. The state layout must
match `solver`; a DQMOM ODE state is scaled weighted-node data, not the public
physical weights and nodes.
"""
function crystallisation_solution(problem::CrystallisationProblem, solver::AbstractSolver, ode_solution)
    isempty(ode_solution.u) && throw(ArgumentError("ODE solution has no saved states."))
    discretised_problem = DiscretisedProblem(problem, solver)
    expected_count = length(_get_initial_state(discretised_problem))
    all(length(saved_state) == expected_count for saved_state in ode_solution.u) ||
        throw(ArgumentError("ODE solution state layout does not match the configured problem."))
    return _wrap_solution(discretised_problem, ode_solution)
end

"""
    PredictionEnsemble

Named sample predictions for one prepared run at the saved times `t`.
`predictions` holds matrices with samples in rows and saved times in columns;
read one with `observable_values(ensemble, name)`. `success` and `diagnostics`
(one `EnsembleSampleDiagnostics` per sample, with its `retcode`) retain every
candidate, including
unsuccessful simulations.
"""
struct PredictionEnsemble{TP <: NamedTuple, TD <: AbstractVector}
    t::Vector{Float64}
    predictions::TP
    success::Vector{Bool}
    diagnostics::TD
end

observable_values(prediction_ensemble::PredictionEnsemble, observable_name::Symbol) =
    getproperty(prediction_ensemble.predictions, observable_name)

"""
    prediction_summary(ensemble, observable; skip_failed=false)

Summarize a named observable across samples with empirical 95% intervals.
Unsuccessful samples cause an error unless explicitly excluded; the returned
sample counts always report that exclusion. A single successful sample has
zero empirical standard deviation.
"""
function prediction_summary(prediction_ensemble::PredictionEnsemble,
                            observable_name::Symbol; skip_failed::Bool = false)
    successful_samples = prediction_ensemble.success
    !skip_failed && !all(successful_samples) &&
        throw(ArgumentError("Ensemble contains failed samples; inspect diagnostics or set skip_failed=true."))
    any(successful_samples) || throw(ArgumentError("Ensemble has no successful samples."))
    included_predictions = observable_values(prediction_ensemble, observable_name)[successful_samples, :]
    all(isfinite, included_predictions) ||
        throw(ArgumentError("Observable :$observable_name contains unavailable or nonfinite predictions."))
    successful_count = size(included_predictions, 1)
    return (; mean = vec(mean(included_predictions; dims = 1)),
        std = vec(std(included_predictions; dims = 1, corrected = successful_count > 1)),
        lower = [quantile(view(included_predictions, :, saved_index), 0.025)
            for saved_index in axes(included_predictions, 2)],
        upper = [quantile(view(included_predictions, :, saved_index), 0.975)
            for saved_index in axes(included_predictions, 2)],
        sample_count = successful_count,
        failed_count = count(!, successful_samples))
end

"""
    EnsembleSampleDiagnostics

Diagnostics of one ensemble sample: whether it `success`fully covered the saved
times, its `retcode`, the ODE `ode_stats` and the run's `initial_conditions`.
A steady sample adds its `convergence` checks; a solve that produced no
physical result records its `reason` as text. Fields that do not apply are
`nothing`.
"""
struct EnsembleSampleDiagnostics{TI <: NamedTuple, TC}
    success::Bool
    retcode::SciMLBase.ReturnCode.T
    ode_stats::Union{Nothing, SciMLBase.DEStats}
    initial_conditions::TI
    convergence::TC
    reason::Union{Nothing, String}
end

_sample_convergence(::AbstractSolution) = nothing
_sample_convergence(steady_result::CrystallisationSteadyStateSolution) = steady_result.diagnostics
_sample_reason(::AbstractSolution) = nothing
_sample_reason(failed_solution::FailedPreparedSimulation) = _failure_reason_text(failed_solution.reason)
_failure_reason_text(failure::Exception) = sprint(showerror, failure)
_failure_reason_text(failure) = string(failure)

# A transient sample must span the requested times; a steady value holds at any time.
_covers_save_times(::SteadyMode, sample_solution, saved_times) = true
_covers_save_times(::TransientMode, sample_solution, saved_times) =
    first(sample_solution.t) <= first(saved_times) && last(sample_solution.t) >= last(saved_times)

# Every built-in trajectory, configured solvent value and, for a flowing reactor, the volume.
function _default_prediction_names(prepared_run::PreparedRun, reference_result)
    # A steady result names its observables through its endpoint trajectory.
    reference_solution = _projection_trajectory(reference_result)
    named_physical_values = merge(state_vars(reference_solution), reactor_vars(reference_solution),
        size_metrics(reference_solution))
    builtin_names = Tuple(observable_name for observable_name in propertynames(named_physical_values)
        if getproperty(named_physical_values, observable_name) isa AbstractVector &&
           length(getproperty(named_physical_values, observable_name)) == length(reference_solution.t))
    return Tuple(unique((builtin_names..., propertynames(prepared_run.problem.initial_solvent_state)...,
        _operation_prediction_names(prepared_run.problem.operation)...)))
end
_operation_prediction_names(::BatchOperation) = ()
_operation_prediction_names(::AbstractCrystallisationOperation) = (:volume,)

"""
    run_ensemble(parameter_samples, prepared_solves; observables=nothing, parallel=EnsembleThreads())
        -> Vector{PredictionEnsemble}

Predict each prepared run for the full kinetic parameter vectors in the columns
of `parameter_samples`. Without `observables`, every built-in trajectory and
configured solvent value is predicted. With `parallel = EnsembleThreads()` (the
default) the (run, sample) pairs are solved on threads, each task on its own copy of the
prepared templates; the output does not depend on `parallel`. Each solution is
written into its run's rows as soon as it is solved and then dropped.
"""
function run_ensemble(parameter_samples::AbstractMatrix, prepared_solves::PreparedSolves;
                      observables = nothing, parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleThreads())
    size(parameter_samples, 1) == _kinetic_parameter_count(prepared_solves) ||
        throw(ArgumentError("Ensemble sample rows must match the kinetic parameter axis."))
    sample_count = size(parameter_samples, 2)
    sample_count > 0 || throw(ArgumentError("An ensemble requires at least one parameter sample."))
    run_count = length(prepared_solves.runs)
    ensemble_solve_source = _ensemble_solve_source(parallel, prepared_solves, run_count * sample_count)
    prediction_ensembles = Vector{PredictionEnsemble}(undef, run_count)
    reference_indices = zeros(Int, run_count)
    _ensemble_foreach(parallel, 1:run_count) do run_index
        reference_indices[run_index], prediction_ensembles[run_index] = _reference_prediction_ensemble(
            ensemble_solve_source, prepared_solves, run_index, parameter_samples, observables)
    end
    # Every other sample is solved, written to its own row and slot, and dropped.
    pair_indices = CartesianIndices((run_count, sample_count))
    _ensemble_foreach(parallel, eachindex(pair_indices)) do flat_index
        run_index, sample_index = Tuple(pair_indices[flat_index])
        sample_index == reference_indices[run_index] && return nothing
        prediction_ensemble = prediction_ensembles[run_index]
        _write_ensemble_sample!(prediction_ensemble, prepared_solves.mode,
            _solve_ensemble_sample(ensemble_solve_source, run_index, parameter_samples, sample_index), sample_index,
            prediction_ensemble.diagnostics[reference_indices[run_index]].initial_conditions)
        return nothing
    end
    # Narrow to the concrete ensemble type when every run shares one.
    return map(identity, prediction_ensembles)
end

# Serial samples solve on the prepared templates; threaded samples borrow a private copy per task.
_ensemble_solve_source(::EnsembleSerial, prepared_solves::PreparedSolves, pair_count) = prepared_solves
_ensemble_solve_source(::EnsembleThreads, prepared_solves::PreparedSolves, pair_count) =
    _private_copy_pool(prepared_solves, min(Threads.nthreads(), pair_count))
_ensemble_solve_source(unsupported::SciMLBase.BasicEnsembleAlgorithm, args...) = _unsupported_parallel(unsupported)

_with_ensemble_solves(evaluate, prepared_solves::PreparedSolves) = evaluate(prepared_solves)
_with_ensemble_solves(evaluate, private_solves::Channel) = _with_private_copy(evaluate, private_solves)

_solve_ensemble_sample(ensemble_solve_source, run_index::Int, parameter_samples::AbstractMatrix, sample_index::Int) =
    _with_ensemble_solves(ensemble_solve_source) do task_solves
        _solve_prepared(task_solves, task_solves.runs[run_index], collect(view(parameter_samples, :, sample_index)))
    end

_ensemble_foreach(ensemble_body, ::EnsembleSerial, ensemble_indices) = foreach(ensemble_body, ensemble_indices)
function _ensemble_foreach(ensemble_body, ::EnsembleThreads, ensemble_indices)
    Threads.@threads for ensemble_index in ensemble_indices
        ensemble_body(ensemble_index)
    end
    return nothing
end

_ensemble_sample_succeeded(simulation_mode::AbstractSimulationMode, sample_solution, saved_times) =
    SciMLBase.successful_retcode(sample_solution) && _covers_save_times(simulation_mode, sample_solution, saved_times)

_ensemble_prediction_names(::Nothing, prepared_run::PreparedRun, reference_solution) =
    _default_prediction_names(prepared_run, reference_solution)
_ensemble_prediction_names(observables, prepared_run::PreparedRun, reference_solution) = Tuple(Symbol.(observables))

"""
Solve one run's samples in order up to the first success, or through all of
them when none succeeds (the first sample is then the reference). Return the
reference index and the run's preallocated ensemble with the reference written.
The reference names the default observables; the convergence types met while
scanning fix the diagnostics type. Scanned failures are solved again later.
"""
function _reference_prediction_ensemble(ensemble_solve_source, prepared_solves::PreparedSolves, run_index::Int,
                                        parameter_samples::AbstractMatrix, observables)
    prepared_run = prepared_solves.runs[run_index]
    simulation_mode = prepared_solves.mode
    saved_times = prepared_run.saveat
    sample_count = size(parameter_samples, 2)
    reference_solution = _solve_ensemble_sample(ensemble_solve_source, run_index, parameter_samples, 1)
    convergence_type = Union{Nothing, typeof(_sample_convergence(reference_solution))}
    reference_index = scanned_count = 1
    while !_ensemble_sample_succeeded(simulation_mode, reference_solution, saved_times) && scanned_count < sample_count
        scanned_count += 1
        scanned_solution = _solve_ensemble_sample(ensemble_solve_source, run_index, parameter_samples, scanned_count)
        convergence_type = Union{convergence_type, typeof(_sample_convergence(scanned_solution))}
        if _ensemble_sample_succeeded(simulation_mode, scanned_solution, saved_times)
            reference_index, reference_solution = scanned_count, scanned_solution
        end
    end
    prediction_names = _ensemble_prediction_names(observables, prepared_run, reference_solution)
    initial_conditions = solvent_state(prepared_run.problem, _get_initial_state(prepared_run.problem))
    prediction_ensemble = PredictionEnsemble(copy(saved_times),
        NamedTuple{prediction_names}(map(observable_name -> fill(NaN, sample_count, length(saved_times)),
            prediction_names)),
        zeros(Bool, sample_count),
        Vector{EnsembleSampleDiagnostics{typeof(initial_conditions), convergence_type}}(undef, sample_count))
    _write_ensemble_sample!(prediction_ensemble, simulation_mode, reference_solution, reference_index,
        initial_conditions)
    return reference_index, prediction_ensemble
end

"""Write one solved sample's success flag, projected rows and diagnostics into its own slots."""
function _write_ensemble_sample!(prediction_ensemble::PredictionEnsemble, simulation_mode::AbstractSimulationMode,
                                 sample_solution, sample_index::Int, initial_conditions)
    sample_succeeded = _ensemble_sample_succeeded(simulation_mode, sample_solution, prediction_ensemble.t)
    prediction_ensemble.success[sample_index] = sample_succeeded
    if sample_succeeded
        for (observable_name, prediction_matrix) in pairs(prediction_ensemble.predictions)
            prediction_matrix[sample_index, :] .= _predicted_series(simulation_mode, (;),
                sample_solution, observable_name, prediction_ensemble.t)
        end
    end
    prediction_ensemble.diagnostics[sample_index] = eltype(prediction_ensemble.diagnostics)(sample_succeeded,
        sample_solution.retcode, sample_solution.ode_stats, initial_conditions,
        _sample_convergence(sample_solution), _sample_reason(sample_solution))
    return prediction_ensemble
end
