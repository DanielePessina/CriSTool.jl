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
read one with `observable_values(ensemble, name)`. The `success` mask and
`diagnostics` (with each sample's `retcode`) retain every candidate, including
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

"""Solver diagnostics kept per ensemble sample; solution types without them report `nothing`."""
_sample_diagnostics(sample_solution::AbstractSolution) =
    (; retcode = sample_solution.retcode, convergence = nothing, reason = nothing)
_sample_diagnostics(failed_solution::FailedPreparedSimulation) =
    (; retcode = failed_solution.retcode, convergence = nothing, reason = failed_solution.reason)
_sample_diagnostics(steady_result::CrystallisationSteadyStateSolution) =
    (; retcode = steady_result.retcode, convergence = steady_result.diagnostics, reason = nothing)

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
prepared templates; the output does not depend on `parallel`.
"""
function run_ensemble(parameter_samples::AbstractMatrix, prepared_solves::PreparedSolves;
                      observables = nothing, parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleThreads())
    size(parameter_samples, 1) == _kinetic_parameter_count(prepared_solves) ||
        throw(ArgumentError("Ensemble sample rows must match the kinetic parameter axis."))
    sample_count = size(parameter_samples, 2)
    sample_count > 0 || throw(ArgumentError("An ensemble requires at least one parameter sample."))
    sample_solutions = Matrix{AbstractSolution}(undef, length(prepared_solves.runs), sample_count)
    _solve_ensemble!(parallel, sample_solutions, parameter_samples, prepared_solves)
    return map(enumerate(prepared_solves.runs)) do (run_index, prepared_run)
        _prediction_ensemble(prepared_solves.mode, prepared_run, view(sample_solutions, run_index, :), observables)
    end
end

function _solve_ensemble!(::EnsembleSerial, sample_solutions, parameter_samples, prepared_solves)
    for sample_index in axes(sample_solutions, 2), run_index in axes(sample_solutions, 1)
        sample_solutions[run_index, sample_index] = _solve_prepared(prepared_solves,
            prepared_solves.runs[run_index], collect(view(parameter_samples, :, sample_index)))
    end
    return sample_solutions
end

function _solve_ensemble!(::EnsembleThreads, sample_solutions, parameter_samples, prepared_solves)
    private_solves = _private_copy_pool(prepared_solves, min(Threads.nthreads(), length(sample_solutions)))
    solution_indices = CartesianIndices(sample_solutions)
    Threads.@threads for flat_index in eachindex(sample_solutions)
        run_index, sample_index = Tuple(solution_indices[flat_index])
        sample_solutions[run_index, sample_index] = _with_private_copy(private_solves) do task_solves
            _solve_prepared(task_solves, task_solves.runs[run_index],
                collect(view(parameter_samples, :, sample_index)))
        end
    end
    return sample_solutions
end

_solve_ensemble!(unsupported::SciMLBase.BasicEnsembleAlgorithm, args...) = _unsupported_parallel(unsupported)

function _prediction_ensemble(simulation_mode::AbstractSimulationMode, prepared_run::PreparedRun,
                              run_solutions::AbstractVector, observables)
    saved_times = prepared_run.saveat
    sample_success = [SciMLBase.successful_retcode(sample_solution) &&
                      _covers_save_times(simulation_mode, sample_solution, saved_times)
                      for sample_solution in run_solutions]
    successful_reference_index = findfirst(sample_success)
    reference_solution = run_solutions[something(successful_reference_index, firstindex(run_solutions))]
    prediction_names = observables === nothing ? _default_prediction_names(prepared_run, reference_solution) :
        Tuple(Symbol.(observables))
    prediction_matrices = map(prediction_names) do observable_name
        prediction_matrix = fill(NaN, length(run_solutions), length(saved_times))
        for (sample_index, sample_solution) in enumerate(run_solutions)
            sample_success[sample_index] || continue
            prediction_matrix[sample_index, :] .= _predicted_series(simulation_mode, (;),
                sample_solution, observable_name, saved_times)
        end
        prediction_matrix
    end
    initial_conditions = solvent_state(prepared_run.problem, _get_initial_state(prepared_run.problem))
    sample_diagnostics = [merge((; success = sample_success[sample_index],
            ode_stats = run_solutions[sample_index].ode_stats, initial_conditions),
            _sample_diagnostics(run_solutions[sample_index]))
        for sample_index in eachindex(run_solutions)]
    return PredictionEnsemble(copy(saved_times), NamedTuple{prediction_names}(Tuple(prediction_matrices)),
        sample_success, sample_diagnostics)
end
