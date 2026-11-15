# `predict` extends StatsAPI.predict, the function Turing and StatsBase export,
# so `using CriSTool, Turing` resolves to one `predict`.
import StatsAPI: predict

"""Declare whether a custom loss represents a supplied-noise negative log likelihood."""
is_likelihood(::AbstractPELossFunction) = false
is_likelihood(::GaussianNLL) = true

"""Named objective accounting. Custom losses default to a total and may extend this seam."""
loss_terms(loss_function::AbstractPELossFunction, prepared_setup::LossSetup, parameter_values) =
    (; total = loss(loss_function, prepared_setup, parameter_values))
function loss_terms(loss_function::Union{GaussianNLL, MeanAbsoluteError}, prepared_setup::LossSetup, parameter_values)
    return NamedTuple{Tuple(prepared_setup.observable_names)}(
        Tuple(_loss_objectives(loss_function, prepared_setup, parameter_values)))
end

# An estimation problem selects its kinetic parameters by named bounds, named priors,
# or both (then with the same names in the same order); the selection follows them.
_selected_tree(::Nothing, ::Nothing) =
    throw(ArgumentError("Select kinetic parameters with named bounds or named priors."))
_selected_tree(bounds::NamedTuple, ::Nothing) = bounds
_selected_tree(::Nothing, priors::NamedTuple) = priors
function _selected_tree(bounds::NamedTuple, priors::NamedTuple)
    first.(_named_parameter_leaves(bounds)) == first.(_named_parameter_leaves(priors)) ||
        throw(ArgumentError("Bounds and priors must name the same kinetic parameters in the same order."))
    return bounds
end

_validate_bounds(::Nothing) = nothing
function _validate_bounds(bounds::NamedTuple)
    for bound_pair in last.(_named_parameter_leaves(bounds))
        bound_pair isa Tuple && length(bound_pair) == 2 &&
            all(bound_value -> bound_value isa Real && isfinite(bound_value), bound_pair) &&
            bound_pair[1] < bound_pair[2] || throw(ArgumentError("Each bound must be a finite (lower, upper) pair."))
    end
end
_validate_priors(::Nothing) = nothing
function _validate_priors(priors::NamedTuple)
    all(prior_value -> prior_value isa Distributions.UnivariateDistribution,
        last.(_named_parameter_leaves(priors))) ||
        throw(ArgumentError("Supply a univariate prior per selected kinetic parameter."))
end

"""`(lower, upper)` bounds of the selection; optimisers need bounds."""
_bound_pairs(::Nothing) = throw(ArgumentError(
    "Optimisers search inside bounds; build the EstimationProblem with `bounds`."))
_bound_pairs(bounds::NamedTuple) = last.(_named_parameter_leaves(bounds))

"""Univariate priors of the selection, in selection order; samplers need priors."""
_prior_distributions(::Nothing) = throw(ArgumentError(
    "Samplers draw from priors; build the EstimationProblem with `priors`."))
_prior_distributions(priors::NamedTuple) = last.(_named_parameter_leaves(priors))

function _named_parameter_leaves(named_tree::NamedTuple, path_prefix::String = "")
    leaf_pairs = Pair{String, Any}[]
    for field_name in keys(named_tree)
        path_name = isempty(path_prefix) ? string(field_name) : path_prefix * "." * string(field_name)
        field_value = getproperty(named_tree, field_name)
        if field_value isa NamedTuple
            append!(leaf_pairs, _named_parameter_leaves(field_value, path_name))
        else
            push!(leaf_pairs, path_name => field_value)
        end
    end
    return leaf_pairs
end

_model_named_values(crystal_model::CrystallisationModel) = NamedTuple{_MODEL_KINETIC_SLOTS}(
    Tuple(getproperty(crystal_model, slot_name).parameters for slot_name in _MODEL_KINETIC_SLOTS))

"""Names, indices and fixed values resolved once outside the solve loop."""
struct KineticSelection{Axis, Baseline}
    names::Vector{String}
    indices::Vector{Int}
    axis::Axis
    baseline::Baseline
end

_kinetic_selection(crystal_model::CrystallisationModel, selected_tree::NamedTuple) =
    _kinetic_selection(crystal_model, first.(_named_parameter_leaves(selected_tree)))

function _kinetic_selection(crystal_model::CrystallisationModel, selected_names::AbstractVector{<:AbstractString})
    _validate_material_model(crystal_model)
    model_values = ComponentArray(; _model_named_values(crystal_model)...)
    all_names = ComponentArrays.labels(model_values)
    isempty(selected_names) && throw(ArgumentError("Select at least one kinetic parameter."))
    selected_indices = map(selected_names) do selected_name
        selected_index = findfirst(==(selected_name), all_names)
        selected_index === nothing && throw(ArgumentError("Unknown kinetic parameter $selected_name."))
        selected_index
    end
    return KineticSelection(collect(String, selected_names), selected_indices, first(getaxes(model_values)), collect(model_values))
end

"""The model's current values of the selected parameters."""
_selected_baseline(kinetic_selection::KineticSelection) = kinetic_selection.baseline[kinetic_selection.indices]

function _expanded_parameters(kinetic_selection::KineticSelection, selected_values::AbstractVector)
    length(selected_values) == length(kinetic_selection.indices) || throw(DimensionMismatch("Selected parameter count mismatch."))
    scalar_type = promote_type(eltype(selected_values), eltype(kinetic_selection.baseline))
    expanded_values = scalar_type.(kinetic_selection.baseline)
    expanded_values[kinetic_selection.indices] .= selected_values
    return expanded_values
end

function _reconstruct_parameter_tree(parameter_template::NamedTuple, named_values)
    return NamedTuple{keys(parameter_template)}(map(keys(parameter_template)) do field_name
        original_value = getproperty(parameter_template, field_name)
        replacement_value = getproperty(named_values, field_name)
        original_value isa NamedTuple ? _reconstruct_parameter_tree(original_value, replacement_value) : replacement_value
    end)
end

function _selected_model(crystal_model::CrystallisationModel, kinetic_selection, selected_values)
    reconstructed_values = ComponentArray(_expanded_parameters(kinetic_selection, selected_values), kinetic_selection.axis)
    reconstructed_kinetics = NamedTuple{_MODEL_KINETIC_SLOTS}(map(_MODEL_KINETIC_SLOTS) do slot_name
        original_kinetic = getproperty(crystal_model, slot_name)
        KineticModel(original_kinetic.law; parameters = _reconstruct_parameter_tree(
            original_kinetic.parameters, getproperty(reconstructed_values, slot_name)))
    end)
    return CrystallisationModel(crystal_model.system; reconstructed_kinetics...)
end

"""Selected named values as a nested tuple, suitable for inspection or rebinding."""
function _selected_named_values(kinetic_selection::KineticSelection, selected_values)
    named_values = (;)
    for (parameter_path, parameter_value) in zip(kinetic_selection.names, selected_values)
        named_values = _set_named_parameter_path(named_values, split(parameter_path, '.'), parameter_value)
    end
    return named_values
end

function _set_named_parameter_path(named_values::NamedTuple, path_parts, parameter_value)
    root_name = Symbol(first(path_parts))
    replacement_value = length(path_parts) == 1 ? parameter_value :
        _set_named_parameter_path(get(named_values, root_name, (;)), path_parts[2:end], parameter_value)
    return merge(named_values, NamedTuple{(root_name,)}((replacement_value,)))
end

# Transient runs read time-zero observations as initial values; steady runs relax from numeric conditions.
_run_conditions(::TransientMode, experiment_run::CrystallisationExperiment) = _experiment_conditions(experiment_run)
function _run_conditions(::SteadyMode, experiment_run::CrystallisationExperiment)
    unresolved_sources = observation_sources(experiment_run.conditions)
    isempty(unresolved_sources) || throw(ArgumentError("Steady fits relax from explicit numeric conditions; " *
        "experiment $(experiment_run.exp_id) reads $(Tuple(unresolved_sources)) from observations."))
    return experiment_run.conditions
end

# Prepared runs integrate over their observation times; the span records the same range.
_model_experiment_problem(crystal_model, experiment_run, simulation_mode; problem_options...) =
    CrystallisationProblem(crystal_model, _run_conditions(simulation_mode, experiment_run),
        _observation_tspan(_observation_times(experiment_run)); problem_options...)
_observation_tspan(requested_times::AbstractVector) = (0.0, max(0.0, maximum(requested_times; init = 0.0)))

"""Likelihood studies need supplied positive variances for every scored, weighted measurement."""
_validate_likelihood_noise(::AbstractPELossFunction, prepared_setup::LossSetup) = nothing
function _validate_likelihood_noise(likelihood::GaussianNLL, prepared_setup::LossSetup)
    likelihood.variance_model isa MeasuredVariance ||
        throw(ArgumentError("Likelihood studies require supplied measured variances; relative noise is not inferred."))
    for (experiment_index, experiment_run) in enumerate(prepared_setup.experiments)
        for (objective_index, observable_name) in enumerate(prepared_setup.observable_names)
            iszero(prepared_setup.observable_weights[objective_index]) && continue
            hasproperty(experiment_run.observables, observable_name) || continue
            target_indices = prepared_setup.included_observations[experiment_index][observable_name]
            isempty(target_indices) && continue
            measured_series = getproperty(experiment_run.observables, observable_name)
            measured_series.variance !== nothing && all(measured_series.variance[target_indices] .> 0) ||
                throw(ArgumentError("Experiment $(experiment_run.exp_id), :$observable_name requires supplied positive variances."))
        end
    end
end

"""
    EstimationProblem(model, experiments; bounds=nothing, priors=nothing, loss=GaussianNLL(),
                      solver=MoM(), mode=TransientMode(), problem_options=(;),
                      observable_order=nothing, observable_projections=(;), alg=nothing,
                      solve_options=(;), callback_factory=nothing)

Estimate the kinetic parameters of `model` named by `bounds` or `priors` from the
measured `experiments`, scored by `loss`, with every run simulated by `solver`.
`solve(problem, alg)` runs the estimation algorithm:

- a Metaheuristics algorithm (`DE()`, `NSGA2()`, ...) or an Optimization.jl
  optimiser (`LBFGS()` from OptimizationOptimJL) searches inside `bounds` and
  returns a `FitResult`;
- `NUTS()` (or another Turing sampler; requires `using Turing`) samples the
  posterior under `priors` and returns an `InferenceResult`;
- `ABCDE()` or `ABCDETurner()` samples by approximate Bayesian computation under
  `priors` and returns an `InferenceResult`.

`bounds` are named `(lower, upper)` pairs and `priors` named univariate
distributions, for example `bounds = (growth = (growth_order = (1.0, 4.0),),)`.
Supplying both lets one problem be optimised and sampled; they must then name
the same parameters in the same order.

Construction resolves each experiment's conditions, validates bounds, priors and
noise, and builds the reusable solve templates once. Time-zero observations
consumed by `FromObservation` conditions are not scored. `mode = SteadyMode(...)`
relaxes each run to its MSMPR steady state and needs numeric conditions to relax
from. `alg`, `solve_options` and `callback_factory` configure the ODE solves as in
`solve(problem, solver)`. A likelihood loss (`GaussianNLL`) needs supplied
measurement variances; noise is never fitted. `remake(problem; kwargs...)`
rebuilds the problem with the given keywords replaced.
"""
struct EstimationProblem{Model, Bounds <: Union{Nothing, NamedTuple}, Priors <: Union{Nothing, NamedTuple},
                         Loss <: AbstractPELossFunction, Solver <: AbstractSolver,
                         ProblemOptions <: NamedTuple, ObservableOrder, Selection <: KineticSelection,
                         Setup <: LossSetup} <: SciMLBase.AbstractSciMLProblem
    model::Model
    bounds::Bounds
    priors::Priors
    loss::Loss
    solver::Solver
    problem_options::ProblemOptions
    observable_order::ObservableOrder
    selection::Selection
    setup::Setup
end

function EstimationProblem(crystal_model::CrystallisationModel,
        measured_runs::AbstractVector{<:CrystallisationExperiment};
        bounds::Union{Nothing, NamedTuple} = nothing, priors::Union{Nothing, NamedTuple} = nothing,
        loss::AbstractPELossFunction = GaussianNLL(), solver::AbstractSolver = MoM(),
        mode::AbstractSimulationMode = TransientMode(), problem_options::NamedTuple = (;),
        observable_order = nothing, observable_projections::NamedTuple = (;), alg = nothing,
        solve_options::NamedTuple = (;), callback_factory = nothing)
    isempty(measured_runs) && throw(ArgumentError("Supply at least one experiment."))
    kinetic_selection = _kinetic_selection(crystal_model, _selected_tree(bounds, priors))
    _validate_bounds(bounds)
    _validate_priors(priors)
    physical_runs = [_model_experiment_problem(crystal_model, experiment_run, mode; problem_options...)
        for experiment_run in measured_runs]
    prepared_setup = prepare_loss(physical_runs, measured_runs; solver, mode, loss_function = loss,
        observable_order, observable_projections, alg, solve_options, callback_factory)
    _validate_likelihood_noise(loss, prepared_setup)
    return EstimationProblem(crystal_model, bounds, priors, loss, solver, problem_options, observable_order,
        kinetic_selection, prepared_setup)
end

"""
    remake(problem::EstimationProblem; model, experiments, bounds, priors, loss, solver, mode,
           problem_options, observable_order, observable_projections, alg, solve_options,
           callback_factory)

Build the estimation problem again with the given keywords replaced; the others
keep their values. `bounds = nothing` or `priors = nothing` drops that selection.
"""
function SciMLBase.remake(estimation_problem::EstimationProblem; model = missing, experiments = missing,
        bounds = missing, priors = missing, loss = missing, solver = missing, mode = missing,
        problem_options = missing, observable_order = missing, observable_projections = missing,
        alg = missing, solve_options = missing, callback_factory = missing)
    prepared_setup = estimation_problem.setup
    prepared_solves = prepared_setup.solves
    return EstimationProblem(coalesce(model, estimation_problem.model),
        coalesce(experiments, prepared_setup.experiments);
        bounds = coalesce(bounds, estimation_problem.bounds),
        priors = coalesce(priors, estimation_problem.priors),
        loss = coalesce(loss, estimation_problem.loss),
        solver = coalesce(solver, estimation_problem.solver),
        mode = coalesce(mode, prepared_solves.mode),
        problem_options = coalesce(problem_options, estimation_problem.problem_options),
        observable_order = coalesce(observable_order, estimation_problem.observable_order),
        observable_projections = coalesce(observable_projections, prepared_setup.observable_projections),
        alg = coalesce(alg, prepared_solves.requested_algorithm),
        solve_options = coalesce(solve_options, prepared_solves.solve_options),
        callback_factory = coalesce(callback_factory, prepared_solves.callback_factory))
end

"""Same problem with the experiments of each loss evaluation solved under `parallel`."""
function _with_parallel(estimation_problem::EstimationProblem, parallel::SciMLBase.BasicEnsembleAlgorithm)
    prepared_setup = estimation_problem.setup
    parallel_setup = LossSetup(prepared_setup.solves, prepared_setup.experiments, prepared_setup.observable_names,
        prepared_setup.included_observations, prepared_setup.observable_weights,
        prepared_setup.observable_projections, parallel)
    return EstimationProblem(estimation_problem.model, estimation_problem.bounds, estimation_problem.priors,
        estimation_problem.loss, estimation_problem.solver, estimation_problem.problem_options,
        estimation_problem.observable_order, estimation_problem.selection, parallel_setup)
end

"""Evaluate the selected-parameter objective; AD scalar types are preserved."""
loss(estimation_problem::EstimationProblem, selected_values::AbstractVector) = loss(estimation_problem.loss,
    estimation_problem.setup, _expanded_parameters(estimation_problem.selection, selected_values))

"""Named objectives of the problem's loss at selected parameter values."""
loss_terms(estimation_problem::EstimationProblem, selected_values::AbstractVector) =
    loss_terms(estimation_problem.loss, estimation_problem.setup,
        _expanded_parameters(estimation_problem.selection, selected_values))

"""
A fitted model, named selected values, observable objectives, the optimisation
algorithm, its native result, and the `EstimationProblem` that was solved.
"""
struct FitResult{Model, Parameters, Objectives, Algorithm, Backend, Problem}
    model::Model
    parameters::Parameters
    objectives::Objectives
    alg::Algorithm
    backend_result::Backend
    estimation_problem::Problem
end

"""Parallel level of a solve: candidates (population, particles or chains) run under `parallel`."""
struct ParallelCandidates end
"""Parallel level of a solve: the experiments inside each loss evaluation run under `parallel`."""
struct ParallelExperiments end

"""
    _parallel_level(alg) -> ParallelCandidates() or ParallelExperiments()

The one level of an estimation that receives `solve(...; parallel)`; the other
level runs serially, since threading both levels only adds scheduling overhead.
MCMC samplers choose by chain count through `_chain_parallel_level`.
"""
_parallel_level(::Metaheuristics.Algorithm) = ParallelCandidates()
_parallel_level(::AbstractABCSampler) = ParallelCandidates()
_parallel_level(optimiser) = ParallelExperiments()
_chain_parallel_level(chain_count::Integer) = chain_count > 1 ? ParallelCandidates() : ParallelExperiments()

_candidate_parallel(::ParallelCandidates, parallel::SciMLBase.BasicEnsembleAlgorithm) = parallel
_candidate_parallel(::ParallelExperiments, parallel::SciMLBase.BasicEnsembleAlgorithm) = EnsembleSerial()
_experiment_parallel(::ParallelCandidates, parallel::SciMLBase.BasicEnsembleAlgorithm) = EnsembleSerial()
_experiment_parallel(::ParallelExperiments, parallel::SciMLBase.BasicEnsembleAlgorithm) = parallel

"""Split `parallel` by level: the candidate ensemble, and the problem whose loss solves experiments under the other."""
function _split_parallel(estimation_problem::EstimationProblem, parallel_level,
                         parallel::SciMLBase.BasicEnsembleAlgorithm)
    return _candidate_parallel(parallel_level, parallel),
           _with_parallel(estimation_problem, _experiment_parallel(parallel_level, parallel))
end

# Backends that take a Bool (Metaheuristics `parallel_evaluation`, KissABC `HPC`) or a
# pool size convert the candidate ensemble here.
_threads_candidates(::EnsembleSerial) = false
_threads_candidates(::EnsembleThreads) = true
_threads_candidates(unsupported::SciMLBase.BasicEnsembleAlgorithm) = _unsupported_parallel(unsupported)
_candidate_copy_count(::EnsembleSerial) = 0
_candidate_copy_count(::EnsembleThreads) = Threads.nthreads()
_candidate_copy_count(unsupported::SciMLBase.BasicEnsembleAlgorithm) = _unsupported_parallel(unsupported)

"""Lower bounds, upper bounds and the model's current selected values, which must lie inside them."""
function _search_box(estimation_problem::EstimationProblem)
    bound_pairs = _bound_pairs(estimation_problem.bounds)
    lower_bounds, upper_bounds = first.(bound_pairs), last.(bound_pairs)
    starting_values = _selected_baseline(estimation_problem.selection)
    all(lower_bounds .<= starting_values .<= upper_bounds) ||
        throw(ArgumentError("Current selected values must lie within bounds."))
    return lower_bounds, upper_bounds, starting_values
end

# Preserve caller configuration while isolating mutable status, parameters and RNG.
function _configured_mh_algorithm(supplied_algorithm::Metaheuristics.Algorithm;
                                  nparticles = nothing, generations = nothing, parallel_evaluation = nothing)
    run_algorithm = deepcopy(supplied_algorithm)
    run_algorithm.status = Metaheuristics.State(nothing, [])
    if nparticles !== nothing
        nparticles > 0 || throw(ArgumentError("nparticles must be positive"))
        run_algorithm.parameters.N = nparticles
    end
    if generations !== nothing
        generations > 0 || throw(ArgumentError("generations must be positive"))
        run_algorithm.options.iterations = generations
    end
    parallel_evaluation === nothing || (run_algorithm.options.parallel_evaluation = parallel_evaluation)
    return run_algorithm
end

"""
    solve(problem::EstimationProblem, optimiser; adtype=AutoForwardDiff(),
          parallel=EnsembleSerial(), kwargs...) -> FitResult

Optimise the selected kinetic parameters with an Optimization.jl optimiser (for
example `LBFGS()` from OptimizationOptimJL), searching inside the bounds from
the model's current values. `adtype` differentiates the loss; the remaining
keywords (such as `maxiters`) go to the Optimization.jl `solve`.
`parallel = EnsembleThreads()` solves the experiments of each loss evaluation
(and so each ForwardDiff gradient) on threads; scores are reduced in experiment
order, so the result is bitwise identical to `EnsembleSerial()`.
"""
function SciMLBase.solve(estimation_problem::EstimationProblem, optimiser; adtype = AutoForwardDiff(),
        parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleSerial(), optimiser_options...)
    _, split_problem = _split_parallel(estimation_problem, _parallel_level(optimiser), parallel)
    lower_bounds, upper_bounds, starting_values = _search_box(split_problem)
    optimization_function = OptimizationBase.OptimizationFunction(
        (candidate_values, fit_context) -> loss(fit_context, candidate_values), adtype)
    optimization_problem = OptimizationBase.OptimizationProblem(optimization_function,
        starting_values, split_problem; lb = lower_bounds, ub = upper_bounds)
    optimization_solution = OptimizationBase.solve(optimization_problem, optimiser; optimiser_options...)
    return _model_fit_result(split_problem, optimiser, optimization_solution.u, optimization_solution)
end

"""
    solve(problem::EstimationProblem, alg::Metaheuristics.Algorithm; parallel=EnsembleSerial())
        -> FitResult or Vector{FitResult}

Population search inside the bounds with a Metaheuristics algorithm (`DE()`,
`ECA()`, `NSGA2()`, ...) run on a private copy, so `alg`, its options, status
and RNG are not consumed. A scalar objective returns a `FitResult`; NSGA2
minimises each named observable objective and returns one `FitResult` per
Pareto candidate. `parallel = EnsembleThreads()` evaluates the population on
threads, each task on a private copy of the solve templates.
"""
function SciMLBase.solve(estimation_problem::EstimationProblem, supplied_algorithm::Metaheuristics.Algorithm;
        parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleSerial())
    candidate_parallel, split_problem = _split_parallel(estimation_problem,
        _parallel_level(supplied_algorithm), parallel)
    lower_bounds, upper_bounds, _ = _search_box(split_problem)
    private_algorithm = _configured_mh_algorithm(supplied_algorithm;
        parallel_evaluation = _threads_candidates(candidate_parallel))
    private_problems = _private_copy_pool(split_problem, _candidate_copy_count(candidate_parallel))
    candidate_objective = _MetaheuristicsObjective(private_algorithm.parameters, split_problem, private_problems)
    final_state = Metaheuristics.optimize(candidate_objective,
        Metaheuristics.boxconstraints(lower_bounds, upper_bounds), private_algorithm)
    return _metaheuristics_fit_results(private_algorithm.parameters, split_problem, supplied_algorithm, final_state)
end

"""Metaheuristics objective: one candidate serially, or a population in parallel on pooled private copies."""
struct _MetaheuristicsObjective{Parameters, Problem}
    parameters::Parameters
    estimation_problem::Problem
    private_problems::Channel{Problem}
end

# Metaheuristics requires a method accepting `Any`; the argument shape then selects the evaluation.
(candidate_objective::_MetaheuristicsObjective)(candidate_values) =
    _evaluate_candidates(candidate_objective, candidate_values)

_evaluate_candidates(candidate_objective::_MetaheuristicsObjective, candidate_values::AbstractVector) =
    _candidate_score(candidate_objective.parameters, candidate_objective.estimation_problem, candidate_values)

function _evaluate_candidates(candidate_objective::_MetaheuristicsObjective, candidate_values::AbstractMatrix)
    candidate_scores = Vector{_candidate_score_type(candidate_objective.parameters)}(undef, size(candidate_values, 1))
    Threads.@threads for candidate_index in axes(candidate_values, 1)
        candidate_scores[candidate_index] = _with_private_copy(candidate_objective.private_problems) do private_problem
            _candidate_score(candidate_objective.parameters, private_problem,
                collect(view(candidate_values, candidate_index, :)))
        end
    end
    return _population_scores(candidate_objective.parameters, candidate_scores)
end

# Scalar algorithms minimise the total loss; NSGA2 minimises each named objective.
_candidate_score(::Metaheuristics.AbstractParameters, estimation_problem, candidate_values) =
    loss(estimation_problem, candidate_values)
_candidate_score(::Metaheuristics.NSGA2, estimation_problem, candidate_values) =
    (collect(Float64, values(loss_terms(estimation_problem, candidate_values))), [0.0], [0.0])

_candidate_score_type(::Metaheuristics.AbstractParameters) = Float64
_candidate_score_type(::Metaheuristics.NSGA2) = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

_population_scores(::Metaheuristics.AbstractParameters, candidate_scores) = candidate_scores
_population_scores(::Metaheuristics.NSGA2, candidate_scores) =
    (permutedims(reduce(hcat, first.(candidate_scores))),
     zeros(length(candidate_scores), 1), zeros(length(candidate_scores), 1))

_metaheuristics_fit_results(::Metaheuristics.AbstractParameters, estimation_problem, supplied_algorithm, final_state) =
    _model_fit_result(estimation_problem, supplied_algorithm, Metaheuristics.minimizer(final_state), final_state)
function _metaheuristics_fit_results(::Metaheuristics.NSGA2, estimation_problem, supplied_algorithm, final_state)
    pareto_solutions = Metaheuristics.get_non_dominated_solutions(final_state.population)
    pareto_values = Metaheuristics.positions(pareto_solutions)
    # The stored objective rows are the `loss_terms` scored for these exact candidates.
    pareto_objectives = Metaheuristics.fvals(pareto_solutions)
    objective_names = Tuple(estimation_problem.setup.observable_names)
    return [_model_fit_result(estimation_problem, supplied_algorithm, collect(candidate_values),
                NamedTuple{objective_names}(Tuple(candidate_objectives)), final_state)
            for (candidate_values, candidate_objectives) in zip(eachrow(pareto_values), eachrow(pareto_objectives))]
end

_model_fit_result(estimation_problem, fit_algorithm, optimal_values, backend_result) =
    _model_fit_result(estimation_problem, fit_algorithm, optimal_values,
        loss_terms(estimation_problem, optimal_values), backend_result)

function _model_fit_result(estimation_problem, fit_algorithm, optimal_values, objective_terms, backend_result)
    fitted_model = _selected_model(estimation_problem.model, estimation_problem.selection, optimal_values)
    return FitResult(fitted_model, _selected_named_values(estimation_problem.selection, optimal_values),
        objective_terms, fit_algorithm, backend_result, estimation_problem)
end

"""Joint parameter draws: selected named paths in rows, common sample IDs in columns."""
struct ParameterSamples{Values, Source}
    names::Vector{String}
    values::Values
    sample_ids::Vector{Int}
    source::Source
    function ParameterSamples(parameter_names, sample_values::AbstractMatrix;
            sample_ids = collect(axes(sample_values, 2)), source = :user)
        named_paths = String.(parameter_names)
        length(named_paths) == size(sample_values, 1) && size(sample_values, 2) > 0 ||
            throw(DimensionMismatch("Sample rows must match selected names and columns must be nonempty."))
        length(unique(named_paths)) == length(named_paths) || throw(ArgumentError("Duplicate parameter paths."))
        length(sample_ids) == size(sample_values, 2) && length(unique(sample_ids)) == length(sample_ids) ||
            throw(ArgumentError("Sample IDs must be unique and match columns."))
        all(isfinite, sample_values) || throw(ArgumentError("Parameter samples must be finite."))
        new{typeof(copy(sample_values)), typeof(source)}(named_paths, copy(sample_values), collect(Int, sample_ids), source)
    end
end

"""Construct correlated named draws from nested tuples of equally sized vectors."""
function ParameterSamples(; source = :user, kwargs...)
    sample_leaves = _named_parameter_leaves((; kwargs...))
    isempty(sample_leaves) && throw(ArgumentError("Supply selected kinetic sample vectors."))
    sample_vectors = last.(sample_leaves)
    all(sample_vector -> sample_vector isa AbstractVector, sample_vectors) || throw(ArgumentError("Each parameter needs a sample vector."))
    return ParameterSamples(first.(sample_leaves), permutedims(hcat(sample_vectors...)); source)
end

"""
Posterior draws, the sampling algorithm, its native chain or sampler result, and
the `EstimationProblem` that was solved.
"""
struct InferenceResult{Model, Samples, Algorithm, Backend, Problem}
    model::Model
    samples::Samples
    alg::Algorithm
    backend_result::Backend
    estimation_problem::Problem
end

"""Check what MCMC sampling needs: priors, a likelihood loss and positive counts."""
function _validate_posterior_sampling(estimation_problem::EstimationProblem, sample_count::Integer,
                                      chain_count::Integer)
    _prior_distributions(estimation_problem.priors)
    is_likelihood(estimation_problem.loss) ||
        throw(ArgumentError("Posterior sampling requires a likelihood loss, not an error discrepancy."))
    sample_count > 0 && chain_count > 0 || throw(ArgumentError("Sample and chain counts must be positive."))
    return nothing
end

"""The named posterior draws of an MCMC chain, as an `InferenceResult` (used by `CriSToolTuringExt`)."""
function _chain_inference_result(estimation_problem::EstimationProblem, sampler, posterior_chain)
    selected_symbols = Symbol.(estimation_problem.selection.names)
    renamed_chain = rename_chain(posterior_chain, selected_symbols)
    posterior_draws = ParameterSamples(estimation_problem.selection.names,
        chains_to_matrix(renamed_chain; params = selected_symbols); source = :nuts)
    return InferenceResult(estimation_problem.model, posterior_draws, sampler, renamed_chain, estimation_problem)
end

"""
    solve(problem::EstimationProblem, sampler::AbstractABCSampler; target, n_particles=1024,
          n_generations=128, stopping=AllGenerations(), parallel=EnsembleSerial()) -> InferenceResult

Sample the priors by approximate Bayesian computation (`ABCDE()` or
`ABCDETurner()`), accepting particles whose loss is within the explicit finite
discrepancy threshold `target`. `parallel = EnsembleThreads()` evaluates
particles on threads, each on a private copy of the solve templates; the
experiments of one discrepancy run serially.
"""
function SciMLBase.solve(estimation_problem::EstimationProblem, abc_sampler::AbstractABCSampler;
        target::Real, n_particles::Integer = 1024, n_generations::Integer = 128, stopping::AbstractABCStopping = AllGenerations(),
        parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleSerial())
    isfinite(target) || throw(ArgumentError("ABC target must be explicit and finite."))
    n_particles > 0 && n_generations > 0 || throw(ArgumentError("Particle and generation counts must be positive."))
    joint_prior = product_distribution(_prior_distributions(estimation_problem.priors))
    particle_parallel, split_problem = _split_parallel(estimation_problem, _parallel_level(abc_sampler), parallel)
    reference_values = _selected_baseline(split_problem.selection)
    abc_result = _runsampler(abc_sampler, joint_prior, _particle_discrepancy(split_problem, particle_parallel),
        target, () -> loss(split_problem, reference_values); nparticles = n_particles,
        generations = n_generations, HPC = _threads_candidates(particle_parallel), stopping)
    posterior_draws = ParameterSamples(split_problem.selection.names, _abc_particles_matrix(abc_result.P);
        source = :abc)
    return InferenceResult(split_problem.model, posterior_draws, abc_sampler, abc_result, split_problem)
end

# Threaded ABC evaluates particles concurrently; each borrows a private copy so mesh
# solvers never share their in-place caches across threads.
_particle_discrepancy(estimation_problem::EstimationProblem, ::EnsembleSerial) =
    candidate_values -> loss(estimation_problem, collect(candidate_values))
_particle_discrepancy(estimation_problem::EstimationProblem, unsupported::SciMLBase.BasicEnsembleAlgorithm) =
    _unsupported_parallel(unsupported)
function _particle_discrepancy(estimation_problem::EstimationProblem, ::EnsembleThreads)
    private_problems = _private_copy_pool(estimation_problem, Threads.nthreads())
    return candidate_values -> _with_private_copy(private_problems) do private_problem
        loss(private_problem, collect(candidate_values))
    end
end

"""Sample a supplied joint distribution in the named selection order; correlations are retained."""
function parameter_samples(parameter_names, joint_distribution::Distributions.Distribution;
        n_samples::Int = 2048, rng = Random.default_rng())
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    joint_draws = rand(rng, joint_distribution, n_samples)
    joint_draws isa AbstractVector && (joint_draws = reshape(joint_draws, 1, :))
    return ParameterSamples(parameter_names, joint_draws; source = :distribution)
end

"""Build named joint samples from bootstrap fits with identical selections."""
function parameter_samples(bootstrap_results::AbstractVector{<:FitResult})
    isempty(bootstrap_results) && throw(ArgumentError("Supply at least one bootstrap fit."))
    selected_names = first(bootstrap_results).estimation_problem.selection.names
    all(bootstrap_result -> bootstrap_result.estimation_problem.selection.names == selected_names, bootstrap_results) ||
        throw(ArgumentError("Bootstrap fits must select the same named parameters in the same order."))
    sample_matrix = hcat([last.(_named_parameter_leaves(bootstrap_result.parameters)) for bootstrap_result in bootstrap_results]...)
    return ParameterSamples(selected_names, sample_matrix; source = :bootstrap)
end

"""Predictions retain common joint sample identity and per-experiment diagnostics."""
struct PredictionResult{Samples, Ensembles}
    samples::Samples
    experiments::Vector{Int}
    ensembles::Ensembles
end

"""
    predict(result; saveat=nothing, observables=nothing, parallel=EnsembleThreads())
    predict(result, runs; saveat=nothing, observables=nothing, parallel=EnsembleThreads())

Propagate a fit or posterior through the same model, solver, mode and numerical
options as the `EstimationProblem` it solved. Without `runs`, the fitted
experiments are predicted at their observation times. `runs` may hold
experiments (conditions resolved from their time-zero observations) or numeric
`RunConditions`, which need `saveat`. A `FitResult` predicts its single fitted
draw. By default (`parallel = EnsembleThreads()`) every (run, draw) pair is
solved on threads, each task on a private copy of the prepared templates;
`EnsembleSerial()` gives identical results on one thread.
"""
function predict(fit_outcome::Union{FitResult, InferenceResult},
        prediction_runs::AbstractVector = fit_outcome.estimation_problem.setup.experiments; kwargs...)
    estimation_problem = fit_outcome.estimation_problem
    fitted_solves = estimation_problem.setup.solves
    # Scoring settings (observable order, weights, projections) belong to the fit, not its predictions.
    return predict(estimation_problem.model, prediction_runs, _result_samples(fit_outcome);
        solver = estimation_problem.solver, mode = fitted_solves.mode,
        problem_options = estimation_problem.problem_options, alg = fitted_solves.requested_algorithm,
        solve_options = fitted_solves.solve_options, callback_factory = fitted_solves.callback_factory,
        kwargs...)
end

_result_samples(inference_result::InferenceResult) = inference_result.samples
_result_samples(fit_result::FitResult) = ParameterSamples(fit_result.estimation_problem.selection.names,
    reshape(last.(_named_parameter_leaves(fit_result.parameters)), :, 1); source = :fit)

"""
    predict(model, runs, samples; solver, mode=TransientMode(), problem_options=(;),
            saveat=nothing, observables=nothing, alg=nothing, solve_options=(;),
            callback_factory=nothing, parallel=EnsembleThreads())

Propagate user-supplied joint kinetic draws. The solver is required so that a
prediction never silently changes numerical method. `alg`, `solve_options` and
`callback_factory` are the ODE solve settings accepted by `EstimationProblem`.
Each draw is reused across all runs; noise is added only through
`measurement_samples`.
"""
function predict(crystal_model::CrystallisationModel, prediction_runs::AbstractVector,
        joint_samples::ParameterSamples; solver::AbstractSolver, mode::AbstractSimulationMode = TransientMode(),
        problem_options::NamedTuple = (;), saveat = nothing, observables = nothing,
        alg = nothing, solve_options::NamedTuple = (;), callback_factory = nothing,
        parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleThreads())
    isempty(prediction_runs) && throw(ArgumentError("Supply prediction experiments or run conditions."))
    sample_selection = _kinetic_selection(crystal_model, joint_samples.names)
    expanded_draws = reduce(hcat, [_expanded_parameters(sample_selection, view(joint_samples.values, :, draw_index))
        for draw_index in axes(joint_samples.values, 2)])
    prediction_specs = [_prediction_run(prediction_run, run_index, crystal_model, mode, saveat, problem_options)
        for (run_index, prediction_run) in enumerate(prediction_runs)]
    prepared_solves = _prepare_solves(first.(prediction_specs), getindex.(prediction_specs, 2), mode;
        solver, alg, solve_options, callback_factory)
    ensemble_predictions = run_ensemble(expanded_draws, prepared_solves; observables, parallel)
    return PredictionResult(joint_samples, last.(prediction_specs), ensemble_predictions)
end

"""`(problem, requested times, run id)` for one prediction run."""
function _prediction_run(experiment_run::CrystallisationExperiment, run_index::Int, crystal_model,
                         simulation_mode, saveat, problem_options)
    requested_times = saveat === nothing ? _observation_times(experiment_run) : collect(Float64, saveat)
    return (CrystallisationProblem(crystal_model, _run_conditions(simulation_mode, experiment_run),
            _observation_tspan(requested_times); problem_options...),
        requested_times, experiment_run.exp_id)
end

function _prediction_run(run_conditions::RunConditions, run_index::Int, crystal_model,
                         simulation_mode, saveat, problem_options)
    saveat === nothing && throw(ArgumentError("Predictions from run conditions need saveat."))
    requested_times = collect(Float64, saveat)
    return (CrystallisationProblem(crystal_model, run_conditions, _observation_tspan(requested_times);
            problem_options...),
        requested_times, run_index)
end

"""Summarise physical trajectory uncertainty, requiring explicit exclusion of failures."""
prediction_summary(predictions::PredictionResult, observable_name::Symbol; kwargs...) =
    [prediction_summary(ensemble_prediction, observable_name; kwargs...) for ensemble_prediction in predictions.ensembles]

"""
    measurement_samples(predictions, observable; noise, rng)

Add explicitly supplied independent additive noise draws to future measurements,
without modifying physical predictions. `noise` is a callable
`(experiment_id, observable, time) -> Distribution`, or a fixed distribution.
The caller owns interpolation and noise assumptions. Failed samples remain NaN.
"""
function measurement_samples(predictions::PredictionResult, observable_name::Symbol;
        noise, rng = Random.default_rng())
    return map(enumerate(predictions.ensembles)) do (experiment_index, ensemble_prediction)
        measurement_matrix = copy(observable_values(ensemble_prediction, observable_name))
        for time_index in eachindex(ensemble_prediction.t)
            noise_distribution = _noise_distribution(noise, predictions.experiments[experiment_index],
                observable_name, ensemble_prediction.t[time_index])
            noise_distribution isa Distributions.UnivariateDistribution || throw(ArgumentError("Supply a univariate additive noise distribution."))
            for draw_index in axes(measurement_matrix, 1)
                ensemble_prediction.success[draw_index] || continue
                measurement_matrix[draw_index, time_index] += rand(rng, noise_distribution)
            end
        end
        measurement_matrix
    end
end

# A fixed distribution applies everywhere; a callable chooses one per run, observable and time.
_noise_distribution(noise::Distributions.UnivariateDistribution, run_id, observable_name, saved_time) = noise
_noise_distribution(noise, run_id, observable_name, saved_time) = noise(run_id, observable_name, saved_time)
