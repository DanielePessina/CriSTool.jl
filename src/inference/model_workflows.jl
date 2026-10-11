import Distributions: fit

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

"""
    OptimisationSpec(; bounds, algorithm, loss=GaussianNLL(), adtype=AutoForwardDiff(), searchoptions=(;))

Select kinetic parameters by named `(lower, upper)` bounds and choose how they
are optimised. `algorithm` is a Metaheuristics algorithm or an Optimization.jl
solver; `adtype` and `searchoptions` apply to Optimization.jl solvers only.
"""
Base.@kwdef struct OptimisationSpec{Bounds <: NamedTuple, Algorithm, Loss, AD, SearchOptions <: NamedTuple}
    bounds::Bounds
    algorithm::Algorithm
    loss::Loss = GaussianNLL()
    adtype::AD = AutoForwardDiff()
    searchoptions::SearchOptions = (;)
end

"""
    BayesianSpec(; priors, sampler=nothing, n_samples=1000, n_chains=1, loss=GaussianNLL())

Select kinetic parameters by named univariate priors and sample their posterior
with a Turing sampler (`nothing` means `Turing.NUTS()`). Fitting requires
`using Turing`, which loads the `CriSToolTuringExt` extension. The likelihood uses supplied measurement variances; noise
is never fitted.
"""
Base.@kwdef struct BayesianSpec{Priors <: NamedTuple, Sampler, Loss}
    priors::Priors
    sampler::Sampler = nothing
    n_samples::Int = 1000
    n_chains::Int = 1
    loss::Loss = GaussianNLL()
end

"""
    ABCSpec(; priors, target, sampler=ABCDESampler(), n_particles=1024,
            n_generations=128, early_stop=false, loss=MeanAbsoluteError())

Select kinetic parameters by named priors and sample them by approximate
Bayesian computation under an explicit finite discrepancy threshold `target`.
"""
Base.@kwdef struct ABCSpec{Priors <: NamedTuple, Target <: Real, Sampler, Loss}
    priors::Priors
    target::Target
    sampler::Sampler = ABCDESampler()
    n_particles::Int = 1024
    n_generations::Int = 128
    early_stop::Bool = false
    loss::Loss = MeanAbsoluteError()
end

_selected_tree(fit_specification::OptimisationSpec) = fit_specification.bounds
_selected_tree(fit_specification::Union{BayesianSpec, ABCSpec}) = fit_specification.priors

function _validate_specification(fit_specification::OptimisationSpec)
    for bound_pair in last.(_named_parameter_leaves(fit_specification.bounds))
        bound_pair isa Tuple && length(bound_pair) == 2 &&
            all(bound_value -> bound_value isa Real && isfinite(bound_value), bound_pair) &&
            bound_pair[1] < bound_pair[2] || throw(ArgumentError("Each bound must be a finite (lower, upper) pair."))
    end
end
function _validate_specification(fit_specification::Union{BayesianSpec, ABCSpec})
    all(prior_value -> prior_value isa Distributions.UnivariateDistribution,
        last.(_named_parameter_leaves(fit_specification.priors))) ||
        throw(ArgumentError("Supply a univariate prior per selected kinetic parameter."))
    _validate_sampling_counts(fit_specification)
end
function _validate_sampling_counts(fit_specification::BayesianSpec)
    is_likelihood(fit_specification.loss) ||
        throw(ArgumentError("BayesianSpec requires a likelihood loss, not an error discrepancy."))
    fit_specification.n_samples > 0 && fit_specification.n_chains > 0 ||
        throw(ArgumentError("Sample and chain counts must be positive."))
end
function _validate_sampling_counts(fit_specification::ABCSpec)
    isfinite(fit_specification.target) || throw(ArgumentError("ABC target must be explicit and finite."))
    fit_specification.n_particles > 0 && fit_specification.n_generations > 0 ||
        throw(ArgumentError("Particle and generation counts must be positive."))
end

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

_model_experiment_problem(crystal_model, experiment_run, selected_solver, simulation_mode; problem_options...) =
    CrystallisationProblem(crystal_model, _run_conditions(simulation_mode, experiment_run);
        solver = selected_solver, problem_options...)

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

"""Prepared experiment solves, the resolved parameter selection, and how they were configured."""
struct PreparedFit{Model, Specification, Selection, Setup <: LossSetup, Solver <: AbstractSolver,
                   ProblemOptions <: NamedTuple}
    model::Model
    specification::Specification
    selection::Selection
    setup::Setup
    solver::Solver
    problem_options::ProblemOptions
end

"""
    prepare_fit(model, experiments, specification; solver=MoM(), mode=TransientMode(),
                problem_options=(;), observable_order=nothing, observable_projections=(;),
                algorithm=nothing, solve_options=(;), callback_factory=nothing,
                parallel=EnsembleSerial())

Resolve each experiment's conditions, validate noise, bounds or priors, and build
reusable solve templates. Time-zero observations consumed by `FromObservation`
conditions are not scored. `mode = SteadyMode(...)` relaxes each run to its
MSMPR steady state and needs numeric conditions to relax from. `algorithm`,
`solve_options` and `callback_factory` configure the ODE solves. `parallel =
EnsembleThreads()` solves the experiments of every loss evaluation (and so every
ForwardDiff gradient) on separate threads; scores are reduced in experiment order,
so the result is bitwise identical to `EnsembleSerial()`, the default.
"""
function prepare_fit(crystal_model::CrystallisationModel, measured_runs::AbstractVector{<:CrystallisationExperiment},
        fit_specification::Union{OptimisationSpec, BayesianSpec, ABCSpec};
        solver::AbstractSolver = MoM(), mode::AbstractSimulationMode = TransientMode(),
        problem_options::NamedTuple = (;), observable_order = nothing, observable_projections::NamedTuple = (;),
        algorithm = nothing, solve_options::NamedTuple = (;), callback_factory = nothing,
        parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleSerial())
    isempty(measured_runs) && throw(ArgumentError("Supply at least one experiment."))
    kinetic_selection = _kinetic_selection(crystal_model, _selected_tree(fit_specification))
    _validate_specification(fit_specification)
    physical_runs = [_model_experiment_problem(crystal_model, experiment_run, solver, mode; problem_options...)
        for experiment_run in measured_runs]
    prepared_setup = prepare_loss(physical_runs, measured_runs; mode, loss_function = fit_specification.loss,
        observable_order, observable_projections, algorithm, solve_options, callback_factory, parallel)
    _validate_likelihood_noise(fit_specification.loss, prepared_setup)
    return PreparedFit(crystal_model, fit_specification, kinetic_selection, prepared_setup, solver, problem_options)
end

"""Same preparation with the experiments of each loss evaluation solved under `parallel`."""
function _with_parallel(prepared_fit::PreparedFit, parallel::SciMLBase.BasicEnsembleAlgorithm)
    prepared_setup = prepared_fit.setup
    parallel_setup = LossSetup(prepared_setup.solves, prepared_setup.experiments, prepared_setup.observable_names,
        prepared_setup.included_observations, prepared_setup.observable_weights,
        prepared_setup.observable_projections, parallel)
    return PreparedFit(prepared_fit.model, prepared_fit.specification, prepared_fit.selection,
        parallel_setup, prepared_fit.solver, prepared_fit.problem_options)
end

"""Evaluate the selected-parameter objective; AD scalar types are preserved."""
loss(prepared_fit::PreparedFit, selected_values::AbstractVector) = loss(prepared_fit.specification.loss,
    prepared_fit.setup, _expanded_parameters(prepared_fit.selection, selected_values))

"""Named objectives of the specification's loss at selected parameter values."""
loss_terms(prepared_fit::PreparedFit, selected_values::AbstractVector) = loss_terms(prepared_fit.specification.loss,
    prepared_fit.setup, _expanded_parameters(prepared_fit.selection, selected_values))

"""A fitted model, named selected values, observable objectives, and native optimizer diagnostics."""
struct FitResult{Model, Parameters, Objectives, Backend, Preparation}
    model::Model
    parameters::Parameters
    objectives::Objectives
    backend_result::Backend
    preparation::Preparation
end

"""
    fit(prepared_fit; threaded=false, parallel=prepared_fit.setup.parallel)

Optimise the selected kinetic parameters with the specification's algorithm.
A scalar objective returns a `FitResult`; NSGA2 returns one `FitResult` per
Pareto candidate. `threaded` evaluates Metaheuristics populations in parallel,
each task on a private copy of the solve templates. `parallel = EnsembleThreads()`
solves the experiments of each loss evaluation on threads, which also speeds up
gradient-based algorithms; it takes effect for this call only.
"""
function fit(supplied_fit::PreparedFit{<:Any, <:OptimisationSpec}; threaded::Bool = false,
             parallel::SciMLBase.BasicEnsembleAlgorithm = supplied_fit.setup.parallel)
    prepared_fit = _with_parallel(supplied_fit, parallel)
    bound_pairs = last.(_named_parameter_leaves(prepared_fit.specification.bounds))
    lower_bounds, upper_bounds = first.(bound_pairs), last.(bound_pairs)
    starting_values = prepared_fit.selection.baseline[prepared_fit.selection.indices]
    all(lower_bounds .<= starting_values .<= upper_bounds) || throw(ArgumentError("Current selected values must lie within bounds."))
    return _optimise(prepared_fit.specification.algorithm, prepared_fit, lower_bounds, upper_bounds,
        starting_values; threaded)
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

"""Optimization.jl solvers: search from the model's current values; `threaded` does not apply."""
function _optimise(optimiser, prepared_fit, lower_bounds, upper_bounds, starting_values; threaded::Bool)
    fit_specification = prepared_fit.specification
    optimization_function = OptimizationBase.OptimizationFunction(
        (candidate_values, fit_context) -> loss(fit_context, candidate_values), fit_specification.adtype)
    optimization_problem = OptimizationBase.OptimizationProblem(optimization_function,
        starting_values, prepared_fit; lb = lower_bounds, ub = upper_bounds)
    optimization_solution = OptimizationBase.solve(optimization_problem, optimiser; fit_specification.searchoptions...)
    return _model_fit_result(prepared_fit, optimization_solution.u, optimization_solution)
end

"""Metaheuristics algorithms: population search inside the bounds on a private algorithm copy."""
function _optimise(supplied_algorithm::Metaheuristics.Algorithm, prepared_fit, lower_bounds, upper_bounds,
                   starting_values; threaded::Bool)
    private_algorithm = _configured_mh_algorithm(supplied_algorithm; parallel_evaluation = threaded)
    private_fit = _cristool_once_per_task(typeof(prepared_fit)) do
        deepcopy(prepared_fit)
    end
    candidate_objective = _MetaheuristicsObjective(private_algorithm.parameters, prepared_fit, private_fit)
    final_state = Metaheuristics.optimize(candidate_objective,
        Metaheuristics.boxconstraints(lower_bounds, upper_bounds), private_algorithm)
    return _metaheuristics_fit_results(private_algorithm.parameters, prepared_fit, final_state)
end

"""Metaheuristics objective: one candidate serially, or a population in parallel on per-task copies."""
struct _MetaheuristicsObjective{Parameters, Fit, PrivateFit}
    parameters::Parameters
    prepared_fit::Fit
    private_fit::PrivateFit
end

# Metaheuristics requires a method accepting `Any`; the argument shape then selects the evaluation.
(candidate_objective::_MetaheuristicsObjective)(candidate_values) =
    _evaluate_candidates(candidate_objective, candidate_values)

_evaluate_candidates(candidate_objective::_MetaheuristicsObjective, candidate_values::AbstractVector) =
    _candidate_score(candidate_objective.parameters, candidate_objective.prepared_fit, candidate_values)

function _evaluate_candidates(candidate_objective::_MetaheuristicsObjective, candidate_values::AbstractMatrix)
    candidate_scores = Vector{_candidate_score_type(candidate_objective.parameters)}(undef, size(candidate_values, 1))
    Threads.@threads for candidate_index in axes(candidate_values, 1)
        candidate_scores[candidate_index] = _candidate_score(candidate_objective.parameters,
            candidate_objective.private_fit(), collect(view(candidate_values, candidate_index, :)))
    end
    return _population_scores(candidate_objective.parameters, candidate_scores)
end

# Scalar algorithms minimise the total loss; NSGA2 minimises each named objective.
_candidate_score(::Metaheuristics.AbstractParameters, prepared_fit, candidate_values) =
    loss(prepared_fit, candidate_values)
_candidate_score(::Metaheuristics.NSGA2, prepared_fit, candidate_values) =
    (collect(Float64, values(loss_terms(prepared_fit, candidate_values))), [0.0], [0.0])

_candidate_score_type(::Metaheuristics.AbstractParameters) = Float64
_candidate_score_type(::Metaheuristics.NSGA2) = Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

_population_scores(::Metaheuristics.AbstractParameters, candidate_scores) = candidate_scores
_population_scores(::Metaheuristics.NSGA2, candidate_scores) =
    (permutedims(reduce(hcat, first.(candidate_scores))),
     zeros(length(candidate_scores), 1), zeros(length(candidate_scores), 1))

_metaheuristics_fit_results(::Metaheuristics.AbstractParameters, prepared_fit, final_state) =
    _model_fit_result(prepared_fit, Metaheuristics.minimizer(final_state), final_state)
function _metaheuristics_fit_results(::Metaheuristics.NSGA2, prepared_fit, final_state)
    pareto_values = Metaheuristics.positions(Metaheuristics.get_non_dominated_solutions(final_state.population))
    return [_model_fit_result(prepared_fit, collect(candidate_row), final_state) for candidate_row in eachrow(pareto_values)]
end

function _model_fit_result(prepared_fit, optimal_values, backend_result)
    fitted_model = _selected_model(prepared_fit.model, prepared_fit.selection, optimal_values)
    return FitResult(fitted_model, _selected_named_values(prepared_fit.selection, optimal_values),
        loss_terms(prepared_fit, optimal_values), backend_result, prepared_fit)
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

"""Posterior draws, the native chain or sampler result, and the preparation that produced them."""
struct InferenceResult{Model, Samples, Backend, Preparation}
    model::Model
    samples::Samples
    backend_result::Backend
    preparation::Preparation
end

"""
    fit(bayesian_prepared; rng=default_rng(), threaded=false,
        parallel=bayesian_prepared.setup.parallel, kwargs...)

Sample the posterior with the specification's sampler and counts. `threaded`
runs chains in parallel (`MCMCThreads`); `parallel = EnsembleThreads()` solves
the experiments inside each log-density evaluation on threads. Both share
Julia's thread pool, so use `parallel` for few chains and many experiments, and
`threaded` when chains are at least as many as threads. Other keywords go to
`Turing.sample` (e.g. `progress`). Requires `using Turing`.
"""
function fit(supplied_fit::PreparedFit{<:Any, <:BayesianSpec};
        rng = Random.default_rng(), threaded::Bool = false,
        parallel::SciMLBase.BasicEnsembleAlgorithm = supplied_fit.setup.parallel, kwargs...)
    _require_turing_extension()
    prepared_fit = _with_parallel(supplied_fit, parallel)
    posterior_chain = _nuts_posterior_chain(prepared_fit, rng, threaded; kwargs...)
    posterior_chain = rename_chain(posterior_chain, Symbol.(prepared_fit.selection.names))
    posterior_draws = ParameterSamples(prepared_fit.selection.names,
        chains_to_matrix(posterior_chain; params = Symbol.(prepared_fit.selection.names)); source = :nuts)
    return InferenceResult(prepared_fit.model, posterior_draws, posterior_chain, prepared_fit)
end

"""
    fit(abc_prepared; threaded=false, parallel=abc_prepared.setup.parallel)

Sample under the specification's discrepancy threshold. `threaded` evaluates
particles in parallel; `parallel = EnsembleThreads()` solves the experiments of
each discrepancy evaluation on threads.
"""
function fit(supplied_fit::PreparedFit{<:Any, <:ABCSpec}; threaded::Bool = false,
             parallel::SciMLBase.BasicEnsembleAlgorithm = supplied_fit.setup.parallel)
    prepared_fit = _with_parallel(supplied_fit, parallel)
    fit_specification = prepared_fit.specification
    joint_prior = product_distribution(last.(_named_parameter_leaves(fit_specification.priors)))
    reference_values = prepared_fit.selection.baseline[prepared_fit.selection.indices]
    abc_result, reached_target = _runsampler(fit_specification.sampler, joint_prior,
        candidate_values -> loss(prepared_fit, collect(candidate_values)), fit_specification.target,
        loss(prepared_fit, reference_values); nparticles = fit_specification.n_particles,
        generations = fit_specification.n_generations, HPC = threaded, earlystop = fit_specification.early_stop)
    sample_matrix = _abc_particles_matrix(abc_result.P)
    sample_matrix === nothing && throw(ArgumentError("ABC returned no posterior particles."))
    posterior_draws = ParameterSamples(prepared_fit.selection.names, sample_matrix; source = :abc)
    return InferenceResult(prepared_fit.model, posterior_draws, abc_result, prepared_fit)
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
    selected_names = first(bootstrap_results).preparation.selection.names
    all(bootstrap_result -> bootstrap_result.preparation.selection.names == selected_names, bootstrap_results) ||
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
    predict(result; saveat=nothing, observables=nothing, parallel=result.preparation.setup.parallel)
    predict(result, runs; saveat=nothing, observables=nothing, parallel=...)

Propagate a fit or posterior through the same model, solver, mode and numerical
options it was prepared with. Without `runs`, the fitted experiments are
predicted at their observation times. `runs` may hold experiments (conditions
resolved from their time-zero observations) or numeric `RunConditions`, which
need `saveat`. A `FitResult` predicts its single fitted draw. `parallel =
EnsembleThreads()` solves every (run, draw) pair on threads, each task on a
private copy of the prepared templates; results equal `EnsembleSerial()`.
"""
function predict(fit_outcome::Union{FitResult, InferenceResult},
        prediction_runs::AbstractVector = fit_outcome.preparation.setup.experiments; kwargs...)
    preparation = fit_outcome.preparation
    fitted_solves = preparation.setup.solves
    # Scoring settings (observable order, weights, projections) belong to the fit, not its predictions.
    return predict(preparation.model, prediction_runs, _result_samples(fit_outcome);
        solver = preparation.solver, mode = fitted_solves.mode, problem_options = preparation.problem_options,
        algorithm = fitted_solves.requested_algorithm, solve_options = fitted_solves.solve_options,
        callback_factory = fitted_solves.callback_factory,
        (; parallel = preparation.setup.parallel, kwargs...)...)
end

_result_samples(inference_result::InferenceResult) = inference_result.samples
_result_samples(fit_result::FitResult) = ParameterSamples(fit_result.preparation.selection.names,
    reshape(last.(_named_parameter_leaves(fit_result.parameters)), :, 1); source = :fit)

"""
    predict(model, runs, samples; solver, mode=TransientMode(), problem_options=(;),
            saveat=nothing, observables=nothing, algorithm=nothing, solve_options=(;),
            callback_factory=nothing, parallel=EnsembleSerial())

Propagate user-supplied joint kinetic draws. The solver is required so that a
prediction never silently changes numerical method. `algorithm`,
`solve_options` and `callback_factory` are the solve settings accepted by
`prepare_fit`. Each draw is reused across all runs; noise is added only through
`measurement_samples`.
"""
function predict(crystal_model::CrystallisationModel, prediction_runs::AbstractVector,
        joint_samples::ParameterSamples; solver::AbstractSolver, mode::AbstractSimulationMode = TransientMode(),
        problem_options::NamedTuple = (;), saveat = nothing, observables = nothing,
        algorithm = nothing, solve_options::NamedTuple = (;), callback_factory = nothing,
        parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleSerial())
    isempty(prediction_runs) && throw(ArgumentError("Supply prediction experiments or run conditions."))
    sample_selection = _kinetic_selection(crystal_model, joint_samples.names)
    expanded_draws = reduce(hcat, [_expanded_parameters(sample_selection, view(joint_samples.values, :, draw_index))
        for draw_index in axes(joint_samples.values, 2)])
    prediction_specs = [_prediction_run(prediction_run, run_index, crystal_model, solver, mode, saveat, problem_options)
        for (run_index, prediction_run) in enumerate(prediction_runs)]
    prepared_solves = _prepare_solves(first.(prediction_specs), getindex.(prediction_specs, 2), mode;
        algorithm, solve_options, callback_factory)
    ensemble_predictions = run_ensemble(expanded_draws, prepared_solves; observables, parallel)
    return PredictionResult(joint_samples, last.(prediction_specs), ensemble_predictions)
end

"""`(problem, requested times, run id)` for one prediction run."""
function _prediction_run(experiment_run::CrystallisationExperiment, run_index::Int, crystal_model,
                         solver, simulation_mode, saveat, problem_options)
    requested_times = saveat === nothing ? _observation_times(experiment_run) : collect(Float64, saveat)
    return (_model_experiment_problem(crystal_model, experiment_run, solver, simulation_mode; problem_options...),
        requested_times, experiment_run.exp_id)
end

function _prediction_run(run_conditions::RunConditions, run_index::Int, crystal_model,
                         solver, simulation_mode, saveat, problem_options)
    saveat === nothing && throw(ArgumentError("Predictions from run conditions need saveat."))
    return (CrystallisationProblem(crystal_model, run_conditions; solver, problem_options...),
        collect(Float64, saveat), run_index)
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
        for time_index in eachindex(ensemble_prediction.time)
            noise_distribution = _noise_distribution(noise, predictions.experiments[experiment_index],
                observable_name, ensemble_prediction.time[time_index])
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
