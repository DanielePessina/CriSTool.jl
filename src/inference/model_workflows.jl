import Distributions: fit
import Turing: predict

"""Declare whether a custom loss represents a supplied-noise negative log likelihood."""
is_likelihood(::AbstractPELossFunction) = false
is_likelihood(::logMLE) = true

"""Named objective accounting. Custom losses default to a total and may extend this seam."""
loss_terms(loss_function::AbstractPELossFunction, prepared_setup::LossSetup, parameter_values) =
    (; total = loss(loss_function, prepared_setup, parameter_values))
function loss_terms(loss_function::Union{logMLE, mae}, prepared_setup::LossSetup, parameter_values)
    return NamedTuple{Tuple(prepared_setup.observable_names)}(
        Tuple(_loss_objectives(loss_function, prepared_setup, parameter_values)))
end

"""Named bounds select the kinetic parameters varied by an optimisation study."""
Base.@kwdef struct OptimisationSpec{Bounds <: NamedTuple, Loss}
    bounds::Bounds
    loss::Loss = logMLE()
end

"""Named priors select Bayesian kinetic parameters. Noise remains supplied and fixed."""
Base.@kwdef struct BayesianSpec{Priors <: NamedTuple, Loss}
    priors::Priors
    loss::Loss = logMLE()
end

"""Named priors, discrepancy and an explicit finite ABC acceptance threshold."""
Base.@kwdef struct ABCSpec{Priors <: NamedTuple, Loss, Target <: Real}
    priors::Priors
    target::Target
    loss::Loss = mae()
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

function _kinetic_selection(crystal_model::CrystallisationModel, selected_tree::NamedTuple)
    _validate_material_model(crystal_model)
    model_values = ComponentArray(; _model_named_values(crystal_model)...)
    all_names = ComponentArrays.labels(model_values)
    selected_leaves = _named_parameter_leaves(selected_tree)
    isempty(selected_leaves) && throw(ArgumentError("Select at least one kinetic parameter."))
    selected_names = first.(selected_leaves)
    selected_indices = map(selected_names) do selected_name
        selected_index = findfirst(==(selected_name), all_names)
        selected_index === nothing && throw(ArgumentError("Unknown kinetic parameter $selected_name."))
        selected_index
    end
    return KineticSelection(selected_names, selected_indices, first(getaxes(model_values)), collect(model_values))
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
    return CrystallisationModel(; system = crystal_model.system, reconstructed_kinetics...)
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

function _initial_observation(experiment_run::CrystallisationExperiment, observable_name::Symbol)
    hasproperty(experiment_run.observables, observable_name) ||
        throw(ArgumentError("Experiment $(experiment_run.exp_id) lacks initial observable :$observable_name."))
    initial_series = getproperty(experiment_run.observables, observable_name)
    initial_index = findfirst(iszero, initial_series.time)
    initial_index === nothing && throw(ArgumentError(
        "Experiment $(experiment_run.exp_id): :$observable_name must have a time-zero initial observation."))
    return initial_series.mean[initial_index]
end

function _mapped_initial_value(experiment_run, declared_value)
    return declared_value isa Symbol ? _initial_observation(experiment_run, declared_value) : declared_value
end

_experiment_temperature(temperature_setting::Real) = ConstantTemperature(temperature_setting)
_experiment_temperature(temperature_setting::AbstractTemperature) = temperature_setting

function _experiment_seed(experiment_run::CrystallisationExperiment)
    seed_mapping = hasproperty(experiment_run.initial_from, :crystals) ?
        experiment_run.initial_from.crystals : nothing
    if seed_mapping === nothing
        experiment_run.seed_shape === nothing || throw(ArgumentError("Seed shape requires an initial crystals mapping."))
        return experiment_run.initial_crystals === nothing ? nothing :
            _checked_seed_description(experiment_run.initial_crystals)
    end
    experiment_run.initial_crystals === nothing || throw(ArgumentError("Supply initial crystal mappings or explicit seeds, not both."))
    seed_mapping isa NamedTuple && hasproperty(seed_mapping, :mass_concentration) ||
        throw(ArgumentError("Seed mapping must supply mass_concentration."))
    seed_mass = _mapped_initial_value(experiment_run, seed_mapping.mass_concentration)
    seed_shape = experiment_run.seed_shape
    seed_shape isa NamedTuple || throw(ArgumentError("Seed mapping requires explicit seed_shape."))
    if hasproperty(seed_shape, :distribution)
        all(shape_name -> shape_name in (:distribution, :weighting, :mass_loss_tolerance), keys(seed_shape)) ||
            throw(ArgumentError("Distribution seeds cannot also declare a characteristic shape or width."))
        keys(seed_mapping) == (:mass_concentration,) || throw(ArgumentError("Distribution seed mapping needs mass only."))
        return DistributionInitialCrystals(seed_shape.distribution;
            mass_concentration = seed_mass,
            weighting = get(seed_shape, :weighting, :number),
            mass_loss_tolerance = get(seed_shape, :mass_loss_tolerance, 1e-4))
    end
    hasproperty(seed_mapping, :d43) || throw(ArgumentError("Characteristic seed mapping must supply d43."))
    Set(keys(seed_mapping)) == Set((:mass_concentration, :d43)) ||
        throw(ArgumentError("Characteristic seed mappings accept mass_concentration and d43 only."))
    seed_d43 = _mapped_initial_value(experiment_run, seed_mapping.d43)
    seed_family = get(seed_shape, :family, nothing)
    if seed_family === :lognormal && hasproperty(seed_shape, :geometric_std)
        Set(keys(seed_shape)) == Set((:family, :geometric_std)) ||
            throw(ArgumentError("Lognormal shape requires only family and geometric_std."))
        return _checked_seed_description(LogNormalInitialCrystals(; mass_concentration = seed_mass,
            d43 = seed_d43, geometric_std = _mapped_initial_value(experiment_run, seed_shape.geometric_std)))
    elseif seed_family === :gaussian && hasproperty(seed_shape, :standard_deviation)
        Set(keys(seed_shape)) == Set((:family, :standard_deviation)) ||
            throw(ArgumentError("Gaussian shape requires only family and standard_deviation."))
        return _checked_seed_description(GaussianInitialCrystals(; mass_concentration = seed_mass,
            d43 = seed_d43, standard_deviation = _mapped_initial_value(experiment_run, seed_shape.standard_deviation)))
    end
    throw(ArgumentError("Seed shape requires :lognormal with geometric_std or :gaussian with standard_deviation."))
end

function _model_experiment_problem(crystal_model, experiment_run, selected_solver; mode = :transient, kwargs...)
    reserved_conditions = intersect(keys(kwargs), (:initial_conditions, :initial_state,
        :initial_crystals, :temperature, :operation, :solver))
    isempty(reserved_conditions) || throw(ArgumentError(
        "Experiment initial observations and declared conditions cannot be overridden through problem_options: $reserved_conditions."))
    initial_seed = _experiment_seed(experiment_run)
    solvent_initial = if mode === :steady
        experiment_run.relaxation_initial isa NamedTuple || throw(ArgumentError(
            "Steady fitting requires explicit relaxation_initial, independent of steady observations."))
        experiment_run.relaxation_initial
    else
        solvent_mapping = Base.structdiff(experiment_run.initial_from, (; crystals = nothing))
        hasproperty(solvent_mapping, :concentration) || throw(ArgumentError("Initial concentration mapping is required."))
        NamedTuple{keys(solvent_mapping)}(map(keys(solvent_mapping)) do solvent_name
            source_name = getproperty(solvent_mapping, solvent_name)
            source_name isa Symbol || throw(ArgumentError("Solvent initial mappings must name observables."))
            _initial_observation(experiment_run, source_name)
        end)
    end
    return CrystallisationProblem(crystal_model; initial_conditions = solvent_initial,
        temperature = _experiment_temperature(experiment_run.temperature),
        operation = experiment_run.operation, initial_crystals = initial_seed, solver = selected_solver, kwargs...)
end

function _initial_source_names(experiment_run)
    source_names = Symbol[]
    for mapping_leaf in _named_parameter_leaves(experiment_run.initial_from)
        last(mapping_leaf) isa Symbol && push!(source_names, last(mapping_leaf))
    end
    if experiment_run.seed_shape isa NamedTuple
        for width_name in (:geometric_std, :standard_deviation)
            width_value = get(experiment_run.seed_shape, width_name, nothing)
            width_value isa Symbol && push!(source_names, width_value)
        end
    end
    return unique(source_names)
end

function _strict_likelihood_variances(loss_function, prepared_setup)
    loss_function isa logMLE || return
    _validate_loss_weights(loss_function, prepared_setup)
    loss_function.variance_model isa MeasuredVariance ||
        throw(ArgumentError("Likelihood studies require supplied measured variances; relative noise is not inferred."))
    for (experiment_index, experiment_run) in enumerate(prepared_setup.experiments)
        for (objective_index, observable_name) in enumerate(prepared_setup.observable_names)
            iszero(_observable_weight(loss_function, objective_index, observable_name)) && continue
            hasproperty(experiment_run.observables, observable_name) || continue
            target_indices = prepared_setup.included_observations[experiment_index][observable_name]
            isempty(target_indices) && continue
            measured_series = getproperty(experiment_run.observables, observable_name)
            measured_series.variance !== nothing && all(measured_series.variance[target_indices] .> 0) ||
                throw(ArgumentError("Experiment $(experiment_run.exp_id), :$observable_name requires supplied positive variances."))
        end
    end
end

"""Prepared experiment solves plus a resolved selection of kinetic parameters."""
struct PreparedFit{Model, Specification, Selection}
    model::Model
    specification::Specification
    selection::Selection
    setup::LossSetup
end

"""
    prepare_fit(model, experiments, specification; solver=MoM(), mode=:transient, kwargs...)

Resolve initial measurements, validate noise/bounds/priors, and build reusable
solve templates. In steady mode provide explicit `relaxation_initial` per run.
"""
function prepare_fit(crystal_model::CrystallisationModel, measured_runs::AbstractVector{<:CrystallisationExperiment},
        fit_specification::Union{OptimisationSpec, BayesianSpec, ABCSpec};
        solver::AbstractSolver = MoM(), mode::Symbol = :transient,
        problem_options::NamedTuple = (;), kwargs...)
    mode in (:transient, :steady) || throw(ArgumentError("mode must be :transient or :steady."))
    isempty(intersect(keys(kwargs), (:initial_time, :exclude_initial_concentration))) ||
        throw(ArgumentError("Model fitting owns the time-zero initial-observation and scoring policy."))
    isempty(measured_runs) && throw(ArgumentError("Supply at least one experiment."))
    selected_tree = fit_specification isa OptimisationSpec ? fit_specification.bounds : fit_specification.priors
    kinetic_selection = _kinetic_selection(crystal_model, selected_tree)
    if fit_specification isa OptimisationSpec
        for bound_pair in last.(_named_parameter_leaves(selected_tree))
            bound_pair isa Tuple && length(bound_pair) == 2 &&
                all(bound_value -> bound_value isa Real && isfinite(bound_value), bound_pair) &&
                bound_pair[1] < bound_pair[2] || throw(ArgumentError("Each bound must be a finite (lower, upper) pair."))
        end
    else
        all(prior_value -> prior_value isa Distributions.UnivariateDistribution,
            last.(_named_parameter_leaves(selected_tree))) || throw(ArgumentError("Supply a univariate prior per selected kinetic parameter."))
    end
    fit_specification isa BayesianSpec && !is_likelihood(fit_specification.loss) &&
        throw(ArgumentError("BayesianSpec requires a likelihood loss, not an error discrepancy."))
    fit_specification isa ABCSpec && !isfinite(fit_specification.target) &&
        throw(ArgumentError("ABC target must be explicit and finite."))
    physical_runs = [_model_experiment_problem(crystal_model, experiment_run, solver; mode, problem_options...)
        for experiment_run in measured_runs]
    prepared_setup = prepare_loss(physical_runs, collect(measured_runs); mode,
        exclude_initial_concentration = false, kwargs...)
    if mode === :transient
        for (experiment_index, experiment_run) in enumerate(measured_runs)
            for initial_name in _initial_source_names(experiment_run)
                initial_series = getproperty(experiment_run.observables, initial_name)
                filter!(sample_index -> !iszero(initial_series.time[sample_index]),
                    prepared_setup.included_observations[experiment_index][initial_name])
            end
        end
    end
    _strict_likelihood_variances(fit_specification.loss, prepared_setup)
    return PreparedFit(crystal_model, fit_specification, kinetic_selection, prepared_setup)
end

"""Evaluate the selected-parameter objective; AD scalar types are preserved."""
loss(prepared_fit::PreparedFit, selected_values::AbstractVector) = loss(prepared_fit.specification.loss,
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
    fit(prepared_fit; algorithm, adtype=AutoForwardDiff(), searchoptions=(;))

Optimise selected kinetic parameters. A scalar objective returns a `FitResult`;
NSGA2 returns one fitted model per Pareto candidate. Native results are retained.
"""
function fit(prepared_fit::PreparedFit{<:Any, <:OptimisationSpec}; algorithm,
        adtype = AutoForwardDiff(), searchoptions::NamedTuple = (;), kwargs...)
    bound_pairs = last.(_named_parameter_leaves(prepared_fit.specification.bounds))
    lower_bounds, upper_bounds = first.(bound_pairs), last.(bound_pairs)
    starting_values = prepared_fit.selection.baseline[prepared_fit.selection.indices]
    all(lower_bounds .<= starting_values .<= upper_bounds) || throw(ArgumentError("Current selected values must lie within bounds."))
    backend_result = if algorithm isa Metaheuristics.Algorithm
        private_algorithm = _configured_mh_algorithm(algorithm; kwargs...)
        candidate_objective = candidate_values -> _selected_optimizer_objective(
            prepared_fit, candidate_values, private_algorithm.parameters isa Metaheuristics.NSGA2)
        Metaheuristics.optimize(candidate_objective, Metaheuristics.boxconstraints(lower_bounds, upper_bounds), private_algorithm)
    else
        isempty(kwargs) || throw(ArgumentError("Use searchoptions for Optimization.jl settings."))
        optimization_function = OptimizationBase.OptimizationFunction(
            (candidate_values, fit_context) -> loss(fit_context, candidate_values), adtype)
        optimization_problem = OptimizationBase.OptimizationProblem(optimization_function,
            starting_values, prepared_fit; lb = lower_bounds, ub = upper_bounds)
        OptimizationBase.solve(optimization_problem, algorithm; searchoptions...)
    end
    optimal_values = if algorithm isa Metaheuristics.Algorithm && algorithm.parameters isa Metaheuristics.NSGA2
        Metaheuristics.positions(Metaheuristics.get_non_dominated_solutions(backend_result.population))
    elseif backend_result isa Metaheuristics.State
        Metaheuristics.minimizer(backend_result)
    else
        backend_result.u
    end
    if optimal_values isa AbstractMatrix
        return [_model_fit_result(prepared_fit, collect(candidate_row), backend_result) for candidate_row in eachrow(optimal_values)]
    end
    return _model_fit_result(prepared_fit, optimal_values, backend_result)
end

function _selected_optimizer_objective(prepared_fit, candidate_values::AbstractVector, multiobjective::Bool)
    if multiobjective
        objective_values = collect(values(loss_terms(prepared_fit.specification.loss, prepared_fit.setup,
            _expanded_parameters(prepared_fit.selection, candidate_values))))
        return (objective_values, [0.0], [0.0])
    end
    return loss(prepared_fit, candidate_values)
end
function _selected_optimizer_objective(prepared_fit, candidate_values::AbstractMatrix, multiobjective::Bool)
    candidate_outputs = Vector{Any}(undef, size(candidate_values, 1))
    Threads.@threads for candidate_index in axes(candidate_values, 1)
        candidate_outputs[candidate_index] = _selected_optimizer_objective(deepcopy(prepared_fit),
            collect(view(candidate_values, candidate_index, :)), multiobjective)
    end
    return multiobjective ? (permutedims(hcat(first.(candidate_outputs)...)),
        zeros(size(candidate_values, 1), 1), zeros(size(candidate_values, 1), 1)) : candidate_outputs
end

function _model_fit_result(prepared_fit, optimal_values, backend_result)
    fitted_model = _selected_model(prepared_fit.model, prepared_fit.selection, optimal_values)
    named_objectives = loss_terms(prepared_fit.specification.loss, prepared_fit.setup,
        _expanded_parameters(prepared_fit.selection, optimal_values))
    return FitResult(fitted_model, _selected_named_values(prepared_fit.selection, optimal_values),
        named_objectives, backend_result, prepared_fit)
end

struct SelectedLikelihood{Loss, Selection} <: AbstractPELossFunction
    lossfunction::Loss
    selection::Selection
end
loss(selected_likelihood::SelectedLikelihood, private_setup::LossSetup, selected_values) =
    loss(selected_likelihood.lossfunction, private_setup,
        _expanded_parameters(selected_likelihood.selection, selected_values))

"""Build a Turing model with explicit priors over selected kinetic parameters."""
function nuts_model(prepared_fit::PreparedFit{<:Any, <:BayesianSpec})
    prior_values = last.(_named_parameter_leaves(prepared_fit.specification.priors))
    private_preparation = _cristool_once_per_task(LossSetup) do
        deepcopy(prepared_fit.setup)
    end
    return _cristool_nuts_loss_model(prior_values,
        SelectedLikelihood(prepared_fit.specification.loss, prepared_fit.selection), private_preparation)
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

"""Posterior draws and their native chain/sampler result, tied to a reference model."""
struct InferenceResult{Model, Samples, Backend}
    model::Model
    samples::Samples
    backend_result::Backend
end
"""Return named joint draws from an inference result."""
parameter_samples(inference_result::InferenceResult) = inference_result.samples

"""`fit(bayesian_prepared; sampler, n_samples, n_chains, rng)` returns named joint posterior draws and the native chain."""
function fit(prepared_fit::PreparedFit{<:Any, <:BayesianSpec};
        sampler = Turing.NUTS(), n_samples::Int = 1000, n_chains::Int = 1,
        rng = Random.default_rng(), kwargs...)
    n_samples > 0 && n_chains > 0 || throw(ArgumentError("Sample and chain counts must be positive."))
    posterior_chain = Turing.sample(rng, nuts_model(prepared_fit), sampler,
        Turing.MCMCThreads(), n_samples, n_chains; kwargs...)
    posterior_chain = rename_chain(posterior_chain, Symbol.(prepared_fit.selection.names))
    posterior_draws = ParameterSamples(prepared_fit.selection.names,
        chains_to_matrix(posterior_chain; params = Symbol.(prepared_fit.selection.names)); source = :nuts)
    return InferenceResult(prepared_fit.model, posterior_draws, posterior_chain)
end

"""`fit(abc_prepared; sampler, nparticles, generations)` samples under an explicit discrepancy threshold."""
function fit(prepared_fit::PreparedFit{<:Any, <:ABCSpec}; sampler::AbstractABCSampler = ABCDESampler(),
        nparticles::Int = 1024, generations::Int = 128, HPC::Bool = false, earlystop::Bool = false)
    prior_values = last.(_named_parameter_leaves(prepared_fit.specification.priors))
    joint_prior = product_distribution(prior_values)
    reference_values = prepared_fit.selection.baseline[prepared_fit.selection.indices]
    abc_result, reached_target = _runsampler(sampler, joint_prior,
        candidate_values -> loss(prepared_fit, collect(candidate_values)), prepared_fit.specification.target,
        loss(prepared_fit, reference_values); nparticles, generations, HPC, earlystop)
    sample_matrix = _abc_particles_matrix(abc_result.P)
    sample_matrix === nothing && throw(ArgumentError("ABC returned no posterior particles."))
    posterior_draws = ParameterSamples(prepared_fit.selection.names, sample_matrix; source = :abc)
    return InferenceResult(prepared_fit.model, posterior_draws, abc_result)
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

function _selection_for_samples(crystal_model, joint_samples::ParameterSamples)
    all_model_values = ComponentArray(; _model_named_values(crystal_model)...)
    all_names = ComponentArrays.labels(all_model_values)
    sample_indices = map(joint_samples.names) do parameter_name
        parameter_index = findfirst(==(parameter_name), all_names)
        parameter_index === nothing && throw(ArgumentError("Unknown kinetic sample path $parameter_name."))
        parameter_index
    end
    return KineticSelection(joint_samples.names, sample_indices, first(getaxes(all_model_values)), collect(all_model_values))
end

function _same_physical_configuration(first_configuration, second_configuration)
    typeof(first_configuration) == typeof(second_configuration) || return false
    return all(isequal(getfield(first_configuration, field_name), getfield(second_configuration, field_name))
        for field_name in fieldnames(typeof(first_configuration)))
end

"""
    predict(model, experiments_or_problems, samples; saveat=nothing, solver=MoM(), observables=nothing)

Propagate joint kinetic draws into physical trajectories. Experiments derive
time-zero initial conditions; configured problems preserve explicit conditions.
Each draw is reused across all runs. Noise is added only through `measurement_samples`.
"""
function predict(crystal_model::CrystallisationModel, prediction_runs::AbstractVector,
        joint_samples::ParameterSamples; saveat = nothing, solver::AbstractSolver = MoM(),
        observables = nothing, mode::Symbol = :transient, problem_options::NamedTuple = (;), kwargs...)
    _validate_material_model(crystal_model)
    mode in (:transient, :steady) || throw(ArgumentError("mode must be :transient or :steady."))
    if any(prediction_run -> prediction_run isa CrystallisationExperiment, prediction_runs)
        get(kwargs, :initial_time, 0.0) == 0 || throw(ArgumentError("Experiment predictions start from time-zero initial observations."))
    end
    isempty(prediction_runs) && throw(ArgumentError("Supply prediction experiments or problems."))
    selected_paths = _selection_for_samples(crystal_model, joint_samples)
    expanded_draws = hcat([_expanded_parameters(selected_paths, view(joint_samples.values, :, draw_index))
        for draw_index in axes(joint_samples.values, 2)]...)
    physical_runs = [prediction_run isa CrystallisationExperiment ?
        _model_experiment_problem(crystal_model, prediction_run, solver; mode, problem_options...) : prediction_run
        for prediction_run in prediction_runs]
    all(physical_run -> physical_run isa CrystallisationProblem, physical_runs) ||
        throw(ArgumentError("Predictions require experiments or configured problems."))
    prediction_experiments = CrystallisationExperiment[]
    expected_axis = paramaxis(crystal_model.nucleation.law, crystal_model.growth.law,
        crystal_model.dissolution.law, crystal_model.aggregation.law, crystal_model.breakage.law)
    for (run_index, physical_run) in enumerate(physical_runs)
        paramaxis(physical_run) == expected_axis || throw(ArgumentError("Prediction run has a different kinetic schema."))
        for material_name in (:crystal_density, :volume_shape_factor, :molecular_volume)
            isequal(getproperty(physical_run, material_name), getproperty(crystal_model.system, material_name)) ||
                throw(ArgumentError("Prediction run conflicts with reference model's $material_name."))
        end
        _same_physical_configuration(physical_run.saturation_model, crystal_model.system.saturation_model) ||
            throw(ArgumentError("Prediction run conflicts with reference solubility model."))
        for slot_name in _MODEL_KINETIC_SLOTS
            field_name = Symbol("kinetics_", slot_name == :growth ? "growth" : string(slot_name), "function")
            _same_physical_configuration(getproperty(physical_run, field_name), getproperty(crystal_model, slot_name).law) ||
                throw(ArgumentError("Prediction run uses a different $slot_name law."))
        end
        requested_times = saveat === nothing ?
            (prediction_runs[run_index] isa CrystallisationExperiment ?
                _loss_saveat(prediction_runs[run_index], physical_run.solver; require_multiple = false) : [0.0, 28800.0]) : collect(saveat)
        _validate_save_times(requested_times)
        experiment_id = prediction_runs[run_index] isa CrystallisationExperiment ? prediction_runs[run_index].exp_id : run_index
        push!(prediction_experiments, CrystallisationExperiment(
            observables = (; concentration = Observable(time = requested_times, mean = zeros(length(requested_times)))),
            temperature = physical_run.temp_profile, exp_id = experiment_id))
    end
    prepared_predictions = prepare_loss(physical_runs, prediction_experiments;
        exclude_initial_concentration = false, mode, kwargs...)
    ensemble_predictions = run_ensemble(expanded_draws, prepared_predictions; observables)
    return PredictionResult(joint_samples, getproperty.(prediction_experiments, :exp_id), ensemble_predictions)
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
            noise_distribution = noise isa Distributions.UnivariateDistribution ? noise :
                noise(predictions.experiments[experiment_index], observable_name, ensemble_prediction.time[time_index])
            noise_distribution isa Distributions.UnivariateDistribution || throw(ArgumentError("Supply a univariate additive noise distribution."))
            for draw_index in axes(measurement_matrix, 1)
                ensemble_prediction.success[draw_index] || continue
                measurement_matrix[draw_index, time_index] += rand(rng, noise_distribution)
            end
        end
        measurement_matrix
    end
end
