"""
    runsimulation(problem::CrystallisationProblem; save_idx, algorithm=nothing,
                  solve_options=(;), callback_factory=nothing)

Simulate a configured physical problem without rebuilding its system properties.
Returns `(problem, solution)`. `initial_concentration(problem)` exposes the
configured initial input separately from the evolving solution concentration.
"""
function runsimulation(problem::CrystallisationProblem;
                       save_idx = 0:300.0:28_800.0, algorithm = nothing,
                       solve_options = (;), callback_factory = nothing)
    _validate_crystallisation_problem(problem)
    save_times = _validate_save_times(save_idx)
    return problem, _simulatecrystallisation(problem, save_times;
        algorithm, solve_options, callback_factory)
end

"""
    runsimulation(parameters, problem::CrystallisationProblem; kwargs...)

Replace only the kinetic parameter blocks of a configured problem and simulate.
Parameter order is the composite `paramaxis(problem)` order.
"""
function runsimulation(parameters::AbstractVector, problem::CrystallisationProblem;
                       kwargs...)
    parameter_blocks = _params_to_p(problem, parameters)
    parameter_problem = _copy_crystallisation_problem(problem;
        parameterset_nucleation = parameter_blocks.nucl,
        parameterset_growth = parameter_blocks.gr,
        parameterset_dissolution = parameter_blocks.diss,
        parameterset_aggregation = parameter_blocks.agg,
        parameterset_breakage = parameter_blocks.br)
    return runsimulation(parameter_problem; kwargs...)
end

"""
    crystallisation_solution(problem, ode_solution)

Convert a SciML solution of the package-generated ODE into its physical result
representation. The state layout must match `problem`; a DQMOM ODE state is
scaled weighted-node data, not the public physical weights and nodes.
"""
function crystallisation_solution(problem::CrystallisationProblem, ode_solution)
    isempty(ode_solution.u) && throw(ArgumentError("ODE solution has no saved states."))
    expected_count = length(_get_initial_state(problem))
    all(length(saved_state) == expected_count for saved_state in ode_solution.u) ||
        throw(ArgumentError("ODE solution state layout does not match the configured problem."))
    return _wrap_solution(problem, ode_solution)
end

function _setup_parameter_count(setup::LossSetup)
    configured_problem = setup.problem
    return _total_nparams(configured_problem.kinetics_nucleationfunction,
        configured_problem.kinetics_growthfunction,
        configured_problem.kinetics_aggregationfunction,
        configured_problem.kinetics_breakagefunction;
        diss = configured_problem.kinetics_dissolutionfunction)
end

function _validate_inference_bounds(setup, lower_bounds, upper_bounds)
    parameter_count = _setup_parameter_count(setup)
    length(lower_bounds) == length(upper_bounds) == parameter_count ||
        throw(ArgumentError("Bounds must contain $parameter_count kinetic parameters."))
    all(isfinite, lower_bounds) && all(isfinite, upper_bounds) &&
        all(lower_bounds .< upper_bounds) ||
        throw(ArgumentError("Parameter bounds must be finite with lower < upper."))
    return parameter_count
end

function _persist_parameter_fit(result, outputdir, extrastring, savetxt)
    savetxt && outputdir !== nothing || return
    mkpath(outputdir)
    fitted_parameters = hasproperty(result, :u) ? result.u : Metaheuristics.minimizer(result)
    objective_value = hasproperty(result, :objective) ? result.objective : Metaheuristics.minimum(result)
    filename = "$(Dates.format(now(), "yy-m-d HH-MM")) $(extrastring).txt"
    open(joinpath(outputdir, filename), "w") do fit_output
        println(fit_output, "Optimal parameters: ", fitted_parameters)
        println(fit_output, "Objective: ", objective_value)
    end
end

"""
    PE_Routine(lossfunction, setup::LossSetup, lower_bounds, upper_bounds; ...)

Fit the exact prepared systems using a configured Metaheuristics algorithm.
The routine preserves custom physical properties and experiment conditions.
"""
function PE_Routine(lossfunction::AbstractPELossFunction, setup::LossSetup,
                    lower_bounds::AbstractVector, upper_bounds::AbstractVector;
                    MHAlgorithm = nothing, nparticles = nothing, generations = nothing,
                    parallel_evaluation = nothing, verbosity = nothing,
                    HPC::Bool = false, outputdir = nothing, extrastring = "",
                    savetxt::Bool = true)
    _validate_inference_bounds(setup, lower_bounds, upper_bounds)
    run_algorithm = _configured_mh_algorithm(MHAlgorithm;
        nparticles, generations, parallel_evaluation, verbosity, HPC)
    parameter_bounds = Metaheuristics.boxconstraints(Float64.(lower_bounds), Float64.(upper_bounds))
    fit_result = _MHoptimise(run_algorithm, lossfunction, setup, parameter_bounds)
    _persist_parameter_fit(fit_result, outputdir, extrastring, savetxt)
    return fit_result
end

"""
    PE_Routine_Optimisation(lossfunction, setup::LossSetup, lower_bounds,
                            upper_bounds; searchalgo, x0=nothing, ...)

Optimization.jl fitting of prepared systems. `searchoptions` control the
optimizer; forward solve settings belong to `prepare_loss`.
"""
function PE_Routine_Optimisation(lossfunction::AbstractPELossFunction,
                                 setup::LossSetup, lower_bounds::AbstractVector,
                                 upper_bounds::AbstractVector; searchalgo,
                                 x0 = nothing, rng = Random.default_rng(),
                                 adtype = AutoForwardDiff(), searchoptions = (;),
                                 verbosity::Int = 1, HPC::Bool = false,
                                 outputdir = nothing, extrastring = "", savetxt::Bool = true)
    parameter_count = _validate_inference_bounds(setup, lower_bounds, upper_bounds)
    initial_parameters = x0 === nothing ?
        lower_bounds .+ (upper_bounds .- lower_bounds) .* rand(rng, parameter_count) : collect(x0)
    length(initial_parameters) == parameter_count &&
        all(lower_bounds .<= initial_parameters .<= upper_bounds) ||
        throw(ArgumentError("x0 must have one in-bounds value per kinetic parameter."))
    optimization_function = OptimizationBase.OptimizationFunction(
        (candidate_parameters, loss_context) -> loss(loss_context[1], loss_context[2], candidate_parameters),
        adtype)
    optimization_problem = OptimizationBase.OptimizationProblem(optimization_function,
        initial_parameters, (lossfunction, setup); lb = collect(lower_bounds), ub = collect(upper_bounds))
    optimizer_options = merge((; progress = verbosity > 0 && !HPC, maxiters = 2_000_000),
        (; searchoptions...))
    fit_result = OptimizationBase.solve(optimization_problem, searchalgo; optimizer_options...)
    _persist_parameter_fit(fit_result, outputdir, extrastring, savetxt)
    return fit_result
end

"""
    nuts_model(setup::LossSetup, prior; lossfunction=logMLE())

Build the Turing likelihood from the exact prepared physical systems. Each
sampling task owns a private copy of mutable preparation state.
"""
function nuts_model(setup::LossSetup, prior::AbstractVector{<:Distributions.Distribution};
                    lossfunction::AbstractPELossFunction = logMLE())
    length(prior) == _setup_parameter_count(setup) ||
        throw(ArgumentError("Supply one prior distribution per kinetic parameter."))
    setup_per_task = _cristool_once_per_task(LossSetup) do
        deepcopy(setup)
    end
    return _cristool_nuts_loss_model(collect(prior), lossfunction, setup_per_task)
end

"""
    MCMC_Routine(setup::LossSetup, prior; lossfunction=logMLE(), ...)

Sample prepared systems without replacing their temperature, feed, solvent
dynamics or initial conditions. Persistence requires an explicit outputdir.
"""
function MCMC_Routine(setup::LossSetup, prior::AbstractVector{<:Distributions.Distribution};
                      lossfunction::AbstractPELossFunction = logMLE(),
                      sampler = Turing.NUTS(1000, 0.65;
                          adtype = Turing.AutoForwardDiff(chunksize = 4)),
                      n_samples::Int = 1000, n_chains::Int = 4, burnin::Int = 0,
                      symbols = nothing, outputdir = nothing, extrastring = "",
                      saveplot::Bool = true, showplot::Bool = false, verbosity::Int = 1)
    n_samples > 0 && n_chains > 0 && 0 <= burnin < n_samples ||
        throw(ArgumentError("Sample/chain counts must be positive and burnin < n_samples."))
    configured_problem = setup.problem
    inferred_symbols = symbols === nothing ? kinetic_parameter_symbols(
        configured_problem.kinetics_nucleationfunction,
        configured_problem.kinetics_growthfunction,
        configured_problem.kinetics_aggregationfunction,
        configured_problem.kinetics_breakagefunction;
        diss = configured_problem.kinetics_dissolutionfunction) : collect(symbols)
    length(inferred_symbols) == _setup_parameter_count(setup) ||
        throw(ArgumentError("Supply one chain symbol per kinetic parameter."))
    sampled_chain = Turing.sample(nuts_model(setup, prior; lossfunction), sampler,
        Turing.MCMCThreads(), n_samples, n_chains; progress = verbosity > 1)
    named_chain = rename_chain(sampled_chain, inferred_symbols)
    burnin > 0 && (named_chain = named_chain[(burnin + 1):end, :, :])
    if outputdir !== nothing
        mkpath(outputdir)
        filename = "$(Dates.format(now(), "yy-m-d HH-MM")) $(extrastring)"
        jldsave(joinpath(outputdir, filename * ".jld2"); chain = named_chain, sampler, prior)
        if saveplot
            ChainPairPlots(named_chain, collect(mean(named_chain).nt.mean);
                prior = product_distribution(collect(prior)), symbols = inferred_symbols,
                savedir = outputdir, savestring = filename, showplot)
            ChainStatsPlots(named_chain; savedir = outputdir, savestring = filename,
                saveplot = true, showplot)
        end
    end
    return named_chain
end

# Problem convenience forms share the prepared workflow, rather than rebuilding
# lysozyme defaults inside each optimizer or sampler.
for workflow_name in (:PE_Routine, :PE_Routine_Optimisation)
    @eval function $workflow_name(lossfunction::AbstractPELossFunction,
                                 configured_problem::CrystallisationProblem, experiments,
                                 lower_bounds::AbstractVector, upper_bounds::AbstractVector;
                                 preparation_options = (;), kwargs...)
        prepared_setup = prepare_loss([configured_problem for _ in experiments], experiments;
            preparation_options...)
        return $workflow_name(lossfunction, prepared_setup, lower_bounds, upper_bounds; kwargs...)
    end
end

for workflow_name in (:nuts_model, :MCMC_Routine)
    @eval function $workflow_name(configured_problem::CrystallisationProblem, experiments,
                                 prior::AbstractVector{<:Distributions.Distribution};
                                 preparation_options = (;), kwargs...)
        prepared_setup = prepare_loss([configured_problem for _ in experiments], experiments;
            preparation_options...)
        return $workflow_name(prepared_setup, prior; kwargs...)
    end
end

"""
    PredictionEnsemble

Named sample predictions from a prepared physical system. `predictions` contains
matrices with samples in rows and saved times in columns. `success` and
`diagnostics` retain every candidate, including unsuccessful simulations.
Legacy concentration and size properties remain available through the embedded
ensemble result. Use `observable_values` for additional solvent/reactor values.
"""
struct PredictionEnsemble{TP <: NamedTuple, TL, TD} <: AbstractSolution
    time::Vector{Float64}
    predictions::TP
    success::Vector{Bool}
    diagnostics::TD
    legacy::TL
end

function Base.getproperty(prediction_ensemble::PredictionEnsemble, field_name::Symbol)
    field_name in fieldnames(typeof(prediction_ensemble)) &&
        return getfield(prediction_ensemble, field_name)
    named_predictions = getfield(prediction_ensemble, :predictions)
    hasproperty(named_predictions, field_name) && return getproperty(named_predictions, field_name)
    return getproperty(getfield(prediction_ensemble, :legacy), field_name)
end

Base.propertynames(prediction_ensemble::PredictionEnsemble) =
    (fieldnames(typeof(prediction_ensemble))...,
     propertynames(getfield(prediction_ensemble, :predictions))...,
     propertynames(getfield(prediction_ensemble, :legacy))...)

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
    run_ensemble(samples, setup::LossSetup; observables=nothing, ...)

Predict each prepared system for parameter samples in columns. Physical
configuration and solve settings are preserved; named solvent/reactor values
are retained alongside the established concentration and size predictions.
"""
function run_ensemble(parameter_samples::AbstractMatrix, setup::LossSetup;
                      observables = nothing, HPC::Bool = false, verbosity::Int = 1)
    size(parameter_samples, 1) == _setup_parameter_count(setup) ||
        throw(ArgumentError("Ensemble sample rows must match the kinetic parameter axis."))
    sample_count = size(parameter_samples, 2)
    sample_count > 0 || throw(ArgumentError("An ensemble requires at least one parameter sample."))
    prepared_predictions = PredictionEnsemble[]
    for prepared_experiment in setup.prepared
        sample_solutions = Vector{AbstractSolution}(undef, sample_count)
        Threads.@threads for sample_index in 1:sample_count
            private_preparation = deepcopy(prepared_experiment)
            sample_solutions[sample_index] = _solve_prepared(private_preparation,
                collect(view(parameter_samples, :, sample_index)))
        end
        reference_solution = first(sample_solutions)
        named_physical_values = merge(state_vars(reference_solution), size_metrics(reference_solution))
        builtin_names = Tuple(observable_name for observable_name in propertynames(named_physical_values)
            if getproperty(named_physical_values, observable_name) isa AbstractVector &&
               length(getproperty(named_physical_values, observable_name)) == length(reference_solution.time))
        prediction_names = observables === nothing ?
            (builtin_names..., propertynames(setup.observable_projections)...) : Tuple(Symbol.(observables))
        saved_times = collect(Float64, prepared_experiment.saveat)
        covers_prediction_times(sample_solution) = sample_solution.success &&
            first(sample_solution.time) <= first(saved_times) &&
            last(sample_solution.time) >= last(saved_times)
        prediction_matrices = map(prediction_names) do observable_name
            prediction_matrix = fill(NaN, sample_count, length(saved_times))
            for sample_index in 1:sample_count
                sample_solution = sample_solutions[sample_index]
                if covers_prediction_times(sample_solution)
                    sample_values = _simulated_at(setup, sample_solution, observable_name, saved_times)
                    length(sample_values) == length(saved_times) ||
                        throw(ArgumentError("Observable :$observable_name does not match saved times."))
                    prediction_matrix[sample_index, :] .= sample_values
                end
            end
            prediction_matrix
        end
        named_predictions = NamedTuple{prediction_names}(Tuple(prediction_matrices))
        # Legacy plotting requires concentration and size fields independently
        # of the optional named projection selection.
        function legacy_samples(observable_name)
            prediction_matrix = fill(NaN, length(saved_times), sample_count)
            for sample_index in 1:sample_count
                sample_solution = sample_solutions[sample_index]
                covers_prediction_times(sample_solution) || continue
                prediction_matrix[:, sample_index] .= _simulated_at(sample_solution, observable_name, saved_times)
            end
            return prediction_matrix
        end
        configured_solver = prepared_experiment.problem.solver
        legacy_result = _create_ensemble_solution(saved_times, legacy_samples(:concentration),
            legacy_samples(:d43), legacy_samples(:d32),
            configured_solver isa AbstractMomentSolver ? nothing : legacy_samples(:d50q),
            configured_solver)
        sample_success = [covers_prediction_times(sample_solution)
            for sample_solution in sample_solutions]
        sample_diagnostics = [(; success = sample_success[sample_index],
            ode_stats = sample_solutions[sample_index].ode_stats,
            reason = hasproperty(sample_solutions[sample_index], :reason) ?
                sample_solutions[sample_index].reason : nothing)
            for sample_index in 1:sample_count]
        push!(prepared_predictions, PredictionEnsemble(saved_times, named_predictions,
            sample_success, sample_diagnostics, legacy_result))
    end
    return prepared_predictions
end

"""
    run_ensemble(distribution, setup::LossSetup; n_samples=2048, rng, ...)

Draw kinetic parameter samples and predict the exact prepared systems.
"""
function run_ensemble(parameter_distribution::Distributions.Distribution,
                      setup::LossSetup; n_samples::Int = 2048,
                      rng = Random.default_rng(), kwargs...)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive."))
    return run_ensemble(rand(rng, parameter_distribution, n_samples), setup; kwargs...)
end

function plot_measurements_vs_ensemble(measurements::Vector{<:AbstractExperiment},
                                       prediction_ensembles::Vector{<:PredictionEnsemble},
                                       optimal_solutions; kwargs...)
    legacy_ensembles = [prediction_ensemble.legacy for prediction_ensemble in prediction_ensembles]
    return plot_measurements_vs_ensemble(measurements, legacy_ensembles, optimal_solutions; kwargs...)
end

function plot_ps_measurements_vs_ensemble(measurements::Vector{<:AbstractExperiment},
                                          prediction_ensembles::Vector{<:PredictionEnsemble},
                                          optimal_solutions; kwargs...)
    legacy_ensembles = [prediction_ensemble.legacy for prediction_ensemble in prediction_ensembles]
    return plot_ps_measurements_vs_ensemble(measurements, legacy_ensembles, optimal_solutions; kwargs...)
end
