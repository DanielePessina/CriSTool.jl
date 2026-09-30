"""
Loss functions and routines for estimating kinetic parameters via metaheuristic optimisation. Provides objective formulations used in ABC and MCMC workflows.
"""

"""
    PE_Routine(lossfunction, setofmeasurements, lb, ub,
               nucleationfunction, growthfunction,
               aggregationfunction, breakagefunction;
               diss = nodissolution(),
               solver, extrastring = "",
               MHAlgorithm = nothing,
               nparticles = nothing, generations = nothing,
               savetxt = true, HPC = false)

Estimate kinetic parameters by minimising `lossfunction` using a
population-based metaheuristic search.

# Arguments
- `setofmeasurements::Vector{<:AbstractExperiment}`: experimental data
  sets used to evaluate the loss.
- `lb`, `ub`: lower and upper bounds for the parameter vector. Both vectors
  must have length `nν + n_g + n_d + n_a + n_b` corresponding to the supplied
  kinetic functions.
- `nucleationfunction`, `growthfunction`, `aggregationfunction`,
  `breakagefunction`: kinetic models.
- `diss`: optional independent dissolution model; its parameter block follows
  the growth block.
- `solver::AbstractSolver`: numerical solver to run each simulation.
- `MHAlgorithm`: configured DE, NSGA2, SA or PSO algorithm. Its parameters,
  options, information and termination criteria are preserved; each run uses
  private copies and fresh status.
- `nparticles`, `generations`, `parallel_evaluation`, `verbosity`: explicit
  overrides of population size, iterations, batch evaluation and optimizer
  verbosity. Omitted keywords preserve supplied algorithm settings. Without
  `MHAlgorithm`, defaults are DE `:best1`, 128 particles/iterations and batch
  evaluation. `HPC = true` suppresses optimizer output.
- `extrastring`, `HPC`, `savetxt`: options for logging and reproducible
  runs.
- `parallel_evaluation`: evaluate Metaheuristics populations concurrently.
  Population matrices are handled through bounded Base-threaded blocks, with
  private setup copies and a collection point between blocks.

# Returns
The optimisation result from Metaheuristics.jl containing the best-fit
parameters and loss value.
"""
function PE_Routine(lossfunction::AbstractPELossFunction,
                    setofmeasurements::Vector{<:AbstractExperiment},
                    lb::AbstractVector{Float64}, ub::AbstractVector{Float64},
                    nucleationfunction::AbstractNucleationFunction,
                    growthfunction::AbstractGrowthFunction,
                    aggregationfunction::AbstractAggregationFunction,
                    breakagefunction::AbstractBreakageFunction;
                    diss::AbstractDissolutionFunction = nodissolution(),
                    solver::AbstractSolver,
                    extrastring::String = "",
                    MHAlgorithm::Union{Nothing, Metaheuristics.AbstractAlgorithm} = nothing,
                    nparticles::Union{Nothing, Int} = nothing,
                    generations::Union{Nothing, Int} = nothing,
                    savetxt::Bool = true,
                    verbosity::Union{Nothing, Int} = nothing,
                    HPC::Bool = false,
                    outputdir::Union{Nothing, AbstractString} = nothing,
                    parallel_evaluation::Union{Nothing, Bool} = nothing)

    parameter_bounds = boxconstraints(lb, ub)

    run_algorithm = _configured_mh_algorithm(MHAlgorithm;
        nparticles, generations, parallel_evaluation, verbosity, HPC)
    run_verbosity = something(verbosity, 1)
    run_population = run_algorithm.parameters.N
    run_generations = run_algorithm.options.iterations

    start_content = build_pe_start_content(lb, ub, lossfunction, solver,
                                           nucleationfunction, growthfunction,
                                           aggregationfunction, breakagefunction;
                                           nparticles = run_population,
                                           generations = run_generations,
                                           extrastring = extrastring)

    print_start_panel("Parameter Estimation (Metaheuristics)", start_content;
                      verbosity = run_verbosity)

    loss_problem = _build_loss_problem(nucleationfunction, growthfunction,
                                       aggregationfunction, breakagefunction, solver;
                                       diss = diss)

    results = _MHoptimise(run_algorithm, lossfunction, loss_problem, setofmeasurements,
                          parameter_bounds)
    #

    now_str::String = Dates.format(now(), "yy-m-d HH-MM")

    end_content = build_pe_end_content(minimizer(results), minimum(results))
    print_end_panel("Parameter Estimation", end_content; verbosity = run_verbosity)

    if savetxt && outputdir !== nothing
        outdir = String(outputdir)
        mkpath(outdir)
        startstring = ("Starting the parameter search at $(Dates.format(now(), "HH-MM"))
            \nOptimisation search settings:
            \nLB: $(lb) \nUB: $(ub)
            \nN: $(run_population)
            \nLoss function: $(lossfunction.string)
            \nSolver: $(solver.string)
            \nNucleation function: $(nucleationfunction.string)
            \nGrowth function: $(growthfunction.string)
            \nAggregation function: $(aggregationfunction.string)
            \nBreakage function: $(breakagefunction.string)
            \nID String: $(extrastring)")

        printstr::String = "Parameter estimation results: $(Dates.format(now(), "HH-MM"))
            \nOptimal parameters are: $(round.(minimizer(results), sigdigits = 5))
            \nLF Value : $(round.(minimum(results), sigdigits = 5))"

        open(joinpath(outdir, "$(now_str) $(extrastring).txt"), "w") do io
            write(io, startstring)
            write(io, "\n\n")
            write(io, printstr)
        end
    end


    return results
end
"""
    PE_Routine_Optimisation(lossfunction, setofmeasurements, lb, ub,
                            nucleationfunction, growthfunction,
                            aggregationfunction, breakagefunction;
                            diss = nodissolution(),
                            searchalgo, solver, x0=nothing, adtype=AutoForwardDiff(), ...)

Estimate kinetic parameters using gradient-based optimization.

# Arguments
- `lossfunction::AbstractPELossFunction`: Loss function to minimize
- `setofmeasurements::Vector{<:AbstractExperiment}`: Experimental data sets
- `lb`, `ub`: Lower and upper bounds for parameter vector
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `diss`: Optional independent dissolution model; its parameter block follows
  the growth block
- `searchalgo`: Optimization algorithm from Optimization.jl
- `solver::AbstractSolver`: Numerical solver for simulations
- `x0`: Optional initial guess (randomly sampled if nothing)
- `adtype`: Automatic differentiation type (default: AutoForwardDiff())
- `searchoptions`: Additional options passed to optimizer
- `extrastring`, `savetxt`, `verbosity`, `HPC`: Logging options

# Returns
- OptimizationResult containing optimal parameters and objective value
"""
function PE_Routine_Optimisation(lossfunction::AbstractPELossFunction,
                                 setofmeasurements::Vector{<:AbstractExperiment},
                                 lb::AbstractVector{Float64}, ub::AbstractVector{Float64},
                                 nucleationfunction::AbstractNucleationFunction,
                                 growthfunction::AbstractGrowthFunction,
                                 aggregationfunction::AbstractAggregationFunction,
                                 breakagefunction::AbstractBreakageFunction;
                                 diss::AbstractDissolutionFunction = nodissolution(),
                                 searchalgo,
                                 x0 = nothing,
                                 searchoptions = Dict{Symbol, Any}(),
                                 solver::AbstractSolver,
                                 extrastring::String = "",
                                 adtype = AutoForwardDiff(),
                                 savetxt::Bool = true,
                                 verbosity::Int64 = 1,
                                 HPC::Bool = false,
                                 outputdir::Union{Nothing, AbstractString} = nothing)


    start_content = build_pe_start_content(lb, ub, lossfunction, solver,
                                           nucleationfunction, growthfunction,
                                           aggregationfunction, breakagefunction;
                                           algorithm = searchalgo,
                                           extrastring = extrastring)

    print_start_panel("Parameter Estimation (Optimisation)", start_content;
                      verbosity = verbosity)

x0 = isnothing(x0) ? [lb[i] + (ub[i] - lb[i]) * rand() for i in eachindex(lb)] : x0

    loss_problem = _build_loss_problem(nucleationfunction, growthfunction,
                                       aggregationfunction, breakagefunction, solver;
                                       diss = diss)
    setup = prepare_loss(loss_problem, setofmeasurements)

    searchf = OptimizationBase.OptimizationFunction((x, (lf, loss_setup)) -> loss(lf,
                                                                                   loss_setup,
                                                                                   x),
                                                    adtype)

    optprob = OptimizationBase.OptimizationProblem(searchf, x0,
                                                   (lossfunction, setup), lb = lb, ub = ub)

    results = OptimizationBase.solve(optprob, searchalgo;
                                     progress = (verbosity > 0 && !HPC), maxiters = 2e6,
                                     searchoptions...)

    now_str::String = Dates.format(now(), "yy-m-d HH-MM")

    end_content = build_pe_end_content(results.u, results.objective; stats = results.stats)
    print_end_panel("Parameter Estimation", end_content; verbosity = verbosity)

    if savetxt && outputdir !== nothing
        outdir = String(outputdir)
        mkpath(outdir)
        startstring = ("Starting the parameter search at $(Dates.format(now(), "HH-MM"))
            \nOptimisation search settings:
            \nLB: $(lb) \nUB: $(ub)
            \nLoss function: $(lossfunction.string)
            \nSolver: $(solver.string)
            \nNucleation function: $(nucleationfunction.string)
            \nGrowth function: $(growthfunction.string)
            \nAggregation function: $(aggregationfunction.string)
            \nBreakage function: $(breakagefunction.string)
            \nID String: $(extrastring)
            \nAlgorithm: $(searchalgo)")

        printstr::String = "Parameter estimation results: $(Dates.format(now(), "HH-MM"))
            \nOptimal parameters are: $(round.(results.u, sigdigits = 5))
            \nLF Value : $(round.(results.objective, sigdigits = 5))"

        open(joinpath(outdir, "$(now_str) $(extrastring).txt"), "w") do io
            write(io, startstring)
            write(io, "\n\n")
            write(io, printstr)
            write(io, "\n\nIterations: $(results.stats.iterations)")
            write(io, "\nTime in seconds: $(results.stats.time)")
            write(io, "\nNumber of function evaluations: $(results.stats.fevals)")
            write(io, "\nNumber of gradient evaluations: $(results.stats.gevals)")
            write(io, "\nNumber of hessian evaluations: $(results.stats.hevals)")
        end
    end

    if verbosity > 1
        display(results.stats)
    end

    return results
end
"""
    MH_minimizer(results) -> Vector

Extract the minimizer (best parameters) from Metaheuristics optimization results.

# Arguments
- `results`: Optimization result from Metaheuristics.jl

# Returns
- Vector of optimal parameter values
"""
function MH_minimizer(results)
    return Metaheuristics.minimizer(results)
end

"""
    _build_loss_problem(nucleationfunction, growthfunction, aggregationfunction,
                        breakagefunction, solver) -> CrystallisationProblem

Build the `CrystallisationProblem` used by `loss` from kinetic models and a
solver. The problem carries the kinetics and solver; per-experiment conditions
(temperature, initial concentration, and initial crystals) are applied inside
`loss`.
"""
function _build_loss_problem(nucleationfunction::AbstractNucleationFunction,
                             growthfunction::AbstractGrowthFunction,
                             aggregationfunction::AbstractAggregationFunction,
                             breakagefunction::AbstractBreakageFunction,
                             solver::AbstractSolver;
                             diss::AbstractDissolutionFunction = nodissolution())
    return CrystallisationProblem(;
        kinetics_nucleationfunction = nucleationfunction,
        kinetics_growthfunction = growthfunction,
        kinetics_dissolutionfunction = diss,
        parameterset_nucleation = zeros(Float64, nucleationfunction.nparams),
        parameterset_growth = zeros(Float64, growthfunction.nparams),
        parameterset_dissolution = zeros(Float64, diss.nparams),
        kinetics_aggregationfunction = aggregationfunction,
        parameterset_aggregation = zeros(Float64, aggregationfunction.nparams),
        kinetics_breakagefunction = breakagefunction,
        parameterset_breakage = zeros(Float64, breakagefunction.nparams),
        solver = solver)
end

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
end

PreparedExperiment(problem, odeproblem, algorithm, saveat) =
    PreparedExperiment(problem, odeproblem, algorithm, saveat, (;), nothing)

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
                      callback_factory = nothing)
    isempty(experiments) && throw(ArgumentError("At least one experiment is required."))
    _validate_crystallisation_solve_options(solve_options)
    prepared = map(experiments) do expt
        per_exp_problem = _experiment_problem(problem, expt)
        saveat = _loss_saveat(expt, problem.solver)
        odeprob, default_algorithm = crystallisation_odeproblem(per_exp_problem,
                                                        saveat)
        PreparedExperiment(per_exp_problem, odeprob,
                           algorithm === nothing ? default_algorithm : algorithm,
                           saveat, solve_options, callback_factory)
    end
    return _build_loss_setup(problem, experiments, prepared;
                             observable_order, exclude_initial_concentration)
end

function _build_loss_setup(problem, experiments, prepared;
                           observable_order = nothing,
                           exclude_initial_concentration::Bool = true)
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
    return LossSetup(problem, experiments, prepared, names, included, heterogeneous,
                     !isnothing(observable_order))
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
                      callback_factory = nothing)
    _validate_crystallisation_solve_options(solve_options)
    length(configured_problems) == length(experiments) && !isempty(experiments) ||
        throw(ArgumentError("Supply one configured problem per experiment."))
    parameter_axis = paramaxis(first(configured_problems))
    all(paramaxis(configured) == parameter_axis for configured in configured_problems) ||
        throw(ArgumentError("Configured problems must share the same kinetic parameter axis."))
    prepared = map(zip(configured_problems, experiments)) do (configured, expt)
        if !isnothing(expt.initial_crystals)
            declared = initial_state_from_characteristics(configured, expt.initial_crystals)
            declared == _get_initial_state(configured) || throw(ArgumentError(
                "Experiment $(expt.exp_id) declares seeds conflicting with its configured problem."))
        end
        saveat = _loss_saveat(expt, configured.solver)
        _validate_crystallisation_problem(configured)
        _validate_save_times(saveat)
        odeproblem, default_algorithm = crystallisation_odeproblem(configured, saveat)
        PreparedExperiment(configured, odeproblem,
            algorithm === nothing ? default_algorithm : algorithm, saveat,
            solve_options, callback_factory)
    end
    return _build_loss_setup(first(configured_problems), experiments, prepared;
                             observable_order, exclude_initial_concentration)
end



"""
    batchLF_procSO(lossfunction, setup::LossSetup, parameter_mat) -> Vector

Evaluate the single-objective loss function for a batch of parameter sets in
parallel (rows of `parameter_mat` are parameter samples), each evaluation
using the prepared `ODEProblem` templates via `remake`.
"""
function batchLF_procSO(lossfunction::AbstractPELossFunction,
                        setup::LossSetup,
                        parameter_mat::AbstractMatrix{Float64})

    fx = zeros(Float64, size(parameter_mat, 1))
    _evaluate_loss_batch!(fx, lossfunction, setup, parameter_mat)

    return fx

end

function _evaluate_loss_batch!(loss_values::AbstractVector{Float64},
                              lossfunction::AbstractPELossFunction,
                              setup::LossSetup,
                              parameter_mat::AbstractMatrix{Float64})
    parameter_indices = axes(parameter_mat, 1)

    # Each spawned task owns its prepared ODE templates.  The loss itself
    # remakes a fresh ODEProblem for every parameter vector; the copy here is
    # only to keep any mutable solver/setup internals out of sibling tasks.
    if Threads.nthreads() == 1 || length(parameter_indices) <= 1
        local_setup = setup
        for i in parameter_indices
            loss_values[i] = loss(lossfunction, local_setup, @view parameter_mat[i, :])
        end
        return loss_values
    end

    first_index = first(parameter_indices)
    last_index = last(parameter_indices)
    for batch_start in first_index:CRISTOOL_PARALLEL_PE_BATCH_SIZE:last_index
        batch_stop = min(last_index, batch_start + CRISTOOL_PARALLEL_PE_BATCH_SIZE - 1)
        batch_length = batch_stop - batch_start + 1
        n_tasks = min(Threads.nthreads(), batch_length)
        chunk_length = cld(batch_length, n_tasks)
        tasks = Task[]
        for task_index in 1:n_tasks
            chunk_start = batch_start + (task_index - 1) * chunk_length
            chunk_stop = min(batch_stop, chunk_start + chunk_length - 1)
            push!(tasks, Threads.@spawn begin
                local_setup = deepcopy(setup)
                for i in chunk_start:chunk_stop
                    loss_values[i] = loss(lossfunction, local_setup, @view parameter_mat[i, :])
                end
            end)
        end
        foreach(fetch, tasks)
        GC.gc(false)
    end
    return loss_values
end

function batchLF_procSO(lossfunction::AbstractPELossFunction,
                        setup::LossSetup,
                        parameters::AbstractVector{<:Real})
    return loss(lossfunction, setup, parameters)
end

"""
    batchLF_procMO(lossfunction, setup::LossSetup, parameter_mat) -> Matrix

Evaluate the multi-objective loss function for a batch of parameter sets in
parallel (rows of `parameter_mat` are parameter samples, columns are the
concentration and particle-size objectives).
"""
function batchLF_procMO(lossfunction::AbstractPELossFunction,
                        setup::LossSetup,
                        parameter_mat::AbstractMatrix{Float64})

    Nt = size(parameter_mat, 1)
    objective_names = setup.observable_names
    fx = zeros(Nt, length(objective_names))
    _evaluate_multiobjective_batch!(fx, lossfunction, setup, parameter_mat)

    return fx

end

function _evaluate_multiobjective_batch!(objective_values::AbstractMatrix{Float64},
                                         lossfunction::AbstractPELossFunction,
                                         setup::LossSetup,
                                         parameter_mat::AbstractMatrix{Float64})
    parameter_indices = axes(parameter_mat, 1)

    function evaluate_range!(local_setup, range)
        for i in range
            objective_values[i, :] .= _loss_objectives(lossfunction, local_setup,
                                                       @view parameter_mat[i, :])
        end
    end

    if Threads.nthreads() == 1 || length(parameter_indices) <= 1
        evaluate_range!(setup, parameter_indices)
        return objective_values
    end

    first_index = first(parameter_indices)
    last_index = last(parameter_indices)
    for batch_start in first_index:CRISTOOL_PARALLEL_PE_BATCH_SIZE:last_index
        batch_stop = min(last_index, batch_start + CRISTOOL_PARALLEL_PE_BATCH_SIZE - 1)
        batch_length = batch_stop - batch_start + 1
        n_tasks = min(Threads.nthreads(), batch_length)
        chunk_length = cld(batch_length, n_tasks)
        tasks = Task[]
        for task_index in 1:n_tasks
            chunk_start = batch_start + (task_index - 1) * chunk_length
            chunk_stop = min(batch_stop, chunk_start + chunk_length - 1)
            push!(tasks, Threads.@spawn evaluate_range!(deepcopy(setup),
                                                        chunk_start:chunk_stop))
        end
        foreach(fetch, tasks)
        GC.gc(false)
    end
    return objective_values
end

function batchLF_procMO(lossfunction::AbstractPELossFunction,
                        setup::LossSetup,
                        parameters::AbstractVector{<:Real})
    return vec(batchLF_procMO(lossfunction, setup,
                              reshape(parameters, 1, length(parameters))))
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

function _MHoptimise(algo::Metaheuristics.Algorithm{<:Union{Metaheuristics.DE,
                     Metaheuristics.SA, Metaheuristics.PSO}}, lossfunction,
                     problem::CrystallisationProblem, experiments, parameter_bounds)
    setup = prepare_loss(problem, experiments)
    return _MHoptimise(algo, lossfunction, setup, parameter_bounds)
end

function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.NSGA2}, lossfunction,
                     problem::CrystallisationProblem, experiments, parameter_bounds)
    setup = prepare_loss(problem, experiments)
    return _MHoptimise(algo, lossfunction, setup, parameter_bounds)
end

function _MHoptimise(algo::Metaheuristics.Algorithm{<:Union{Metaheuristics.DE,
                     Metaheuristics.SA, Metaheuristics.PSO}}, lossfunction,
                     setup::LossSetup, parameter_bounds)
    return optimize(parameters -> batchLF_procSO(lossfunction, setup, parameters),
                    parameter_bounds, algo)
end

function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.NSGA2}, lossfunction,
                     setup::LossSetup, parameter_bounds)
    return optimize(parameters -> batchLF_procMO(lossfunction, setup, parameters),
                    parameter_bounds, algo)
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
                             expt::CrystallisationExperiment)
    template_solvent = isnothing(problem.initial_state) ?
                       problem.initial_solvent_state : solvent_state(problem, problem.initial_state)
    experiment_solvent = merge(template_solvent,
                               (; concentration = initial_concentration(expt)))
    experiment_state = if isnothing(problem.initial_state)
        nothing
    else
        # Preserve public population coordinates, including DQMOM weights/nodes.
        # Solvent conditions belong to the experiment; never mutate its template.
        vcat(problem.initial_state[_population_state_range(problem)],
             collect(values(experiment_solvent)))
    end
    experiment_problem = _copy_crystallisation_problem(problem;
        temp_profile = ConstantTemperature(expt.temperature),
        initial_concentration = initial_concentration(expt),
        initial_solvent_state = experiment_solvent,
        initial_state = experiment_state)
    isnothing(expt.initial_crystals) && return experiment_problem
    return _problem_with_initial_state(
        experiment_problem,
        initial_state_from_characteristics(experiment_problem, expt.initial_crystals))
end

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
    sol = _solve_crystallisation_ode(prep.problem, remade, prep.algorithm, prep.saveat;
                                     solve_options = prep.solve_options,
                                     callback_factory = prep.callback_factory)
    return _wrap_solution(prep.problem, sol)
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

function _loss_saveat(expt::CrystallisationExperiment, solver::AbstractSolver)
    times = Float64[]
    for name in _loss_observable_names(expt, solver)
        measured = getproperty(expt.observables, name)
        append!(times, Float64.(measured.time))
    end
    saveat = sort!(unique!(times))
    length(saveat) >= 2 ||
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
    fallback_variance = (0.1 * abs(mean_value))^2
    variance_floor = _scaled_variance_floor(mean_value, relative_variance_floor)
    return measured_variance === nothing ? max(variance_floor, fallback_variance) :
           max(measured_variance, variance_floor)
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
    expected_count = length(setup.prepared[1].odeproblem.p)
    length(params) == expected_count || throw(DimensionMismatch(
        "Expected $expected_count kinetic parameters, received $(length(params))."))
    objective_sums = [zero(eltype(params)) for _ in setup.observable_names]
    observation_counts = zeros(Int, length(setup.observable_names))
    for (experiment_index, (prep, expt)) in enumerate(zip(setup.prepared, setup.experiments))
        solution = _solve_prepared(prep, params)
        # Numerical failures have a definite penalty. Exceptions from data,
        # kinetics and custom observables propagate, including DomainError.
        solution.success || return fill(zero(eltype(params)) + CRISTOOL_FAILED_SIMULATION_PENALTY,
                                        length(setup.observable_names))
        for (objective_index, name) in enumerate(setup.observable_names)
            hasproperty(expt.observables, name) || continue
            measured = getproperty(expt.observables, name)
            included_indices = _included_observation_indices(setup, experiment_index, name)
            predicted = _simulated_at(solution, name, measured.time)
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
