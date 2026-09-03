"""
Loss functions and routines for estimating kinetic parameters via metaheuristic optimisation. Provides objective formulations used in ABC and MCMC workflows.
"""

"""
    PE_Routine(lossfunction, setofmeasurements, lb, ub,
               nucleationfunction, growthfunction,
               aggregationfunction, breakagefunction;
               solver, extrastring = "",
               MHAlgorithm = Metaheuristics.DE(),
               nparticles = 128, generations = 128,
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
- `solver::AbstractSolver`: numerical solver to run each simulation.
- `nparticles`, `generations`: population size and number of iterations
  for the optimisation algorithm.
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
                    MHAlgorithm::Metaheuristics.AbstractAlgorithm = Metaheuristics.DE(),
                    nparticles::Int64 = 128,
                    generations::Int64 = 128,
                    savetxt::Bool = true,
                    verbosity::Int64 = 1,
                    HPC::Bool = false,
                    outputdir::Union{Nothing, AbstractString} = nothing,
                    parallel_evaluation::Bool = true)

    parameter_bounds = boxconstraints(lb, ub)

    MHOptions = Metaheuristics.Options(iterations = generations, store_convergence = false,
                                       verbose = (verbosity > 1 && !HPC),
                                       parallel_evaluation = parallel_evaluation,
                                       f_calls_limit = CRISTOOL_MAX_OPTIMISER_CALLS)

    start_content = build_pe_start_content(lb, ub, lossfunction, solver,
                                           nucleationfunction, growthfunction,
                                           aggregationfunction, breakagefunction;
                                           nparticles = nparticles,
                                           generations = generations,
                                           extrastring = extrastring)

    print_start_panel("Parameter Estimation (Metaheuristics)", start_content;
                      verbosity = verbosity)

    loss_problem = _build_loss_problem(nucleationfunction, growthfunction,
                                       aggregationfunction, breakagefunction, solver;
                                       diss = diss)

    results = _MHoptimise(MHAlgorithm, lossfunction, loss_problem, setofmeasurements,
                          parameter_bounds, nparticles, MHOptions)
    #

    now_str::String = Dates.format(now(), "yy-m-d HH-MM")

    end_content = build_pe_end_content(minimizer(results), minimum(results))
    print_end_panel("Parameter Estimation", end_content; verbosity = verbosity)

    if savetxt && outputdir !== nothing
        outdir = String(outputdir)
        mkpath(outdir)
        startstring = ("Starting the parameter search at $(Dates.format(now(), "HH-MM"))
            \nOptimisation search settings:
            \nLB: $(lb) \nUB: $(ub)
            \nN: $(nparticles)
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
                            searchalgo, solver, x0=nothing, adtype=AutoForwardDiff(), ...)

Estimate kinetic parameters using gradient-based optimization.

# Arguments
- `lossfunction::AbstractPELossFunction`: Loss function to minimize
- `setofmeasurements::Vector{<:AbstractExperiment}`: Experimental data sets
- `lb`, `ub`: Lower and upper bounds for parameter vector
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
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
end

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
end

"""
    prepare_loss(problem::CrystallisationProblem,
                 experiments::Vector{CrystallisationExperiment}) -> LossSetup

Build per-experiment `ODEProblem` templates (u0 from each experiment's
initial concentration and initial crystals, tspan from its measurement grid,
constant temperature profile). Each loss evaluation then only remakes the parameter
vector — no ODEProblem construction in the optimisation loop.
"""
function prepare_loss(problem::CrystallisationProblem,
                      experiments::Vector{<:AbstractExperiment})
    prepared = map(experiments) do expt
        per_exp_problem = _experiment_problem(problem, expt)
        saveat = _loss_saveat(expt, problem.solver)
        odeprob, algorithm = crystallisation_odeproblem(per_exp_problem,
                                                        saveat)
        PreparedExperiment(per_exp_problem, odeprob, algorithm,
                           saveat)
    end
    return LossSetup(problem, experiments, prepared)
end


"""
    batchLF_procSO(lossfunction, setup::LossSetup, parameter_mat) -> Vector

Evaluate the single-objective loss function for a batch of parameter sets in
parallel (rows of `parameter_mat` are parameter samples), each evaluation
using the prepared `ODEProblem` templates via `remake`.
"""
function batchLF_procSO(lossfunction::AbstractPELossFunction,
                        setup::LossSetup,
                        parameter_mat::AbstractArray{Float64})

    fx = zeros(Float64, size(parameter_mat, 1))
    _evaluate_loss_batch!(fx, lossfunction, setup, parameter_mat)

    return fx

end

function _evaluate_loss_batch!(loss_values::AbstractVector{Float64},
                              lossfunction::AbstractPELossFunction,
                              setup::LossSetup,
                              parameter_mat::AbstractArray{Float64})
    parameter_indices = axes(parameter_mat, 1)

    # Each spawned task owns its prepared ODE templates.  The loss itself
    # remakes a fresh ODEProblem for every parameter vector; the copy here is
    # only to keep any mutable solver/setup internals out of sibling tasks.
    if Threads.nthreads() == 1 || length(parameter_indices) <= 1
        local_setup = setup
        for i in parameter_indices
            try
                loss_values[i] = loss(lossfunction, local_setup,
                                      @view parameter_mat[i, :])
            catch exception
                @debug "Candidate simulation failed during parameter estimation" exception =
                    (exception, catch_backtrace())
                loss_values[i] = CRISTOOL_FAILED_SIMULATION_PENALTY
            end
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
                    try
                        loss_values[i] = loss(lossfunction, local_setup,
                                              @view parameter_mat[i, :])
                    catch exception
                        @debug "Candidate simulation failed during parameter estimation" exception =
                            (exception, catch_backtrace())
                        loss_values[i] = CRISTOOL_FAILED_SIMULATION_PENALTY
                    end
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
    try
        return loss(lossfunction, setup, parameters)
    catch exception
        @debug "Candidate simulation failed during parameter estimation" exception =
            (exception, catch_backtrace())
        return CRISTOOL_FAILED_SIMULATION_PENALTY
    end
end

"""
    batchLF_procMO(lossfunction, setup::LossSetup, parameter_mat) -> Matrix

Evaluate the multi-objective loss function for a batch of parameter sets in
parallel (rows of `parameter_mat` are parameter samples, columns are the
concentration and particle-size objectives).
"""
function batchLF_procMO(lossfunction::AbstractPELossFunction,
                        setup::LossSetup,
                        parameter_mat::AbstractArray{Float64})

    Nt = size(parameter_mat, 1)
    objective_names = _loss_observable_names(first(setup.experiments), setup.problem.solver)
    fx = zeros(Nt, length(objective_names))
    _evaluate_multiobjective_batch!(fx, lossfunction, setup, parameter_mat)

    return fx

end

function _evaluate_multiobjective_batch!(objective_values::AbstractMatrix{Float64},
                                         lossfunction::AbstractPELossFunction,
                                         setup::LossSetup,
                                         parameter_mat::AbstractArray{Float64})
    parameter_indices = axes(parameter_mat, 1)

    function evaluate_range!(local_setup, range)
        for i in range
            try
                for (prep, expt) in zip(local_setup.prepared, local_setup.experiments)
                    objectives = _experiment_objectives(lossfunction, expt,
                                                        _solve_prepared(prep,
                                                                        parameter_mat[i, :]))
                    for j in eachindex(objectives)
                        objective_values[i, j] += objectives[j]
                    end
                end
            catch exception
                @debug "Candidate simulation failed during multi-objective parameter estimation" exception =
                    (exception, catch_backtrace())
                objective_values[i, :] .= CRISTOOL_FAILED_SIMULATION_PENALTY
            end
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

"""
    _MHoptimise(algo::Algorithm{DE}, ...) -> OptimizationResult

Run Differential Evolution optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.DE}, lossfunction,
                     problem::CrystallisationProblem, experiments, parameter_bounds,
                     nparticles, options)
    setup = prepare_loss(problem, experiments)
    return optimize((x) -> batchLF_procSO(lossfunction, setup, x),
                    parameter_bounds,
                    DE(;
                       N = nparticles,
                       strategy = :best1,
                       options = options))
end

"""
    _MHoptimise(algo::Algorithm{NSGA2}, ...) -> OptimizationResult

Run NSGA-II multi-objective optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.NSGA2}, lossfunction,
                     problem::CrystallisationProblem, experiments, parameter_bounds,
                     nparticles, options)
    setup = prepare_loss(problem, experiments)
    return optimize((x) -> batchLF_procMO(lossfunction, setup, x),
                    parameter_bounds,
                    NSGA2(;
                          N = nparticles,
                          options = options))
end

"""
    _MHoptimise(algo::Algorithm{SA}, ...) -> OptimizationResult

Run Simulated Annealing optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.SA}, lossfunction,
                     problem::CrystallisationProblem, experiments, parameter_bounds,
                     nparticles, options)
    setup = prepare_loss(problem, experiments)
    return optimize((x) -> batchLF_procSO(lossfunction, setup, x),
                    parameter_bounds,
                    SA(;
                       N = nparticles,
                       options = options))
end

"""
    _MHoptimise(algo::Algorithm{PSO}, ...) -> OptimizationResult

Run Particle Swarm Optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.PSO}, lossfunction,
                     problem::CrystallisationProblem, experiments, parameter_bounds,
                     nparticles, options)
    setup = prepare_loss(problem, experiments)
    return optimize((x) -> batchLF_procSO(lossfunction, setup, x),
                    parameter_bounds,
                    PSO(;
                        N = nparticles,
                        options = options))
end

"""
    _experiment_problem(problem, expt) -> CrystallisationProblem

The problem with the experiment's conditions applied (temperature profile,
initial concentration, and initial crystal characteristics). Kinetics,
saturation model, and physical constants carry over from the base problem.
"""
function _experiment_problem(problem::CrystallisationProblem,
                             expt::CrystallisationExperiment)
    experiment_problem = CrystallisationProblem(;
        temp_profile = ConstantTemperature(expt.temperature),
        crystal_density = problem.crystal_density,
        initial_concentration = initial_concentration(expt),
        initial_solvent_state = merge(problem.initial_solvent_state,
                                      (; concentration = initial_concentration(expt))),
        solvent_dynamics = problem.solvent_dynamics,
        saturation_model = problem.saturation_model,
        volume_shape_factor = problem.volume_shape_factor,
        solid_mass_concentration_threshold = problem.solid_mass_concentration_threshold,
        molecular_volume = problem.molecular_volume,
        kinetics_nucleationfunction = problem.kinetics_nucleationfunction,
        kinetics_growthfunction = problem.kinetics_growthfunction,
        kinetics_dissolutionfunction = problem.kinetics_dissolutionfunction,
        kinetics_aggregationfunction = problem.kinetics_aggregationfunction,
        kinetics_breakagefunction = problem.kinetics_breakagefunction,
        parameterset_nucleation = problem.parameterset_nucleation,
        parameterset_growth = problem.parameterset_growth,
        parameterset_dissolution = problem.parameterset_dissolution,
        parameterset_aggregation = problem.parameterset_aggregation,
        parameterset_breakage = problem.parameterset_breakage,
        R_gas_constant = problem.R_gas_constant,
        boltzmann_constant = problem.boltzmann_constant,
        solver = problem.solver)
    isnothing(expt.initial_crystals) && return experiment_problem
    return _problem_with_initial_state(
        experiment_problem,
        initial_state_from_characteristics(experiment_problem, expt.initial_crystals))
end

"""
    _params_to_p(prob, params) -> NamedTuple

Split a flat parameter vector (documented order
`[p_nucleation; p_growth; p_aggregation; p_breakage]`) into the named-tuple
parameter container used by the ODE right-hand sides. The field types match
the template's `p` exactly for the same parameter element type, so `remake`
keeps the problem type stable across evaluations.
"""
function _params_to_p(prob::CrystallisationProblem, params)
    # ComponentArray with the composite kinetic axis: `p.nucl`/`p.gr` are
    # views, so the rate functions' `_named_params` short-circuits without
    # rebuilding the parameter container on every ODE step.
    return ComponentArray(params, paramaxis(prob))
end

function _solve_prepared_problem(odeproblem, algorithm, saveat,
                                 solver::AbstractMomentSolver)
    return solve(odeproblem, algorithm;
                 saveat = saveat,
                 reltol = solver.reltol,
                 abstol = solver.abstol,
                 dense = false,
                 maxiters = CRISTOOL_MAX_PREPARED_SOLVER_ITERS,
                 maxtime = CRISTOOL_MAX_PREPARED_SOLVER_SECONDS)
end

function _solve_prepared_problem(odeproblem, algorithm, saveat,
                                 solver::AbstractDiscretisedSolver)
    return solve(odeproblem, algorithm;
                 saveat = saveat,
                 reltol = solver.reltol,
                 abstol = solver.abstol,
                 dense = false,
                 alg_hints = [:stiff],
                 maxiters = CRISTOOL_MAX_PREPARED_DISCRETIZED_SOLVER_ITERS,
                 maxtime = CRISTOOL_MAX_PREPARED_SOLVER_SECONDS)
end

"""
    _solve_prepared(prep::PreparedExperiment, params) -> AbstractSolution

Solve a prepared experiment for a parameter vector: `remake` the template
ODEProblem with the named parameter container, then solve with the same
options as `_simulatecrystallisation`.
"""
function _solve_prepared(prep::PreparedExperiment, params)
    remade = remake(prep.odeproblem; p = _params_to_p(prep.problem, params))
    sol = _solve_prepared_problem(remade, prep.algorithm, prep.saveat,
                                  prep.problem.solver)
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

function _observable_weight(lossfunction::AbstractPELossFunction, index::Int)
    return index <= length(lossfunction.weighting) ? lossfunction.weighting[index] : 1.0
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
        first_index = name === :concentration ? 2 : 1

        if lf isa logMLE
            objective = 0.0
            for index in first_index:length(measured_mean)
                variance = _measurement_variance(measured, measured_mean[index], index,
                                                 lf.variance_model,
                                                 lf.relative_variance_floor)
                residual = simulated_mean[index] - measured_mean[index]
                objective += log(2π * variance) + residual^2 / variance
            end
            return _observable_weight(lf, observable_index) * 0.5 * objective
        end

        if !solution.success
            return _observable_weight(lf, observable_index) * CRISTOOL_FAILED_SIMULATION_PENALTY
        end
        objective = first_index > length(measured_mean) ? zero(eltype(simulated_mean)) :
                    mean(abs.(simulated_mean[first_index:end] .-
                              measured_mean[first_index:end]))
        return _observable_weight(lf, observable_index) * objective
    end
    return contributions
end

"""
    loss(lf::logMLE, setup::LossSetup, params) -> Real

Log Maximum Likelihood Estimation loss over all prepared experiments. The
initial concentration point is excluded because it defines the experiment's
initial condition; all points of every other observable contribute.
"""
function loss(lf::logMLE, setup::LossSetup, params)
    total = 0.0
    for (prep, expt) in zip(setup.prepared, setup.experiments)
        try
            solution = _solve_prepared(prep, params)
            if solution.success
                total += sum(_experiment_objectives(lf, expt, solution))
            else
                total += CRISTOOL_FAILED_SIMULATION_PENALTY
            end
        catch
            total += CRISTOOL_FAILED_SIMULATION_PENALTY
        end
    end
    return total
end

"""
    loss(lf::mae, setup::LossSetup, params) -> Real

Mean Absolute Error loss over all prepared experiments. Every point of every
observable contributes to its observable-specific mean absolute error. Failed
simulations contribute the configured failure penalty per observable.
"""
function loss(lf::mae, setup::LossSetup, params)
    objective_names = _loss_observable_names(first(setup.experiments), setup.problem.solver)
    error_sums = [zero(eltype(params)) for _ in objective_names]
    error_counts = zeros(Int, length(objective_names))
    for (prep, expt) in zip(setup.prepared, setup.experiments)
        try
            solution = _solve_prepared(prep, params)
            for (observable_index, name) in enumerate(_loss_observable_names(expt, solution))
                measured = getproperty(expt.observables, name)
                measured_time, measured_mean = _measurement_data(measured)
                if solution.success
                    simulated_mean = _simulated_at(solution, name, measured_time)
                    first_index = name === :concentration ? 2 : 1
                    error_sums[observable_index] +=
                        sum(abs.(simulated_mean[first_index:end] .-
                                measured_mean[first_index:end]))
                    error_counts[observable_index] +=
                        max(length(measured_mean) - first_index + 1, 0)
                else
                    error_sums[observable_index] += CRISTOOL_FAILED_SIMULATION_PENALTY
                    error_counts[observable_index] += 1
                end
            end
        catch
            error_sums .+= CRISTOOL_FAILED_SIMULATION_PENALTY
            error_counts .+= 1
        end
    end
    return sum(error_counts[i] == 0 ? zero(eltype(params)) :
               _observable_weight(lf, i) * error_sums[i] / error_counts[i]
               for i in eachindex(objective_names))
end

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
