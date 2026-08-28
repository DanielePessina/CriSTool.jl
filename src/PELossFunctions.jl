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
- `lb`, `ub`: lower and upper bounds for the parameter vector. Both
  vectors must have length `nν + n_g + n_a + n_b` corresponding to the
  parameters of the supplied kinetic functions.
- `nucleationfunction`, `growthfunction`, `aggregationfunction`,
  `breakagefunction`: kinetic models.
- `solver::AbstractSolver`: numerical solver to run each simulation.
- `nparticles`, `generations`: population size and number of iterations
  for the optimisation algorithm.
- `extrastring`, `HPC`, `savetxt`: options for logging and reproducible
  runs.

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
                    solver::AbstractSolver,
                    extrastring::String = "",
                    MHAlgorithm::Metaheuristics.AbstractAlgorithm = Metaheuristics.DE(),
                    nparticles::Int64 = 128,
                    generations::Int64 = 128,
                    savetxt::Bool = true,
                    verbosity::Int64 = 1,
                    HPC::Bool = false)

    parameter_bounds = boxconstraints(lb, ub)

    MHOptions = Metaheuristics.Options(iterations = generations, store_convergence = false,
                                       verbose = (verbosity > 1 && !HPC),
                                       parallel_evaluation = true,
                                       f_calls_limit = 1e18)

    start_content = build_pe_start_content(lb, ub, lossfunction, solver,
                                           nucleationfunction, growthfunction,
                                           aggregationfunction, breakagefunction;
                                           nparticles = nparticles,
                                           generations = generations,
                                           extrastring = extrastring)

    print_start_panel("Parameter Estimation (Metaheuristics)", start_content;
                      verbosity = verbosity)

    loss_problem = _build_loss_problem(nucleationfunction, growthfunction,
                                       aggregationfunction, breakagefunction, solver)

    results = _MHoptimise(MHAlgorithm, lossfunction, loss_problem, setofmeasurements,
                          parameter_bounds, nparticles, MHOptions)
    #

    now_str::String = Dates.format(now(), "yy-m-d HH-MM")

    end_content = build_pe_end_content(minimizer(results), minimum(results))
    print_end_panel("Parameter Estimation", end_content; verbosity = verbosity)

    if savetxt
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

        open(joinpath(pwd(), "R - PE Logs", "$(now_str) $(extrastring).txt"), "w") do io
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
                                 searchalgo,
                                 x0 = nothing,
                                 searchoptions = Dict{Symbol, Any}(),
                                 solver::AbstractSolver,
                                 extrastring::String = "",
                                 adtype = AutoForwardDiff(),
                                 savetxt::Bool = true,
                                 verbosity::Int64 = 1,
                                 HPC::Bool = false)


    start_content = build_pe_start_content(lb, ub, lossfunction, solver,
                                           nucleationfunction, growthfunction,
                                           aggregationfunction, breakagefunction;
                                           algorithm = searchalgo,
                                           extrastring = extrastring)

    print_start_panel("Parameter Estimation (Optimisation)", start_content;
                      verbosity = verbosity)

x0 = isnothing(x0) ? [lb[i] + (ub[i] - lb[i]) * rand() for i in eachindex(lb)] : x0

    loss_problem = _build_loss_problem(nucleationfunction, growthfunction,
                                       aggregationfunction, breakagefunction, solver)

    searchf = OptimizationBase.OptimizationFunction((x,
                                                     (lf, problem, experiments)) -> loss(lf,
                                                                                         problem,
                                                                                         x,
                                                                                         experiments),
                                                    adtype)

    optprob = OptimizationBase.OptimizationProblem(searchf, x0,
                                                   (lossfunction, loss_problem,
                                                    setofmeasurements), lb = lb, ub = ub)

    results = OptimizationBase.solve(optprob, searchalgo;
                                     progress = (verbosity > 0 && !HPC), maxiters = 2e6,
                                     searchoptions...)

    now_str::String = Dates.format(now(), "yy-m-d HH-MM")

    end_content = build_pe_end_content(results.u, results.objective; stats = results.stats)
    print_end_panel("Parameter Estimation", end_content; verbosity = verbosity)

    if savetxt
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

        open(joinpath(pwd(), "R - PE Logs", "$(now_str) $(extrastring).txt"), "w") do io
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
(temperature, loading, initial concentration) are applied inside `loss`.
"""
function _build_loss_problem(nucleationfunction::AbstractNucleationFunction,
                             growthfunction::AbstractGrowthFunction,
                             aggregationfunction::AbstractAggregationFunction,
                             breakagefunction::AbstractBreakageFunction,
                             solver::AbstractSolver)
    return CrystallisationProblem(;
        kinetics_nucleationfunction = nucleationfunction,
        kinetics_growthfunction = growthfunction,
        kinetics_aggregationfunction = aggregationfunction,
        kinetics_breakagefunction = breakagefunction,
        solver = solver)
end

"""
    batchLF_procSO(lossfunction, problem, parameter_mat, experiments) -> Vector

Evaluate the single-objective loss function for a batch of parameter sets in
parallel (rows of `parameter_mat` are parameter samples).
"""
function batchLF_procSO(lossfunction::AbstractPELossFunction,
                        problem::CrystallisationProblem,
                        parameter_mat::AbstractArray{Float64},
                        experiments::Vector{<:AbstractExperiment})

    fx = zeros(Float64, size(parameter_mat, 1))

    @floop for (i, θ) in enumerate(eachrow(parameter_mat))
        fx[i] = loss(lossfunction, problem, θ, experiments)
    end

    return fx

end

"""
    batchLF_procMO(lossfunction, problem, parameter_mat, experiments) -> Matrix

Evaluate the multi-objective loss function for a batch of parameter sets in
parallel (rows of `parameter_mat` are parameter samples, columns are the
concentration and particle-size objectives).
"""
function batchLF_procMO(lossfunction::AbstractPELossFunction,
                        problem::CrystallisationProblem,
                        parameter_mat::AbstractArray{Float64},
                        experiments::Vector{<:AbstractExperiment})

    Nt = size(parameter_mat, 1)
    fx = zeros(Nt, 2)

    @floop for i in 1:Nt
        fx[i, :] = _loss_objectives(lossfunction, problem, parameter_mat[i, :], experiments)
    end

    return fx

end

"""
    _MHoptimise(algo::Algorithm{DE}, ...) -> OptimizationResult

Run Differential Evolution optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.DE}, lossfunction,
                     problem::CrystallisationProblem, experiments, parameter_bounds,
                     nparticles, options)
    return optimize((x) -> batchLF_procSO(lossfunction, problem, x, experiments),
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
    return optimize((x) -> batchLF_procMO(lossfunction, problem, x, experiments),
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
    return optimize((x) -> batchLF_procSO(lossfunction, problem, x, experiments),
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
    return optimize((x) -> batchLF_procSO(lossfunction, problem, x, experiments),
                    parameter_bounds,
                    PSO(;
                        N = nparticles,
                        options = options))
end

##### Loss Functions

"""
    _solve_experiment(problem, params, expt) -> AbstractSolution

Simulate one experiment with `params` under the experiment's conditions
(initial concentration from the `concentration` series observable, constant
temperature profile, loading), saving at the measurement time grid.
"""
function _solve_experiment(problem::CrystallisationProblem, params,
                           expt::CrystallisationExperiment)
    obs = expt.observables
    p, solution = runsimulation(params;
                                nucl = problem.kinetics_nucleationfunction,
                                gr = problem.kinetics_growthfunction,
                                agg = problem.kinetics_aggregationfunction,
                                br = problem.kinetics_breakagefunction,
                                initial_concentration = initial_concentration(expt),
                                save_idx = obs.concentration.time,
                                solver = problem.solver,
                                loading = expt.loading,
                                temp_profile = ConstantTemperature(expt.temperature))
    return solution
end

"""
    _size_pair(problem, observables, solution) -> (ScalarObservable, AbstractVector)

Return the particle-size observable and simulated size trajectory used by
losses: `d43` for the MoM solver, `d50q` for discretised solvers (matching the
legacy loss behaviour).
"""
_size_pair(problem::CrystallisationProblem, observables, solution) =
    problem.solver isa MoM ? (observables.d43, solution.d43) :
    (observables.d50q, solution.d50q)

"""
    _loss_objectives(lf::AbstractPELossFunction, problem, params, experiments) -> (Float64, Float64)

Per-experiment (concentration, particle-size) objective contributions, summed
over experiments, with the legacy weighting semantics
(`weighting[i] * 0.5 * objective_i`).
"""
function _loss_objectives(lf::AbstractPELossFunction, problem::CrystallisationProblem,
                          params, experiments::Vector{<:AbstractExperiment})
    conc_contrib = 0.0
    size_contrib = 0.0
    for expt in experiments
        obs = expt.observables
        conc = obs.concentration
        solution = _solve_experiment(problem, params, expt)
        size_obs, size_sim = _size_pair(problem, obs, solution)

        conc_contrib += sum(log.(2π .* (conc.variance[2:end] .+ 1e-6)) .+
                            ((solution.concentration[2:end] .- conc.mean[2:end]) .^ 2) ./
                            (conc.variance[2:end] .+ 1e-6))
        size_contrib += log(2π * (size_obs.variance + 1e-6)) +
                        ((size_sim[end] - size_obs.value)^2) / (size_obs.variance + 1e-6)
    end
    return (lf.weighting[1] * 0.5 * conc_contrib, lf.weighting[2] * 0.5 * size_contrib)
end

"""
    loss(lf::logMLE, problem::CrystallisationProblem, params,
         experiments::Vector{CrystallisationExperiment}) -> Real

Log Maximum Likelihood Estimation loss over all experiments.

Evaluates the negative log-likelihood combining concentration trajectory
(measured times, first timepoint excluded) and final particle size
(`d43` for MoM, `d50q` for discretised solvers), with a `1e-6` variance floor,
matching the legacy `parameterestimation_lossfunction` semantics.
"""
function loss(lf::logMLE, problem::CrystallisationProblem, params,
              experiments::Vector{<:AbstractExperiment})
    return sum(_loss_objectives(lf, problem, params, experiments))
end

"""
    loss(lf::mae, problem::CrystallisationProblem, params,
         experiments::Vector{CrystallisationExperiment}) -> Real

Mean Absolute Error loss over all experiments: mean absolute concentration
error over all timepoints plus mean absolute final particle-size error
(`d43` for MoM, `d50q` for discretised solvers). Failed simulations
contribute a `1e6` penalty per timepoint.
"""
function loss(lf::mae, problem::CrystallisationProblem, params,
              experiments::Vector{<:AbstractExperiment})

    error_values = map(experiments) do expt
        obs = expt.observables
        conc = obs.concentration
        solution = _solve_experiment(problem, params, expt)
        size_obs, size_sim = _size_pair(problem, obs, solution)

        if solution.success
            (abs.(solution.concentration .- conc.mean),
             [abs(size_sim[end] - size_obs.value)])
        else
            (fill(1e6, length(conc.mean)), [1e6])
        end
    end

    all_conc_errors = vcat(map(x -> x[1], error_values)...)
    all_q_errors = vcat(map(x -> x[2], error_values)...)

    total_conc_mae = mean(all_conc_errors)
    total_q_mae = mean(all_q_errors)

    return lf.weighting[1] * total_conc_mae + lf.weighting[2] * total_q_mae
end
