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
- `setofmeasurements::Vector{<:AbstractMeasurements}`: experimental data
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
                    setofmeasurements::Vector{<:AbstractMeasurements},
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

    results = _MHoptimise(MHAlgorithm, lossfunction, setofmeasurements, parameter_bounds,
                          nucleationfunction, growthfunction, aggregationfunction,
                          breakagefunction, solver, parameter_bounds, nparticles, MHOptions)
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
- `setofmeasurements::Vector{<:AbstractMeasurements}`: Experimental data sets
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
                                 setofmeasurements::Vector{<:AbstractMeasurements},
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

    searchf = OptimizationBase.OptimizationFunction((x,
                                                     (lf, measurements, nucleationf,
                                                      growthf, aggf, brf, solv)) -> CriSTool.parameterestimation_lossfunction(lf,
                                                                                                                              measurements,
                                                                                                                              x,
                                                                                                                              nucleationf,
                                                                                                                              growthf,
                                                                                                                              aggf,
                                                                                                                              brf,
                                                                                                                              solv),
                                                    adtype)

    optprob = OptimizationBase.OptimizationProblem(searchf, x0,
                                                   (lossfunction, setofmeasurements,
                                                    nucleationfunction, growthfunction,
                                                    aggregationfunction, breakagefunction,
                                                    solver), lb = lb, ub = ub)

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
    batchLF_procSO(lossfunction, datasets, parameter_mat, nucleationfunction,
                   growthfunction, aggregationfunction, breakagefunction; solver) -> Vector

Evaluate single-objective loss function for a batch of parameter sets in parallel.

# Arguments
- `lossfunction::AbstractPELossFunction`: Loss function to evaluate
- `datasets::Vector{<:AbstractMeasurements}`: Experimental data
- `parameter_mat::AbstractArray{Float64}`: Matrix of parameter sets (rows are samples)
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::AbstractSolver`: Numerical solver

# Returns
- Vector of loss values for each parameter set
"""
function batchLF_procSO(lossfunction::AbstractPELossFunction,
                        datasets::Vector{<:AbstractMeasurements},
                        parameter_mat::AbstractArray{Float64},
                        nucleationfunction::AbstractNucleationFunction,
                        growthfunction::AbstractGrowthFunction,
                        aggregationfunction::AbstractAggregationFunction,
                        breakagefunction::AbstractBreakageFunction;
                        solver::AbstractSolver)

    # parameter_mat_tp = transpose(parameter_mat)

    fx = zeros(Float64, size(parameter_mat, 1))

    @floop for (i, θ) in enumerate(eachrow(parameter_mat))
        fx[i] = parameterestimation_lossfunction(lossfunction, datasets, θ,
                                                 nucleationfunction, growthfunction,
                                                 aggregationfunction, breakagefunction,
                                                 solver)

    end

    return fx

end
"""
    batchLF_procMO(lossfunction, datasets, parameter_mat, nucleationfunction,
                   growthfunction, aggregationfunction, breakagefunction; solver) -> Matrix

Evaluate multi-objective loss function for a batch of parameter sets in parallel.

# Arguments
- `lossfunction::AbstractPELossFunction`: Loss function to evaluate
- `datasets::Vector{<:AbstractMeasurements}`: Experimental data
- `parameter_mat::AbstractArray{Float64}`: Matrix of parameter sets (rows are samples)
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::AbstractSolver`: Numerical solver

# Returns
- Matrix of loss values (rows are samples, columns are objectives)
"""
function batchLF_procMO(lossfunction::AbstractPELossFunction,
                        datasets::Vector{<:AbstractMeasurements},
                        parameter_mat::AbstractArray{Float64},
                        nucleationfunction::AbstractNucleationFunction,
                        growthfunction::AbstractGrowthFunction,
                        aggregationfunction::AbstractAggregationFunction,
                        breakagefunction::AbstractBreakageFunction;
                        solver::AbstractSolver)

    Nt = size(parameter_mat, 1)
    fx = zeros(Nt, 2)

    @floop for i in 1:Nt
        fx[i, :] = parameterestimation_lossfunction(lossfunction, datasets,
                                                    parameter_mat[i, :], nucleationfunction,
                                                    growthfunction, aggregationfunction,
                                                    breakagefunction, solver)

    end

    return fx

end
"""
    _MHoptimise(algo::Algorithm{DE}, ...) -> OptimizationResult

Run Differential Evolution optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.DE}, lossfunction,
                     setofmeasurements, x, nucleationfunction, growthfunction,
                     aggregationfunction, breakagefunction, solver, parameter_bounds,
                     nparticles, options)
    return optimize((x) -> batchLF_procSO(lossfunction, setofmeasurements, x,
                                          nucleationfunction, growthfunction,
                                          aggregationfunction, breakagefunction,
                                          solver = solver), parameter_bounds,
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
                     setofmeasurements, x, nucleationfunction, growthfunction,
                     aggregationfunction, breakagefunction, solver, parameter_bounds,
                     nparticles, options)
    return optimize((x) -> batchLF_procMO(lossfunction, setofmeasurements, x,
                                          nucleationfunction, growthfunction,
                                          aggregationfunction, breakagefunction,
                                          solver = solver), parameter_bounds,
                    NSGA2(;
                          N = nparticles,
                          options = options))
end

"""
    _MHoptimise(algo::Algorithm{SA}, ...) -> OptimizationResult

Run Simulated Annealing optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{Metaheuristics.SA}, lossfunction,
                     setofmeasurements, x, nucleationfunction, growthfunction,
                     aggregationfunction, breakagefunction, solver, parameter_bounds,
                     nparticles, options)
    return optimize((x) -> batchLF_procSO(lossfunction, setofmeasurements, x,
                                          nucleationfunction, growthfunction,
                                          aggregationfunction, breakagefunction,
                                          solver = solver), parameter_bounds,
                    SA(;
                       N = nparticles,
                       options = options))
end

"""
    _MHoptimise(algo::Algorithm{PSO}, ...) -> OptimizationResult

Run Particle Swarm Optimization for parameter estimation.
"""
function _MHoptimise(algo::Metaheuristics.Algorithm{PSO}, lossfunction, setofmeasurements,
                     x, nucleationfunction, growthfunction, aggregationfunction,
                     breakagefunction, solver, parameter_bounds, nparticles, options)
    return optimize((x) -> batchLF_procSO(lossfunction, setofmeasurements, x,
                                          nucleationfunction, growthfunction,
                                          aggregationfunction, breakagefunction,
                                          solver = solver), parameter_bounds,
                    PSO(;
                        N = nparticles,
                        options = options))
end






##### Loss Functions

"""
    parameterestimation_lossfunction(lf::logMLE, datasets, parameters,
                                     nucleationfunction, growthfunction,
                                     aggregationfunction, breakagefunction,
                                     solver::AbstractDiscretisedSolver) -> Real

Compute log Maximum Likelihood Estimation loss for discretised solvers.

Evaluates the negative log-likelihood over all datasets, combining concentration
trajectory and final particle size (d50q) errors with Gaussian likelihood.

# Arguments
- `lf::logMLE`: Loss function with weighting factors
- `datasets::Vector{CrystallisationRepeatMeasurements}`: Experimental data
- `parameters::AbstractArray{<:Real}`: Parameter vector to evaluate
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::AbstractDiscretisedSolver`: Finite volume or WENO solver

# Returns
- Total weighted log-likelihood loss value
"""
function parameterestimation_lossfunction(lf::logMLE,
                                          datasets::Vector{CrystallisationRepeatMeasurements},
                                          parameters::TArr,
                                          nucleationfunction::AbstractNucleationFunction,
                                          growthfunction::AbstractGrowthFunction,
                                          aggregationfunction::AbstractAggregationFunction,
                                          breakagefunction::AbstractBreakageFunction,
                                          solver::AbstractDiscretisedSolver) where {TArr <:
                                                                                    AbstractArray{<:Real}}
    loss_values = map(m -> begin
                          problem,
                          solution = runsimulation(parameters,
                                                   nucl = nucleationfunction,
                                                   gr = growthfunction,
                                                   agg = aggregationfunction,
                                                   br = breakagefunction,
                                                   initial_concentration = datasets[m].concentrationmean[1],
                                                   save_idx = datasets[m].time,
                                                   solver = solver,
                                                   loading = datasets[m].loading,
                                                   temp_profile = ConstantTemperature(datasets[m].temperature))

                          loss1 = sum(log.(2π *
                                           (datasets[m].concentrationvariance[2:end] .+
                                            1e-6)) .+
                                      ((solution.concentration[2:end] .-
                                        datasets[m].concentrationmean[2:end]) .^ 2) ./
                                      (datasets[m].concentrationvariance[2:end] .+ 1e-6))
                          loss2 = (log(2π * (datasets[m].quantilevariance .+ 1e-6)) +
                                   ((solution.d50q[end] .- datasets[m].quantilemean) .^ 2) ./
                                   (datasets[m].quantilevariance .+ 1e-6))
                          loss3 = sum(lf.weighting[1] * 0.5 * loss1 +
                                      lf.weighting[2] * 0.5 * loss2)


                          return (loss1, loss2, loss3)
                      end, eachindex(datasets))

    total_loss = sum(map(x -> x[3], loss_values))

    return total_loss
end
"""
    parameterestimation_lossfunction(lf::logMLE, datasets, parameters,
                                     nucleationfunction, growthfunction,
                                     aggregationfunction, breakagefunction,
                                     solver::MoM) -> Real

Compute log Maximum Likelihood Estimation loss for Method of Moments solver.

Uses d43 instead of d50q for particle size comparison since MoM doesn't compute quantiles.

# Arguments
- `lf::logMLE`: Loss function with weighting factors
- `datasets::Vector{CrystallisationRepeatMeasurements}`: Experimental data
- `parameters::AbstractVector{<:Real}`: Parameter vector to evaluate
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::MoM`: Method of Moments solver

# Returns
- Total weighted log-likelihood loss value
"""
function parameterestimation_lossfunction(lf::logMLE,
                                          datasets::Vector{CrystallisationRepeatMeasurements},
                                          parameters::TArr,
                                          nucleationfunction::AbstractNucleationFunction,
                                          growthfunction::AbstractGrowthFunction,
                                          aggregationfunction::AbstractAggregationFunction,
                                          breakagefunction::AbstractBreakageFunction,
                                          solver::MoM) where {TArr <:
                                                              AbstractVector{<:Real}}

    loss_values = map(m -> begin
                          problem,
                          solution = runsimulation(parameters,
                                                   nucl = nucleationfunction,
                                                   gr = growthfunction,
                                                   agg = aggregationfunction,
                                                   br = breakagefunction,
                                                   initial_concentration = datasets[m].concentrationmean[1],
                                                   save_idx = datasets[m].time,
                                                   solver = solver,
                                                   loading = datasets[m].loading,
                                                   temp_profile = ConstantTemperature(datasets[m].temperature))

                          loss1 = sum(log.(2π *
                                           (datasets[m].concentrationvariance[2:end] .+
                                            1e-6)) .+
                                      ((solution.concentration[2:end] .-
                                        datasets[m].concentrationmean[2:end]) .^ 2) ./
                                      (datasets[m].concentrationvariance[2:end] .+ 1e-6))
                          loss2 = (log(2π * (datasets[m].d43var .+ 1e-6)) +
                                   ((solution.d43[end] .- datasets[m].d43) .^ 2) ./
                                   (datasets[m].d43var .+ 1e-6))
                          loss3 = sum(lf.weighting[1] * 0.5 * loss1 +
                                      lf.weighting[2] * 0.5 * loss2)

                          return (loss1, loss2, loss3)
                      end, eachindex(datasets))

    total_loss = sum(map(x -> x[3], loss_values))

    return total_loss
end

"""
    parameterestimation_lossfunction(lf::mae, datasets, parameters,
                                     nucleationfunction, growthfunction,
                                     aggregationfunction, breakagefunction,
                                     solver::AbstractDiscretisedSolver) -> Real

Compute Mean Absolute Error loss for discretised solvers.

# Arguments
- `lf::mae`: Loss function with weighting factors
- `datasets::Vector{<:AbstractMeasurements}`: Experimental data
- `parameters::AbstractVector{<:Real}`: Parameter vector to evaluate
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::AbstractDiscretisedSolver`: Finite volume or WENO solver

# Returns
- Weighted sum of concentration MAE and particle size MAE
"""
function parameterestimation_lossfunction(lf::mae, datasets::Vector{<:AbstractMeasurements},
                                          parameters::AbstractVector{TPara},
                                          nucleationfunction::AbstractNucleationFunction,
                                          growthfunction::AbstractGrowthFunction,
                                          aggregationfunction::AbstractAggregationFunction,
                                          breakagefunction::AbstractBreakageFunction,
                                          solver::AbstractDiscretisedSolver) where {TPara <:
                                                                                    Real}

    error_values = map(datasets) do dataset
        problem,
        solution = runsimulation(parameters,
                                 nucl = nucleationfunction,
                                 gr = growthfunction,
                                 agg = aggregationfunction,
                                 br = breakagefunction,
                                 initial_concentration = dataset.concentrationmean[1],
                                 save_idx = dataset.time,
                                 solver = solver,
                                 loading = dataset.loading,
                                 temp_profile = ConstantTemperature(dataset.temperature))

        if solution.success
            conc_errors = abs.(solution.concentration .- dataset.concentrationmean)
            q_errors = [abs(solution.d50q[end] - dataset.quantilemean)]
            (conc_errors, q_errors)
        else
            # Penalty values for failed simulations
            penalty_conc = fill(1e6, length(dataset.concentrationmean))
            penalty_q = [1e6]
            (penalty_conc, penalty_q)
        end
    end

    # Concatenate all errors into single vectors
    all_conc_errors = vcat(map(x -> x[1], error_values)...)
    all_q_errors = vcat(map(x -> x[2], error_values)...)

    total_conc_mae = mean(all_conc_errors)
    total_q_mae = mean(all_q_errors)

    return lf.weighting[1] * total_conc_mae + lf.weighting[2] * total_q_mae
end

"""
    parameterestimation_lossfunction(lf::mae, datasets, parameters,
                                     nucleationfunction, growthfunction,
                                     aggregationfunction, breakagefunction,
                                     solver::MoM) -> Real

Compute Mean Absolute Error loss for Method of Moments solver.

Uses d43 instead of d50q for particle size comparison.

# Arguments
- `lf::mae`: Loss function with weighting factors
- `datasets::Vector{<:AbstractMeasurements}`: Experimental data
- `parameters::AbstractVector{<:Real}`: Parameter vector to evaluate
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::MoM`: Method of Moments solver

# Returns
- Weighted sum of concentration MAE and particle size MAE
"""
function parameterestimation_lossfunction(lf::mae, datasets::Vector{<:AbstractMeasurements},
                                          parameters::AbstractVector{TPara},
                                          nucleationfunction::AbstractNucleationFunction,
                                          growthfunction::AbstractGrowthFunction,
                                          aggregationfunction::AbstractAggregationFunction,
                                          breakagefunction::AbstractBreakageFunction,
                                          solver::MoM) where {TPara <: Real}

    error_values = map(datasets) do dataset
        problem,
        solution = runsimulation(parameters,
                                 nucl = nucleationfunction,
                                 gr = growthfunction,
                                 agg = aggregationfunction,
                                 br = breakagefunction,
                                 initial_concentration = dataset.concentrationmean[1],
                                 save_idx = dataset.time,
                                 solver = solver,
                                 loading = dataset.loading,
                                 temp_profile = ConstantTemperature(dataset.temperature))

        if solution.success
            conc_errors = abs.(solution.concentration .- dataset.concentrationmean)
            q_errors = [abs(solution.d43[end] - dataset.d43)]
            (conc_errors, q_errors)
        else
            # Penalty values for failed simulations
            penalty_conc = fill(1e6, length(dataset.concentrationmean))
            penalty_q = [1e6]
            (penalty_conc, penalty_q)
        end
    end

    # Concatenate all errors into single vectors
    all_conc_errors = vcat(map(x -> x[1], error_values)...)
    all_q_errors = vcat(map(x -> x[2], error_values)...)

    total_conc_mae = mean(all_conc_errors)
    total_q_mae = mean(all_q_errors)

    return lf.weighting[1] * total_conc_mae + lf.weighting[2] * total_q_mae
end


"""
    parameterestimation_lossfunction(lf::logMLE_Indiana, datasets, parameters,
                                     nucleationfunction, growthfunction,
                                     aggregationfunction, breakagefunction,
                                     solver::AbstractDiscretisedSolver) -> Real

Compute log MLE loss with Indiana-style relative variance weighting for particle size.

Uses a percentage-based variance for particle size instead of measured variance.

# Arguments
- `lf::logMLE_Indiana`: Loss function with percentage-based size weighting
- `datasets::Vector{CrystallisationRepeatMeasurements}`: Experimental data
- `parameters::AbstractArray{<:Real}`: Parameter vector to evaluate
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::AbstractDiscretisedSolver`: Finite volume or WENO solver

# Returns
- Total weighted log-likelihood loss value
"""
function parameterestimation_lossfunction(lf::logMLE_Indiana,
                                          datasets::Vector{CrystallisationRepeatMeasurements},
                                          parameters::TArr,
                                          nucleationfunction::AbstractNucleationFunction,
                                          growthfunction::AbstractGrowthFunction,
                                          aggregationfunction::AbstractAggregationFunction,
                                          breakagefunction::AbstractBreakageFunction,
                                          solver::AbstractDiscretisedSolver) where {TArr <:
                                                                                    AbstractArray{<:Real}}
    loss_values = map(m -> begin
                          problem,
                          solution = runsimulation(parameters,
                                                   nucl = nucleationfunction,
                                                   gr = growthfunction,
                                                   agg = aggregationfunction,
                                                   br = breakagefunction,
                                                   initial_concentration = datasets[m].concentrationmean[1],
                                                   save_idx = datasets[m].time,
                                                   solver = solver,
                                                   loading = datasets[m].loading,
                                                   temp_profile = ConstantTemperature(datasets[m].temperature))

                          loss1 = sum(log.(2π *
                                           (datasets[m].concentrationvariance[2:end] .+
                                            1e-6)) .+
                                      ((solution.concentration[2:end] .-
                                        datasets[m].concentrationmean[2:end]) .^ 2) ./
                                      (datasets[m].concentrationvariance[2:end] .+ 1e-6))
                          loss2 = (log(2π *
                                       ((datasets[m].quantilemean * lf.weighting[2] * 1e-2)^2 .+
                                        1e-6)) +
                                   ((solution.d50q[end] .- datasets[m].quantilemean) .^ 2) ./
                                   ((datasets[m].quantilemean * lf.weighting[2] * 1e-2)^2 .+
                                    1e-6))
                          loss3 = sum(lf.weighting[1] * 0.5 * loss1 + 0.5 * loss2)

                          return (loss1, loss2, loss3)
                      end, eachindex(datasets))

    total_loss = sum(map(x -> x[3], loss_values))

    return total_loss
end
"""
    parameterestimation_lossfunction(lf::logMLE_Indiana, datasets, parameters,
                                     nucleationfunction, growthfunction,
                                     aggregationfunction, breakagefunction,
                                     solver::MoM) -> Real

Compute log MLE loss with Indiana-style weighting for MoM solver.

Uses d43 instead of d50q and percentage-based variance for particle size.

# Arguments
- `lf::logMLE_Indiana`: Loss function with percentage-based size weighting
- `datasets::Vector{CrystallisationRepeatMeasurements}`: Experimental data
- `parameters::AbstractVector{<:Real}`: Parameter vector to evaluate
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::MoM`: Method of Moments solver

# Returns
- Total weighted log-likelihood loss value
"""
function parameterestimation_lossfunction(lf::logMLE_Indiana,
                                          datasets::Vector{CrystallisationRepeatMeasurements},
                                          parameters::TArr,
                                          nucleationfunction::AbstractNucleationFunction,
                                          growthfunction::AbstractGrowthFunction,
                                          aggregationfunction::AbstractAggregationFunction,
                                          breakagefunction::AbstractBreakageFunction,
                                          solver::MoM) where {TArr <:
                                                              AbstractVector{<:Real}}

    loss_values = map(m -> begin
                          problem,
                          solution = runsimulation(parameters,
                                                   nucl = nucleationfunction,
                                                   gr = growthfunction,
                                                   agg = aggregationfunction,
                                                   br = breakagefunction,
                                                   initial_concentration = datasets[m].concentrationmean[1],
                                                   save_idx = datasets[m].time,
                                                   solver = solver,
                                                   loading = datasets[m].loading,
                                                   temp_profile = ConstantTemperature(datasets[m].temperature))

                          loss1 = sum(log.(2π *
                                           (datasets[m].concentrationvariance[2:end] .+
                                            1e-6)) .+
                                      ((solution.concentration[2:end] .-
                                        datasets[m].concentrationmean[2:end]) .^ 2) ./
                                      (datasets[m].concentrationvariance[2:end] .+ 1e-6))
                          loss2 = (log(2π *
                                       ((datasets[m].d43 * lf.weighting[2] * 1e-2)^2 .+
                                        1e-6)) +
                                   ((solution.d43[end] .- datasets[m].d43) .^ 2) ./
                                   ((datasets[m].d43 * lf.weighting[2] * 1e-2)^2 .+ 1e-6))
                          loss3 = sum(0.5 * loss1 + 1 / (lf.weighting[1]) * 0.5 * loss2)

                          return (loss1, loss2, loss3)
                      end, eachindex(datasets))

    total_loss = sum(map(x -> x[3], loss_values))

    return total_loss
end

"""
    parameterestimation_lossfunction(lf::logMLE_Han, datasets, parameters,
                                     nucleationfunction, growthfunction,
                                     aggregationfunction, breakagefunction,
                                     solver::AbstractDiscretisedSolver) -> Real

Compute log MLE loss with Han-style relative variance weighting for concentration.

Uses a percentage-based variance for concentration instead of measured variance.

# Arguments
- `lf::logMLE_Han`: Loss function with percentage-based concentration weighting
- `datasets::Vector{CrystallisationRepeatMeasurements}`: Experimental data
- `parameters::AbstractArray{<:Real}`: Parameter vector to evaluate
- `nucleationfunction`, `growthfunction`, `aggregationfunction`, `breakagefunction`: Kinetic models
- `solver::AbstractDiscretisedSolver`: Finite volume or WENO solver

# Returns
- Total weighted log-likelihood loss value
"""
function parameterestimation_lossfunction(lf::logMLE_Han,
                                          datasets::Vector{CrystallisationRepeatMeasurements},
                                          parameters::TArr,
                                          nucleationfunction::AbstractNucleationFunction,
                                          growthfunction::AbstractGrowthFunction,
                                          aggregationfunction::AbstractAggregationFunction,
                                          breakagefunction::AbstractBreakageFunction,
                                          solver::AbstractDiscretisedSolver) where {TArr <:
                                                                                    AbstractArray{<:Real}}
    loss_values = map(m -> begin
                          problem,
                          solution = runsimulation(parameters,
                                                   nucl = nucleationfunction,
                                                   gr = growthfunction,
                                                   agg = aggregationfunction,
                                                   br = breakagefunction,
                                                   initial_concentration = datasets[m].concentrationmean[1],
                                                   save_idx = datasets[m].time,
                                                   solver = solver,
                                                   temp_profile = ConstantTemperature(datasets[m].temperature))

                          newconcvariance = (datasets[m].concentrationmean .*
                                             lf.weighting[1] .* 1e-2) .^ 2
                          loss1 = sum(log.(2π * (newconcvariance .+ 1e-6)) .+
                                      ((solution.concentration .-
                                        datasets[m].concentrationmean) .^ 2) ./
                                      (newconcvariance .+ 1e-6))
                          loss2 = (log(2π * (datasets[m].quantilevariance .+ 1e-6)) +
                                   ((solution.d50q[end] .- datasets[m].quantilemean) .^ 2) ./
                                   (datasets[m].quantilevariance .+ 1e-6))
                          loss3 = sum(0.5 * loss1 + 0.5 * loss2)

                          return (loss1, loss2, loss3)
                      end, eachindex(datasets))

    total_loss = sum(map(x -> x[3], loss_values))

    return total_loss
end
