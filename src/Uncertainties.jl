"""
Tools for uncertainty quantification of crystallisation simulations. Provides ensemble simulation utilities and interfaces with Monte Carlo sampling.
"""

"""
    run_ensemble(distribution::D, measurements, nucleationfunction, growthfunction,
                 aggregationfunction, breakagefunction, solver;
                 n_samples::Int64=2048, HPC::Bool=false, verbosity::Int64=1,
                 time_idx::T=0:5:305, temp_profile=nothing,
                 initial_concentration=nothing, use_measurement_time::Bool=true)
                 -> Vector{Union{EnsembleFVSolution, EnsembleMoMSolution}}
                 where {D<:Distributions.Distribution, T<:AbstractArray}

Run ensemble simulations by sampling parameters from a distribution.

# Arguments
- `distribution::D`: Parameter distribution to sample from
- `measurements`: Vector of measurement objects defining experimental conditions
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::AbstractSolver`: Numerical solver (MoM, FiniteVol, or WENO)
- `n_samples::Int64=2048`: Number of parameter samples to draw
- `HPC::Bool=false`: Whether running on HPC (disables progress bar)
- `verbosity::Int64=1`: Verbosity level (0 = silent)
- `time_idx::T=0:5:305`: Time points for saving results
- `temp_profile=nothing`: Override temperature profile; defaults to measurement temperature
- `initial_concentration=nothing`: Override initial concentration; defaults to measurement value
- `use_measurement_time::Bool=true`: When true, infer time grid from measurements

# Returns
- `Vector{Union{EnsembleFVSolution, EnsembleMoMSolution}}`: Ensemble solutions for each measurement
"""
function run_ensemble(distribution::D,
                      measurements,
                      nucleationfunction::AbstractNucleationFunction,
                      growthfunction::AbstractGrowthFunction,
                      aggregationfunction::AbstractAggregationFunction,
                      breakagefunction::AbstractBreakageFunction,
                      solver::AbstractSolver;
                      n_samples::Int64 = 2048,
                      HPC::Bool = false,
                      verbosity::Int64 = 1,
                      time_idx::T = 0:5:305,
                      temp_profile = nothing,
                      initial_concentration = nothing,
                      use_measurement_time::Bool = true) where {D <:
                                                                Distributions.Distribution,
                                                                T <: AbstractArray}

    samples = rand(distribution, n_samples)

    return _run_ensemble_internal(samples, measurements, nucleationfunction, growthfunction,
                                  aggregationfunction, breakagefunction, solver;
                                  time_idx = time_idx,
                                  temp_profile = temp_profile,
                                  initial_concentration = initial_concentration,
                                  use_measurement_time = use_measurement_time,
                                  HPC = HPC,
                                  verbosity = verbosity)
end

"""
    run_ensemble(samples::Matrix{Float64}, measurements, nucleationfunction, growthfunction,
                 aggregationfunction, breakagefunction, solver;
                 HPC::Bool=false, verbosity::Int64=1, time_idx::T=0:5:305,
                 temp_profile=nothing, initial_concentration=nothing,
                 use_measurement_time::Bool=true)
                 -> Vector{Union{EnsembleFVSolution, EnsembleMoMSolution}}
                 where {T<:AbstractArray}

Run ensemble simulations from pre-sampled parameter matrix.

# Arguments
- `samples::Matrix{Float64}`: Parameter matrix of size (n_params, n_samples)
- `measurements`: Vector of measurement objects defining experimental conditions
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::AbstractSolver`: Numerical solver (MoM, FiniteVol, or WENO)
- `HPC::Bool=false`: Whether running on HPC (disables progress bar)
- `verbosity::Int64=1`: Verbosity level (0 = silent)
- `time_idx::T=0:5:305`: Time points for saving results
- `temp_profile=nothing`: Override temperature profile; defaults to measurement temperature
- `initial_concentration=nothing`: Override initial concentration; defaults to measurement value
- `use_measurement_time::Bool=true`: When true, infer time grid from measurements

# Returns
- `Vector{Union{EnsembleFVSolution, EnsembleMoMSolution}}`: Ensemble solutions for each measurement
"""
function run_ensemble(samples::Matrix{Float64},
                      measurements,
                      nucleationfunction::AbstractNucleationFunction,
                      growthfunction::AbstractGrowthFunction,
                      aggregationfunction::AbstractAggregationFunction,
                      breakagefunction::AbstractBreakageFunction,
                      solver::AbstractSolver;
                      HPC::Bool = false,
                      verbosity::Int64 = 1,
                      time_idx::T = 0:5:305,
                      temp_profile = nothing,
                      initial_concentration = nothing,
                      use_measurement_time::Bool = true) where {T <: AbstractArray}
    return _run_ensemble_internal(samples, measurements, nucleationfunction, growthfunction,
                                  aggregationfunction, breakagefunction, solver;
                                  time_idx = time_idx,
                                  temp_profile = temp_profile,
                                  initial_concentration = initial_concentration,
                                  use_measurement_time = use_measurement_time,
                                  HPC = HPC,
                                  verbosity = verbosity)
end

"""
    _create_ensemble_solution(time, concentration, d43, d32, d50q, solver::AbstractDiscretisedSolver) -> EnsembleFVSolution

Create ensemble solution object for discretised solvers (FiniteVol, WENO).

# Arguments
- `time`: Time vector
- `concentration`: Concentration matrix (n_time, n_samples)
- `d43`: D43 matrix (n_time, n_samples)
- `d32`: D32 matrix (n_time, n_samples)
- `d50q`: D50q matrix (n_time, n_samples)
- `solver::AbstractDiscretisedSolver`: Discretised solver instance

# Returns
- `EnsembleFVSolution`: Solution with mean, std, and 95% confidence bounds
"""
function _create_ensemble_solution(time, concentration, d43, d32, d50q,
                                   solver::AbstractDiscretisedSolver)
    concentration_mean = vec(mean(concentration, dims = 2))
    concentration_std = vec(std(concentration, dims = 2))
    z = 1.96
    concentration_lb = concentration_mean - z * concentration_std
    concentration_ub = concentration_mean + z * concentration_std
    d43_mean = vec(mean(d43, dims = 2))
    d43_std = vec(std(d43, dims = 2))
    d32_mean = vec(mean(d32, dims = 2))
    d32_std = vec(std(d32, dims = 2))
    d50q_mean = vec(mean(d50q, dims = 2))
    d50q_std = vec(std(d50q, dims = 2))

    return EnsembleFVSolution(collect(time), permutedims(concentration), permutedims(d43),
                              permutedims(d32), permutedims(d50q),
                              concentration_mean, concentration_std, concentration_lb,
                              concentration_ub,
                              d43_mean, d43_std, d32_mean, d32_std, d50q_mean, d50q_std)
end

"""
    _create_ensemble_solution(time, concentration, d43, d32, _, solver::MoM) -> EnsembleMoMSolution

Create ensemble solution object for Method of Moments solver.

# Arguments
- `time`: Time vector
- `concentration`: Concentration matrix (n_time, n_samples)
- `d43`: D43 matrix (n_time, n_samples)
- `d32`: D32 matrix (n_time, n_samples)
- `_`: Unused (d50q not available for MoM)
- `solver::MoM`: Method of Moments solver instance

# Returns
- `EnsembleMoMSolution`: Solution with mean, std, and 95% confidence bounds
"""
function _create_ensemble_solution(time, concentration, d43, d32, _, solver::MoM)
    concentration_mean = vec(mean(concentration, dims = 2))
    concentration_std = vec(std(concentration, dims = 2))
    concentration_lb = concentration_mean - 1.96 * concentration_std
    concentration_ub = concentration_mean + 1.96 * concentration_std
    d43_mean = vec(mean(d43, dims = 2))
    d43_std = vec(std(d43, dims = 2))
    d32_mean = vec(mean(d32, dims = 2))
    d32_std = vec(std(d32, dims = 2))

    return EnsembleMoMSolution(collect(time), permutedims(concentration), permutedims(d43),
                               permutedims(d32),
                               concentration_mean, concentration_std, concentration_lb,
                               concentration_ub,
                               d43_mean, d43_std, d32_mean, d32_std)
end

"""
    _run_ensemble_internal(samples::Matrix{Float64}, measurements::Vector{<:AbstractExperiment},
                           nucleationfunction, growthfunction, aggregationfunction, breakagefunction,
                           solver; time_idx::T=0:5:305, verbosity::Int64=1, HPC::Bool=false,
                           temp_profile=nothing, initial_concentration=nothing,
                           use_measurement_time::Bool=true)
                           -> Vector{Union{EnsembleFVSolution, EnsembleMoMSolution}}
                           where {T<:AbstractArray}

Internal function to run ensemble simulations across multiple measurements.

# Arguments
- `samples::Matrix{Float64}`: Parameter matrix of size (n_params, n_samples)
- `measurements::Vector{<:AbstractExperiment}`: Vector of measurement conditions
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::AbstractSolver`: Numerical solver
- `time_idx::T=0:5:305`: Time points for saving results
- `verbosity::Int64=1`: Verbosity level
- `HPC::Bool=false`: Whether running on HPC
- `temp_profile=nothing`: Override temperature profile; defaults to measurement temperature
- `initial_concentration=nothing`: Override initial concentration; defaults to measurement value
- `use_measurement_time::Bool=true`: When true, infer time grid from measurements

# Returns
- `Vector{Union{EnsembleFVSolution, EnsembleMoMSolution}}`: Ensemble solutions
"""
function _run_ensemble_internal(samples::Matrix{Float64},
                                measurements::Vector{<:AbstractExperiment},
                                nucleationfunction::AbstractNucleationFunction,
                                growthfunction::AbstractGrowthFunction,
                                aggregationfunction::AbstractAggregationFunction,
                                breakagefunction::AbstractBreakageFunction,
                                solver::AbstractSolver; time_idx::T = 0:5:305,
                                verbosity::Int64 = 1,
                                HPC::Bool = false,
                                temp_profile = nothing,
                                initial_concentration = nothing,
                                use_measurement_time::Bool = true) where {T <: AbstractArray}

    print_ensemble_diagnostics(samples; verbosity = verbosity)

    n_timepoints = length(time_idx)
    n_samples = size(samples, 2)
    n_measurements = length(measurements)

    ensemble_solutions = Vector{Union{EnsembleFVSolution, EnsembleMoMSolution}}(undef,
                                                                                n_measurements)

    prog = Progress(n_samples * n_measurements,
                    desc = "Simulating ensemble: ",
                    dt = 0.1,
                    showspeed = true,
                    barlen = 12,
                    enabled = (verbosity > 0 && !HPC))

    for m in 1:n_measurements
        time = if use_measurement_time
            range(measurements[m].observables.concentration.time[1], measurements[m].observables.concentration.time[end], n_timepoints)
        else
            time_idx
        end
        run_temp_profile = isnothing(temp_profile) ?
                           ConstantTemperature(measurements[m].temperature) :
                           temp_profile
        run_initial_concentration = isnothing(initial_concentration) ?
                                    initial_concentration(measurements[m]) :
                                    initial_concentration
        concentration = Matrix{Float64}(undef, n_timepoints, n_samples)
        d43 = Matrix{Float64}(undef, n_timepoints, n_samples)
        d32 = Matrix{Float64}(undef, n_timepoints, n_samples)
        d50q = _d50q_buffer(solver, n_timepoints, n_samples)

        @floop for i in 1:n_samples
            prob,
            sol = runsimulation(samples[:, i],
                                nucl = nucleationfunction,
                                gr = growthfunction,
                                agg = aggregationfunction,
                                br = breakagefunction,
                                initial_concentration = run_initial_concentration,
                                save_idx = time,
                                solver = solver,
                                temp_profile = run_temp_profile,
                                loading = measurements[m].loading)

            concentration[:, i] = sol.concentration
            d43[:, i] = sol.d43
            d32[:, i] = sol.d32
            _store_d50q!(d50q, sol, i, solver)
            next!(prog)
        end

        ensemble_solutions[m] = _create_ensemble_solution(time, concentration, d43, d32,
                                                          d50q, solver)
    end
    return ensemble_solutions
end


"""
    _d50q_buffer(solver, n_timepoints, n_samples)

d50q accumulation buffer: `nothing` for the MoM solver (no quantiles),
a `Matrix{Float64}` for discretised solvers (solver-type dispatch).
"""
_d50q_buffer(solver::MoM, n_timepoints, n_samples) = nothing
_d50q_buffer(solver::AbstractDiscretisedSolver, n_timepoints, n_samples) =
    Matrix{Float64}(undef, n_timepoints, n_samples)

"""
    _store_d50q!(buffer, sol, i, solver)

Store the i-th d50q trajectory into the buffer (no-op for MoM).
"""
_store_d50q!(buffer, sol, i, solver::MoM) = nothing
_store_d50q!(buffer::AbstractMatrix, sol, i, solver::AbstractDiscretisedSolver) =
    (buffer[:, i] = sol.d50q; nothing)

"""
    run_ensemble_fixed(samples::Matrix{Float64},
                       nucleationfunction, growthfunction, aggregationfunction,
                       breakagefunction, solver;
                       time_idx::T=0:5:305, temp_profile,
                       initial_concentration, loading::Float64=0.0,
                       verbosity::Int64=1, HPC::Bool=false)
                       -> Union{EnsembleFVSolution, EnsembleMoMSolution}
                       where {T<:AbstractArray}

Run ensemble simulations without measurement objects by providing fixed inputs.

# Arguments
- `samples::Matrix{Float64}`: Parameter matrix of size (n_params, n_samples)
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::AbstractSolver`: Numerical solver
- `time_idx::T=0:5:305`: Time points for saving results
- `temp_profile`: Temperature profile (required)
- `initial_concentration`: Initial concentration (required)
- `loading::Float64=0.0`: Fixed loading value
- `verbosity::Int64=1`: Verbosity level (0 = silent)
- `HPC::Bool=false`: Whether running on HPC

# Returns
- `Union{EnsembleFVSolution, EnsembleMoMSolution}`: Ensemble solution
"""
function run_ensemble_fixed(samples::Matrix{Float64},
                            nucleationfunction::AbstractNucleationFunction,
                            growthfunction::AbstractGrowthFunction,
                            aggregationfunction::AbstractAggregationFunction,
                            breakagefunction::AbstractBreakageFunction,
                            solver::AbstractSolver;
                            time_idx::T = 0:5:305,
                            temp_profile = nothing,
                            initial_concentration = nothing,
                            loading::Float64 = 0.0,
                            verbosity::Int64 = 1,
                            HPC::Bool = false) where {T <: AbstractArray}

    isnothing(temp_profile) &&
        error("run_ensemble_fixed requires temp_profile (no measurement default).")
    isnothing(initial_concentration) &&
        error("run_ensemble_fixed requires initial_concentration (no measurement default).")

    print_ensemble_diagnostics(samples; verbosity = verbosity)

    n_timepoints = length(time_idx)
    n_samples = size(samples, 2)

    concentration = Matrix{Float64}(undef, n_timepoints, n_samples)
    d43 = Matrix{Float64}(undef, n_timepoints, n_samples)
    d32 = Matrix{Float64}(undef, n_timepoints, n_samples)
    d50q = _d50q_buffer(solver, n_timepoints, n_samples)

    prog = Progress(n_samples,
                    desc = "Simulating ensemble: ",
                    dt = 0.1,
                    showspeed = true,
                    barlen = 12,
                    enabled = (verbosity > 0 && !HPC))

    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(samples[:, i],
                            nucl = nucleationfunction,
                            gr = growthfunction,
                            agg = aggregationfunction,
                            br = breakagefunction,
                            initial_concentration = initial_concentration,
                            save_idx = time_idx,
                            solver = solver,
                            temp_profile = temp_profile,
                            loading = loading)

        concentration[:, i] = sol.concentration
        d43[:, i] = sol.d43
        d32[:, i] = sol.d32
        _store_d50q!(d50q, sol, i, solver)
        next!(prog)
    end

    return _create_ensemble_solution(time_idx, concentration, d43, d32, d50q, solver)
end

"""
    _simulateensembleuncertainty(samples::Matrix{Float64}, c_array::Vector{Float64},
                                 nucleationfunction, growthfunction, aggregationfunction,
                                 breakagefunction, solver::AbstractDiscretisedSolver;
                                 time_idx::T=0:5:305, HPC::Bool=false) -> Vector{EnsembleFVSolution} where {T<:AbstractArray}

Simulate ensemble uncertainty for multiple initial concentrations using discretised solver.

# Arguments
- `samples::Matrix{Float64}`: Parameter matrix of size (n_params, n_samples)
- `c_array::Vector{Float64}`: Vector of initial concentrations
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::AbstractDiscretisedSolver`: Discretised solver (FiniteVol or WENO)
- `time_idx::T=0:5:305`: Time points for saving results
- `HPC::Bool=false`: Whether running on HPC

# Returns
- `Vector{EnsembleFVSolution}`: Ensemble solutions for each concentration
"""
function _simulateensembleuncertainty(samples::Matrix{Float64}, c_array::Vector{Float64},
                                      nucleationfunction::AbstractNucleationFunction,
                                      growthfunction::AbstractGrowthFunction,
                                      aggregationfunction::AbstractAggregationFunction,
                                      breakagefunction::AbstractBreakageFunction,
                                      solver::AbstractDiscretisedSolver;
                                      time_idx::T = 0:5:305,
                                      HPC::Bool = false) where {T <: AbstractArray}

    n_samples = size(samples, 2)
    n_measurements = length(c_array)
    n_timepoints = length(time_idx)
    ensemble_solutions = Vector{EnsembleFVSolution}(undef, n_measurements)

    prog = Progress(n_samples * n_measurements,
                    desc = "Simulating ensemble: ",
                    dt = 0.1,
                    showspeed = true,
                    barlen = 12,
                    enabled = !HPC)

    for m in 1:n_measurements
        # Pre-allocate matrices for this measurement
        concentration = Matrix{Float64}(undef, n_timepoints, n_samples)
        d43 = Matrix{Float64}(undef, n_timepoints, n_samples)
        d32 = Matrix{Float64}(undef, n_timepoints, n_samples)
        d50q = Matrix{Float64}(undef, n_timepoints, n_samples)

        # Run simulations for each sample
        @floop for i in 1:n_samples
            prob,
            sol = runsimulation(samples[:, i],
                                nucl = nucleationfunction,
                                gr = growthfunction,
                                agg = aggregationfunction,
                                br = breakagefunction,
                                initial_concentration = c_array[m],
                                save_idx = time_idx,
                                solver = solver)

            concentration[:, i] = sol.concentration
            d43[:, i] = sol.d43
            d32[:, i] = sol.d32
            d50q[:, i] = sol.d50q

            next!(prog)
        end

        # Compute statistics
        concentration_mean = vec(mean(concentration, dims = 2))
        concentration_std = vec(std(concentration, dims = 2))

        # Compute confidence bounds (95% confidence interval)
        z = 1.96
        concentration_lb = concentration_mean - z * concentration_std
        concentration_ub = concentration_mean + z * concentration_std

        # Compute diameter statistics
        d43_mean = vec(mean(d43, dims = 2))
        d43_std = vec(std(d43, dims = 2))
        d32_mean = vec(mean(d32, dims = 2))
        d32_std = vec(std(d32, dims = 2))
        d50q_mean = vec(mean(d50q, dims = 2))
        d50q_std = vec(std(d50q, dims = 2))

        # Create EnsembleFVSolution for this measurement
        ensemble_solutions[m] = EnsembleFVSolution(time_idx,
                                                   permutedims(concentration),
                                                   permutedims(d43),
                                                   permutedims(d32),
                                                   permutedims(d50q),
                                                   concentration_mean,
                                                   concentration_std,
                                                   concentration_lb,
                                                   concentration_ub,
                                                   d43_mean,
                                                   d43_std,
                                                   d32_mean,
                                                   d32_std,
                                                   d50q_mean,
                                                   d50q_std)
    end

    return ensemble_solutions
end

"""
    _simulateensembleuncertainty(samples::Matrix{Float64}, c_array::Vector{Float64},
                                 nucleationfunction, growthfunction, aggregationfunction,
                                 breakagefunction, solver::MoM;
                                 time_idx::T=0:5:305, HPC::Bool=false) -> Vector{EnsembleMoMSolution} where {T<:AbstractArray}

Simulate ensemble uncertainty for multiple initial concentrations using MoM solver.

# Arguments
- `samples::Matrix{Float64}`: Parameter matrix of size (n_params, n_samples)
- `c_array::Vector{Float64}`: Vector of initial concentrations
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::MoM`: Method of Moments solver
- `time_idx::T=0:5:305`: Time points for saving results
- `HPC::Bool=false`: Whether running on HPC

# Returns
- `Vector{EnsembleMoMSolution}`: Ensemble solutions for each concentration
"""
function _simulateensembleuncertainty(samples::Matrix{Float64}, c_array::Vector{Float64},
                                      nucleationfunction::AbstractNucleationFunction,
                                      growthfunction::AbstractGrowthFunction,
                                      aggregationfunction::AbstractAggregationFunction,
                                      breakagefunction::AbstractBreakageFunction,
                                      solver::MoM; time_idx::T = 0:5:305,
                                      HPC::Bool = false) where {T <: AbstractArray}

    n_samples = size(samples, 2)
    n_measurements = length(c_array)
    n_timepoints = length(time_idx)
    ensemble_solutions = Vector{EnsembleMoMSolution}(undef, n_measurements)

    prog = Progress(n_samples * n_measurements,
                    desc = "Simulating ensemble: ",
                    dt = 0.1,
                    showspeed = true,
                    barlen = 12,
                    enabled = !HPC)

    for m in 1:n_measurements
        # Pre-allocate matrices for this measurement
        concentration = Matrix{Float64}(undef, n_timepoints, n_samples)
        d43 = Matrix{Float64}(undef, n_timepoints, n_samples)
        d32 = Matrix{Float64}(undef, n_timepoints, n_samples)

        # Run simulations for each sample
        @floop for i in 1:n_samples
            prob,
            sol = runsimulation(samples[:, i],
                                nucleationfunction,
                                growthfunction,
                                aggregationfunction,
                                breakagefunction,
                                c_array[m],
                                save_idx = time_idx,
                                solver = solver)

            concentration[:, i] = sol.concentration
            d43[:, i] = sol.d43
            d32[:, i] = sol.d32

            next!(prog)
        end

        # Compute statistics
        concentration_mean = vec(mean(concentration, dims = 2))
        concentration_std = vec(std(concentration, dims = 2))

        # Compute confidence bounds (95% confidence interval)
        z = 1.96
        concentration_lb = concentration_mean - z * concentration_std
        concentration_ub = concentration_mean + z * concentration_std

        # Compute diameter statistics
        d43_mean = vec(mean(d43, dims = 2))
        d43_std = vec(std(d43, dims = 2))
        d32_mean = vec(mean(d32, dims = 2))
        d32_std = vec(std(d32, dims = 2))

        # Create EnsembleFVSolution for this measurement
        ensemble_solutions[m] = EnsembleMoMSolution(time_idx,
                                                    permutedims(concentration),
                                                    permutedims(d43),
                                                    permutedims(d32),
                                                    concentration_mean,
                                                    concentration_std,
                                                    concentration_lb,
                                                    concentration_ub,
                                                    d43_mean,
                                                    d43_std,
                                                    d32_mean,
                                                    d32_std)
    end

    return ensemble_solutions
end

"""
    _simulateensembleuncertainty(samples::Matrix{Float64}, conc::Real,
                                 nucleationfunction, growthfunction, aggregationfunction,
                                 breakagefunction, solver::AbstractDiscretisedSolver;
                                 time_idx::T=0:5:305, HPC::Bool=false) -> EnsembleFVSolution where {T<:AbstractArray}

Simulate ensemble uncertainty for a single initial concentration using discretised solver.

# Arguments
- `samples::Matrix{Float64}`: Parameter matrix of size (n_params, n_samples)
- `conc::Real`: Initial concentration
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::AbstractDiscretisedSolver`: Discretised solver (FiniteVol or WENO)
- `time_idx::T=0:5:305`: Time points for saving results
- `HPC::Bool=false`: Whether running on HPC

# Returns
- `EnsembleFVSolution`: Ensemble solution with statistics
"""
function _simulateensembleuncertainty(samples::Matrix{Float64}, conc::Real,
                                      nucleationfunction::AbstractNucleationFunction,
                                      growthfunction::AbstractGrowthFunction,
                                      aggregationfunction::AbstractAggregationFunction,
                                      breakagefunction::AbstractBreakageFunction,
                                      solver::AbstractDiscretisedSolver;
                                      time_idx::T = 0:5:305,
                                      HPC::Bool = false) where {T <: AbstractArray}

    n_samples = size(samples, 2)
    n_timepoints = length(time_idx)

    prog = Progress(n_samples,
                    desc = "Simulating ensemble: ",
                    dt = 0.1,
                    showspeed = true,
                    barlen = 12,
                    enabled = !HPC)

    # Pre-allocate matrices for this measurement
    concentration = Matrix{Float64}(undef, n_timepoints, n_samples)
    d43 = Matrix{Float64}(undef, n_timepoints, n_samples)
    d32 = Matrix{Float64}(undef, n_timepoints, n_samples)
    d50q = Matrix{Float64}(undef, n_timepoints, n_samples)

    # Run simulations for each sample
    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(samples[:, i],
                            nucleationfunction,
                            growthfunction,
                            aggregationfunction,
                            breakagefunction,
                            conc,
                            save_idx = time_idx,
                            solver = solver)

        concentration[:, i] = sol.concentration
        d43[:, i] = sol.d43
        d32[:, i] = sol.d32
        d50q[:, i] = sol.d50q

        next!(prog)
    end

    # Compute statistics
    concentration_mean = vec(mean(concentration, dims = 2))
    concentration_std = vec(std(concentration, dims = 2))

    # Compute confidence bounds (95% confidence interval)
    z = 1.96
    concentration_lb = concentration_mean - z * concentration_std
    concentration_ub = concentration_mean + z * concentration_std

    # Compute diameter statistics
    d43_mean = vec(mean(d43, dims = 2))
    d43_std = vec(std(d43, dims = 2))
    d32_mean = vec(mean(d32, dims = 2))
    d32_std = vec(std(d32, dims = 2))
    d50q_mean = vec(mean(d50q, dims = 2))
    d50q_std = vec(std(d50q, dims = 2))

    # Create EnsembleFVSolution for this measurement
    return EnsembleFVSolution(time_idx,
                              permutedims(concentration),
                              permutedims(d43),
                              permutedims(d32),
                              permutedims(d50q),
                              concentration_mean,
                              concentration_std,
                              concentration_lb,
                              concentration_ub,
                              d43_mean,
                              d43_std,
                              d32_mean,
                              d32_std,
                              d50q_mean,
                              d50q_std)



end

"""
    _simulateensembleuncertainty(samples::Matrix{Float64}, conc::Real,
                                 nucleationfunction, growthfunction, aggregationfunction,
                                 breakagefunction, solver::MoM;
                                 time_idx::T=0:5:305, HPC::Bool=false) -> EnsembleMoMSolution where {T<:AbstractArray}

Simulate ensemble uncertainty for a single initial concentration using MoM solver.

# Arguments
- `samples::Matrix{Float64}`: Parameter matrix of size (n_params, n_samples)
- `conc::Real`: Initial concentration
- `nucleationfunction::AbstractNucleationFunction`: Nucleation kinetic model
- `growthfunction::AbstractGrowthFunction`: Growth kinetic model
- `aggregationfunction::AbstractAggregationFunction`: Aggregation kinetic model
- `breakagefunction::AbstractBreakageFunction`: Breakage kinetic model
- `solver::MoM`: Method of Moments solver
- `time_idx::T=0:5:305`: Time points for saving results
- `HPC::Bool=false`: Whether running on HPC

# Returns
- `EnsembleMoMSolution`: Ensemble solution with statistics
"""
function _simulateensembleuncertainty(samples::Matrix{Float64}, conc::Real,
                                      nucleationfunction::AbstractNucleationFunction,
                                      growthfunction::AbstractGrowthFunction,
                                      aggregationfunction::AbstractAggregationFunction,
                                      breakagefunction::AbstractBreakageFunction,
                                      solver::MoM; time_idx::T = 0:5:305,
                                      HPC::Bool = false) where {T <: AbstractArray}

    n_samples = size(samples, 2)
    n_timepoints = length(time_idx)

    prog = Progress(n_samples,
                    desc = "Simulating ensemble: ",
                    dt = 0.1,
                    showspeed = true,
                    barlen = 12,
                    enabled = !HPC)

    # Pre-allocate matrices for this measurement
    concentration = Matrix{Float64}(undef, n_timepoints, n_samples)
    d43 = Matrix{Float64}(undef, n_timepoints, n_samples)
    d32 = Matrix{Float64}(undef, n_timepoints, n_samples)

    # Run simulations for each sample
    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(samples[:, i],
                            nucleationfunction,
                            growthfunction,
                            aggregationfunction,
                            breakagefunction,
                            conc,
                            save_idx = time_idx,
                            solver = solver)

        concentration[:, i] = sol.concentration
        d43[:, i] = sol.d43
        d32[:, i] = sol.d32

        next!(prog)
    end

    # Compute statistics
    concentration_mean = vec(mean(concentration, dims = 2))
    concentration_std = vec(std(concentration, dims = 2))

    # Compute confidence bounds (95% confidence interval)
    z = 1.96
    concentration_lb = concentration_mean - z * concentration_std
    concentration_ub = concentration_mean + z * concentration_std

    # Compute diameter statistics
    d43_mean = vec(mean(d43, dims = 2))
    d43_std = vec(std(d43, dims = 2))
    d32_mean = vec(mean(d32, dims = 2))
    d32_std = vec(std(d32, dims = 2))

    # Create EnsembleMoMSolution for this measurement
    return EnsembleMoMSolution(time_idx,
                               permutedims(concentration),
                               permutedims(d43),
                               permutedims(d32),
                               concentration_mean,
                               concentration_std,
                               concentration_lb,
                               concentration_ub,
                               d43_mean,
                               d43_std,
                               d32_mean,
                               d32_std)

end
