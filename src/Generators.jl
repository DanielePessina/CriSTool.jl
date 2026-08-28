
"""
Utilities for generating parameter samples and design spaces. Runs batches of crystallisation simulations to build surrogate datasets.
"""

"""
    samplespace(n_samples, samplingmethod, lb, ub,
                nucleationfunction, growthfunction,
                aggregationfunction, breakagefunction, solver;
                tspan = (0.0, 460.0), save_every = 5.0,
                xlsx = true, HPC = false, filestr = "",
                kwargs...)

Generate a quasi-random sample of kinetic parameters and run
crystallisation simulations for each sample.

# Arguments
- `n_samples::Int`: number of parameter vectors to evaluate.
- `samplingmethod`: a `QuasiMonteCarlo.SamplingAlgorithm` such as
  `SobolSeq()`.
- `lb`, `ub`: lower and upper bounds for each parameter. Both vectors
  must have length `nν + n_g + n_a + n_b + 1` corresponding to
  `[p_ν; p_g; p_agg; p_br; c₀]` where `c₀` is the initial concentration.
- `solver::AbstractDiscretisedSolver`: numerical solver to use.
- `tspan`, `save_every`: simulation time span (minutes) and saving
  interval. The number of stored time points is
  `Int(floor(tspan[2] / save_every)) + 1`.
- `xlsx`: if `true`, results are written to an Excel file.
- `HPC`: disable progress output when running on a cluster.
- `filestr`: additional string appended to the output filename.
- `kwargs`: forwarded to [`runsimulation`](@ref).

# Returns
A named tuple containing
`(samples, conc_mat, d50matrix, d10matrix, d32matrix, d43matrix)` where
each matrix has size `(n_samples, n_tsteps)` with `n_tsteps` equal to the
number of saved time points.
"""
function samplespace(n_samples::Int64, samplingmethod::QuasiMonteCarlo.SamplingAlgorithm,
                     lb::AbstractArray{Float64}, ub::AbstractArray{Float64},
                     nucleationfunction::AbstractNucleationFunction,
                     growthfunction::AbstractGrowthFunction,
                     aggregationfunction::AbstractAggregationFunction,
                     breakagefunction::AbstractBreakageFunction,
                     solver::AbstractDiscretisedSolver;
                     tspan::Tuple{Float64, Float64} = (0.0, 460.0),
                     save_every::Float64 = 5.0, xlsx::Bool = true, HPC::Bool = false,
                     filestr::String = "", kwargs...)

    if nucleationfunction.nparams + growthfunction.nparams + 1 != length(lb)
        throw(ArgumentError("The number of parameters in the nucleation and growth functions + initial concentration must match the length of the lower bounds vector"))
    end

    samples = QuasiMonteCarlo.sample(n_samples, lb, ub, samplingmethod)

    conc_mat = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d50matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d10matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d32matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d43matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    prog = Progress(n_samples, desc = "Sampling across input parameters: ", dt = 0.1,
                    showspeed = true, enabled = !HPC, barlen = 15)

    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(samples[1:(end - 1), i];
                            nucl = nucleationfunction,
                            gr = growthfunction,
                            agg = aggregationfunction,
                            br = breakagefunction,
                            initial_concentration = samples[end, i],
                            save_idx = tspan[1]:save_every:tspan[2],
                            solver = solver,
                            kwargs...)
        #

        conc_mat[i, :] = sol.concentration

        d50matrix[i, :] = sol.d50q
        d10matrix[i, :] = sol.d10q

        momentsizes = getmomentsizes(prob, sol)

        d32matrix[i, :] = momentsizes.d32

        d43matrix[i, :] = momentsizes.d43

        next!(prog)

    end

    ## save

    printstr = "Lower bounds: $(lb) \nUpper bounds: $(ub)
                \nNucleation function: $(nucleationfunction.string) \nGrowth function: $(growthfunction.string),
                \nTime span: $(tspan)
                \nSave every: $(save_every)
                \nSolver: $(solver.string)"

    now_str = Dates.format(now(), "yy-m-d HH-MM")

    if xlsx
        _savegeneratoroutput(samples, conc_mat, d50matrix, d10matrix, d32matrix, d43matrix,
                             "$now_str $filestr", printstr)
    end

    return (samples = samples, conc_mat = conc_mat, d50matrix = d50matrix,
            d10matrix = d10matrix, d32matrix = d32matrix, d43matrix = d43matrix)

end
"""
    samplespace(n_samples, samplingmethod, lb, ub,
                nucleationfunction, growthfunction,
                aggregationfunction, breakagefunction, solver::MoM;
                tspan = (0.0, 460.0), save_every = 5.0,
                xlsx = true, HPC = false, filestr = "",
                kwargs...)

Identical to the general [`samplespace`](@ref) but using a method of
moments solver. The output matrices contain moment-based size metrics
`d10`, `d32` and `d43` instead of quantiles.
"""
function samplespace(n_samples::Int64, samplingmethod::QuasiMonteCarlo.SamplingAlgorithm,
                     lb::AbstractArray{Float64}, ub::AbstractArray{Float64},
                     nucleationfunction::AbstractNucleationFunction,
                     growthfunction::AbstractGrowthFunction,
                     aggregationfunction::AbstractAggregationFunction,
                     breakagefunction::AbstractBreakageFunction, solver::MoM;
                     tspan::Tuple{Float64, Float64} = (0.0, 460.0),
                     save_every::Float64 = 5.0, xlsx::Bool = true, HPC::Bool = false,
                     filestr::String = "", kwargs...)

    if nucleationfunction.nparams + growthfunction.nparams + 1 != length(lb)
        throw(ArgumentError("The number of parameters in the nucleation and growth functions + initial concentration must match the length of the lower bounds vector"))
    end

    samples = QuasiMonteCarlo.sample(n_samples, lb, ub, samplingmethod)

    conc_mat = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d10matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d32matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d43matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    prog = Progress(n_samples, desc = "Sampling across input parameters: ", dt = 0.1,
                    showspeed = true, enabled = !HPC, barlen = 15)

    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(samples[1:(end - 1), i];
                            nucl = nucleationfunction,
                            gr = growthfunction,
                            agg = aggregationfunction,
                            br = breakagefunction,
                            initial_concentration = samples[end, i],
                            save_idx = tspan[1]:save_every:tspan[2],
                            solver = solver,
                            kwargs...)
        #

        conc_mat[i, :] = sol.concentration

        d10matrix[i, :] = sol.d10

        d32matrix[i, :] = sol.d32

        d43matrix[i, :] = sol.d43

        next!(prog)

    end

    ## save

    printstr = "Lower bounds: $(lb) \nUpper bounds: $(ub)
                \nNucleation function: $(nucleationfunction.string) \nGrowth function: $(growthfunction.string),
                \nTime span: $(tspan)
                \nSave every: $(save_every)
                \nSolver: $(solver.string)"

    now_str = Dates.format(now(), "yy-m-d HH-MM")

    if xlsx
        _savegeneratoroutput(samples, conc_mat, d10matrix, d32matrix, d43matrix,
                             "$now_str $filestr", printstr)
    end

    return (samples = samples, conc_mat = conc_mat, d10matrix = d10matrix,
            d32matrix = d32matrix, d43matrix = d43matrix)

end

## with initial concentration
"""
    samplespace(n_samples::Int64, samplingmethod::QuasiMonteCarlo.SamplingAlgorithm,
    lb::AbstractArray{Float64}, ub::AbstractArray{Float64},
    nucleationfunction::AbstractNucleationFunction, growthfunction::AbstractGrowthFunction, aggregationfunction::AbstractAggregationFunction, breakagefunction::AbstractBreakageFunction, initialconcentration::Float64, solver::AbstractDiscretisedSolver; tspan::Tuple{Float64,Float64}=(0.0, 460.0), save_every::Float64=5.0, xlsx::Bool=true, HPC::Bool=false, filestr::String="", kwargs...)

"""
function samplespace(n_samples::Int64, samplingmethod::QuasiMonteCarlo.SamplingAlgorithm,
                     lb::AbstractArray{Float64}, ub::AbstractArray{Float64},
                     nucleationfunction::AbstractNucleationFunction,
                     growthfunction::AbstractGrowthFunction,
                     aggregationfunction::AbstractAggregationFunction,
                     breakagefunction::AbstractBreakageFunction,
                     initialconcentration::Float64, solver::AbstractDiscretisedSolver;
                     tspan::Tuple{Float64, Float64} = (0.0, 460.0),
                     save_every::Float64 = 5.0, xlsx::Bool = true, HPC::Bool = false,
                     filestr::String = "", kwargs...)

    if nucleationfunction.nparams + growthfunction.nparams != length(lb)
        throw(ArgumentError("The number of parameters in the nucleation and growth functions + initial concentration must match the length of the lower bounds vector"))
    end

    samples = QuasiMonteCarlo.sample(n_samples, lb, ub, samplingmethod)

    conc_mat = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d50matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d10matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d32matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d43matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    prog = Progress(n_samples, desc = "Sampling across input parameters: ", dt = 0.1,
                    showspeed = true, enabled = !HPC, barlen = 15)

    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(samples[:, i];
                            nucl = nucleationfunction,
                            gr = growthfunction,
                            agg = aggregationfunction,
                            br = breakagefunction,
                            initial_concentration = initialconcentration,
                            save_idx = tspan[1]:save_every:tspan[2],
                            solver = solver,
                            kwargs...)
        #

        conc_mat[i, :] = sol.concentration

        d50matrix[i, :] = sol.d50q
        d10matrix[i, :] = sol.d10q

        momentsizes = getmomentsizes(prob, sol)

        d32matrix[i, :] = momentsizes.d32

        d43matrix[i, :] = momentsizes.d43

        next!(prog)

    end

    ## save

    printstr = "Lower bounds: $(lb) \nUpper bounds: $(ub)
                \nNucleation function: $(nucleationfunction.string) \nGrowth function: $(growthfunction.string)
                \nInitial concentration: $(initialconcentration)
                \nTime span: $(tspan)
                \nSave every: $(save_every)
                \nSolver: $(solver.string)"

    now_str = Dates.format(now(), "yy-m-d HH-MM")

    if xlsx
        _savegeneratoroutput(samples, conc_mat, d50matrix, d10matrix, d32matrix, d43matrix,
                             "$now_str $filestr", printstr)
    end

    return (samples = samples, conc_mat = conc_mat, d50matrix = d50matrix,
            d10matrix = d10matrix, d32matrix = d32matrix, d43matrix = d43matrix)

end

"""
    samplespace(n_samples::Int64, samplingmethod::QuasiMonteCarlo.SamplingAlgorithm,
    lb::AbstractArray{Float64}, ub::AbstractArray{Float64},
    nucleationfunction::AbstractNucleationFunction, growthfunction::AbstractGrowthFunction, aggregationfunction::AbstractAggregationFunction, breakagefunction::AbstractBreakageFunction, initialconcentration::Float64, solver::MoM; tspan::Tuple{Float64,Float64}=(0.0, 460.0), save_every::Float64=5.0, xlsx::Bool=true, HPC::Bool=false, filestr::String="", kwargs...)

"""
function samplespace(n_samples::Int64, samplingmethod::QuasiMonteCarlo.SamplingAlgorithm,
                     lb::AbstractArray{Float64}, ub::AbstractArray{Float64},
                     nucleationfunction::AbstractNucleationFunction,
                     growthfunction::AbstractGrowthFunction,
                     aggregationfunction::AbstractAggregationFunction,
                     breakagefunction::AbstractBreakageFunction,
                     initialconcentration::Float64, solver::MoM;
                     tspan::Tuple{Float64, Float64} = (0.0, 460.0),
                     save_every::Float64 = 5.0, xlsx::Bool = true, HPC::Bool = false,
                     filestr::String = "", kwargs...)

    if nucleationfunction.nparams + growthfunction.nparams != length(lb)
        throw(ArgumentError("The number of parameters in the nucleation and growth functions + initial concentration must match the length of the lower bounds vector"))
    end

    samples = QuasiMonteCarlo.sample(n_samples, lb, ub, samplingmethod)

    conc_mat = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d10matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d32matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d43matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    prog = Progress(n_samples, desc = "Sampling across input parameters: ", dt = 0.1,
                    showspeed = true, enabled = !HPC, barlen = 15)

    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(samples[:, i];
                            nucl = nucleationfunction,
                            gr = growthfunction,
                            agg = aggregationfunction,
                            br = breakagefunction,
                            initial_concentration = initialconcentration,
                            save_idx = tspan[1]:save_every:tspan[2],
                            solver = solver,
                            kwargs...)
        #

        conc_mat[i, :] = sol.concentration

        d10matrix[i, :] = sol.d10

        d32matrix[i, :] = sol.d32

        d43matrix[i, :] = sol.d43

        next!(prog)

    end

    ## save

    printstr = "Lower bounds: $(lb) \nUpper bounds: $(ub)
                \nNucleation function: $(nucleationfunction.string) \nGrowth function: $(growthfunction.string)
                \nInitial concentration: $(initialconcentration)
                \nTime span: $(tspan)
                \nSave every: $(save_every)
                \nSolver: $(solver.string)"

    now_str = Dates.format(now(), "yy-m-d HH-MM")

    if xlsx
        _savegeneratoroutput(samples, conc_mat, d10matrix, d32matrix, d43matrix,
                             "$now_str $filestr", printstr)
    end

    return (samples = samples, conc_mat = conc_mat, d10matrix = d10matrix,
            d32matrix = d32matrix, d43matrix = d43matrix)

end



function _savegeneratoroutput(samples::AbstractMatrix{Float64},
                              conc_mat::AbstractMatrix{Float64},
                              d50qmatrix::AbstractMatrix{Float64},
                              d10matrix::AbstractMatrix{Float64},
                              d32matrix::AbstractMatrix{Float64},
                              d43matrix::AbstractMatrix{Float64}, savestring::String,
                              printstr::String)

    open("Generated Datasets/$savestring Info.txt", "w") do io
        write(io, printstr)
    end

    df_samples = DataFrame(permutedims(samples), :auto)

    df_concentration = DataFrame((conc_mat), :auto)

    df_d50 = DataFrame((d50qmatrix), :auto)

    df_d10 = DataFrame((d10matrix), :auto)

    df_d32 = DataFrame((d32matrix), :auto)

    df_d43 = DataFrame((d43matrix), :auto)

    XLSX.openxlsx("Generated Datasets/$savestring.xlsx", mode = "w") do xf

        XLSX.rename!(xf[1], "Samples")
        XLSX.addsheet!(xf, "Concentration")
        XLSX.addsheet!(xf, "D50q")
        XLSX.addsheet!(xf, "D10")
        XLSX.addsheet!(xf, "D32")
        XLSX.addsheet!(xf, "D43")


        XLSX.writetable!(xf[1], df_samples)
        XLSX.writetable!(xf[2], df_concentration)
        XLSX.writetable!(xf[3], df_d50)
        XLSX.writetable!(xf[4], df_d10)
        XLSX.writetable!(xf[5], df_d32)
        XLSX.writetable!(xf[6], df_d43)

    end

end

function _savegeneratoroutput(samples::AbstractMatrix{Float64},
                              conc_mat::AbstractMatrix{Float64},
                              d10matrix::AbstractMatrix{Float64},
                              d32matrix::AbstractMatrix{Float64},
                              d43matrix::AbstractMatrix{Float64}, savestring::String,
                              printstr::String)

    open("Generated Datasets/$savestring Info.txt", "w") do io
        write(io, printstr)
    end

    df_samples = DataFrame(permutedims(samples), :auto)

    df_concentration = DataFrame((conc_mat), :auto)

    df_d10 = DataFrame((d10matrix), :auto)
    df_d32 = DataFrame((d32matrix), :auto)

    df_d43 = DataFrame((d43matrix), :auto)

    XLSX.openxlsx("Generated Datasets/$savestring.xlsx", mode = "w") do xf

        XLSX.rename!(xf[1], "Samples")
        XLSX.addsheet!(xf, "Concentration")
        XLSX.addsheet!(xf, "D10")
        XLSX.addsheet!(xf, "D32")
        XLSX.addsheet!(xf, "D43")


        XLSX.writetable!(xf[1], df_samples)
        XLSX.writetable!(xf[2], df_concentration)
        XLSX.writetable!(xf[3], df_d10)
        XLSX.writetable!(xf[4], df_d32)
        XLSX.writetable!(xf[5], df_d43)

    end

end

"""
    generatedesignspace(parameters, nucleationfunction, growthfunction,
                        aggregationfunction, breakagefunction, concrange,
                        solver; tspan = (0.0, 460.0), save_every = 5.0,
                        n_samples = 256, kwargs...)

Sweep over a range of initial concentrations to build a process design
space.

# Arguments
- `parameters::AbstractArray{Float64}`: kinetic parameter vector of
  length `nν + n_g + n_a + n_b`.
- `concrange::Tuple{Float64, Float64}`: lower and upper limits for the
  initial concentration `c₀`.
- `n_samples::Int`: number of concentration points in the sweep.
- `solver`: finite-volume or similar discretised solver.
- `tspan`, `save_every`: simulation time span and saving interval.
- `kwargs`: passed to [`runsimulation`](@ref).

# Returns
A named tuple `(c0, time, concentration, quantile, yield, d10, d32, d43)`
where each matrix has size `(n_samples, n_tsteps)`.
"""
function generatedesignspace(parameters::AbstractArray{Float64},
                             nucleationfunction::AbstractNucleationFunction,
                             growthfunction::AbstractGrowthFunction,
                             aggregationfunction::AbstractAggregationFunction,
                             breakagefunction::AbstractBreakageFunction,
                             concrange::Tuple{Float64, Float64},
                             solver::AbstractDiscretisedSolver;
                             tspan::Tuple{Float64, Float64} = (0.0, 460.0),
                             save_every::Float64 = 5.0, n_samples::Int64 = 256, kwargs...)


    samples = LinRange(concrange[1], concrange[2], n_samples)

    conc_mat = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d50matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d10matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d32matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d43matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    prog = Progress(n_samples, desc = "Generating DS...", dt = 0.1, showspeed = true,
                    barlen = 15)

    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(parameters;
                            nucl = nucleationfunction,
                            gr = growthfunction,
                            agg = aggregationfunction,
                            br = breakagefunction,
                            initial_concentration = samples[i],
                            save_idx = tspan[1]:save_every:tspan[2],
                            solver = solver,
                            kwargs...)
        #

        conc_mat[i, :] = sol.concentration

        d50matrix[i, :] = sol.d50q
        d10matrix[i, :] = sol.d10q

        momentsizes = getmomentsizes(prob, sol)

        d32matrix[i, :] = momentsizes.d32

        d43matrix[i, :] = momentsizes.d43

        next!(prog)

    end

    yield_mat = similar(conc_mat)
    yield_mat = 40.0 .* (conc_mat[:, 1] .- conc_mat)

    return (c0 = samples, time = collect(0:save_every:tspan[2]), concentration = conc_mat,
            quantile = d50matrix, yield = yield_mat, d10 = d10matrix, d32 = d32matrix,
            d43 = d43matrix)
end
"""
    generatedesignspace(parameters, nucleationfunction, growthfunction,
                        aggregationfunction, breakagefunction, concrange,
                        solver::MoM; tspan = (0.0, 460.0),
                        save_every = 5.0, n_samples = 256, kwargs...)

`MoM`-specific overload of [`generatedesignspace`](@ref) returning
matrices of `d10`, `d32` and `d43` obtained directly from the
moment-based solver.
"""
function generatedesignspace(parameters::AbstractArray{Float64},
                             nucleationfunction::AbstractNucleationFunction,
                             growthfunction::AbstractGrowthFunction,
                             aggregationfunction::AbstractAggregationFunction,
                             breakagefunction::AbstractBreakageFunction,
                             concrange::Tuple{Float64, Float64}, solver::MoM;
                             tspan::Tuple{Float64, Float64} = (0.0, 460.0),
                             save_every::Float64 = 5.0, n_samples::Int64 = 256, kwargs...)


    samples = LinRange(concrange[1], concrange[2], n_samples)

    conc_mat = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d10matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d32matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    d43matrix = Matrix{Float64}(undef, n_samples, Int64(tspan[2] ÷ save_every + 1))

    prog = Progress(n_samples, desc = "Generating DS...", dt = 0.1, showspeed = true,
                    barlen = 15)

    @floop for i in 1:n_samples
        prob,
        sol = runsimulation(parameters;
                            nucl = nucleationfunction,
                            gr = growthfunction,
                            agg = aggregationfunction,
                            br = breakagefunction,
                            initial_concentration = samples[i],
                            save_idx = tspan[1]:save_every:tspan[2],
                            solver = solver,
                            kwargs...)
        #

        conc_mat[i, :] = sol.concentration

        d10matrix[i, :] = sol.d10

        d32matrix[i, :] = sol.d32

        d43matrix[i, :] = sol.d43

        next!(prog)

    end

    yield_mat = similar(conc_mat)
    yield_mat = 40.0 .* (conc_mat[:, 1] .- conc_mat)

    return (c0 = samples, time = collect(0:save_every:tspan[2]), concentration = conc_mat,
            yield = yield_mat, d10 = d10matrix, d32 = d32matrix, d43 = d43matrix)
end
