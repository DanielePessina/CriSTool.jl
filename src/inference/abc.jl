"""
Approximate Bayesian computation routines for parameter inference in crystallisation models. Implements the ABCDE algorithm and related plotting helpers.
"""
function _dofcalculator(::AbstractPELossFunction, datasets::Vector{<:AbstractExperiment})
    return sum(length(observable.mean)
               for experiment in datasets
               for observable in values(experiment.observables))
end

function _abcde_target(optimallossfunction::Real, dof::Integer, nparams::Integer,
                       confidenceinterval::Real; test::Symbol = :f)
    if test == :f || test == :fstat || test == :ftest
        return optimallossfunction * (1 +
                nparams / dof *
                quantile(FDist(nparams, dof), confidenceinterval))
    elseif test == :wilks || test == :chisq || test == :chisqtest
        return optimallossfunction + 0.5 * quantile(Chisq(nparams), confidenceinterval)
    else
        throw(ArgumentError("Unknown ABCDE target test: $test. Use :f or :wilks."))
    end
end

function _resolve_abc_savedir(savedir::Union{Nothing, AbstractString})
    return isnothing(savedir) ? nothing : String(savedir)
end

function _abc_particles_matrix(particles)
    try
        samples = Matrix(particles)
        return isempty(samples) ? nothing : samples
    catch
        return nothing
    end
end

_abc_has_particles(particles) = !isnothing(_abc_particles_matrix(particles))

# ============================================================================
# Sampler dispatch
#
# Each ABC sampler is described by a small struct that holds its
# algorithm-specific knobs. `run_abc` owns the common scaffolding (target
# computation, panels, lambda construction, persistence, plotting) and
# dispatches into `_runsampler` for the algorithm-specific call. To add a new
# sampler (e.g. SMC), define a new <:AbstractABCSampler struct, add a
# _runsampler method, and add label/metadata helpers.
# ============================================================================

abstract type AbstractABCSampler end

"""
    ABCDESampler(α=0)

Standard ABCDE (Differential Evolution) sampler from KissABC.
"""
Base.@kwdef struct ABCDESampler <: AbstractABCSampler
    α::Int64 = 0
end

"""
    ABCDETurnerSampler(K=8, p_migration=0.10, p_crossover=0.90, κ=1.0,
                      γ2_burnin=0.5, kernel=:gaussian, burnin_frac=0.3)

Turner & Sederberg (2012) ABCDE variant with K-group migration, DE crossover/
mutation, and a kernel-based proposal distribution.
"""
Base.@kwdef struct ABCDETurnerSampler <: AbstractABCSampler
    K::Int = 8
    p_migration::Float64 = 0.10
    p_crossover::Float64 = 0.90
    κ::Float64 = 1.0
    γ2_burnin::Float64 = 0.5
    kernel::Symbol = :gaussian
    burnin_frac::Float64 = 0.3
end

# Sampler-specific call into KissABC. Returns (res, reached_ϵ).
function _runsampler(s::ABCDESampler, prior, lossfn, target, optimallossfunction;
                     nparticles, generations, HPC, earlystop)
    res = ABCDE(prior, lossfn, target;
                nparticles = nparticles, generations = generations,
                α = s.α, HPC = HPC, earlystop = earlystop)
    return res, true  # ABCDE returns when target is reached or generations exhausted
end

function _runsampler(s::ABCDETurnerSampler, prior, lossfn, target, optimallossfunction;
                     nparticles, generations, HPC, earlystop)
    δ_derived = max(target - optimallossfunction, eps())
    res = ABCDE_Turner(prior, lossfn, target;
                       nparticles = nparticles, generations = generations,
                       K = s.K, p_migration = s.p_migration, p_crossover = s.p_crossover,
                       κ = s.κ, γ2_burnin = s.γ2_burnin, kernel = s.kernel,
                       δ = δ_derived, best_cost = optimallossfunction,
                       burnin_frac = s.burnin_frac, earlystop = earlystop, HPC = HPC)
    return res, res.reached_ϵ
end

# Display labels and persistence keys per sampler. Two suffix conventions are
# used historically ("[Turner]" for panel titles, " Turner" for filenames);
# both are preserved here.
_routine_label(::ABCDESampler) = "ABCDE Routine"
_routine_label(::ABCDETurnerSampler) = "ABCDE_Turner Routine"
_panel_suffix(::ABCDESampler) = ""
_panel_suffix(::ABCDETurnerSampler) = " [Turner]"
_save_suffix(::ABCDESampler) = ""
_save_suffix(::ABCDETurnerSampler) = " Turner"

# Extra fields included in jldsave + result dict for sampler-specific metadata.
_sampler_metadata(::ABCDESampler) = NamedTuple()
_sampler_metadata(s::ABCDETurnerSampler) = (K = s.K, kernel = s.kernel)

"""
    run_abc(lossfunction, measurement, optimalpara, prior,
            nucleationfunction, growthfunction,
            aggregationfunction, breakagefunction;
            solver, sampler::AbstractABCSampler = ABCDESampler(),
            validation = nothing, extrastring = "Empty",
            nparticles = 1024, generations = 128, saveplot = true,
            confidenceinterval = 0.95, HPC = false, verbosity = 1,
            earlystop = false, test = :f, outputdir = nothing)

Domain-level ABC inference for crystallisation kinetic parameters. Owns target
computation, the loss-function lambda, posterior persistence and plotting; the
choice of sampler (ABCDE, Turner ABCDE, …) is selected via `sampler`.

# Arguments
- `measurement`: experimental datasets used to compute the discrepancy.
- `optimalpara`: reference parameter vector of length nν+ng+na+nb.
- `prior`: prior distribution (e.g. `Distributions.product_distribution([...])`).
- `solver`: numerical solver used by the loss function.
- `diss`: optional independent dissolution model; its parameter block follows
  the growth block.
- `sampler`: which ABC algorithm to run. Defaults to `ABCDESampler()`.
- `nparticles`, `generations`: ABC population size and iteration count.
- `confidenceinterval`, `test`: stopping criterion (F-statistic or χ²).
- `validation`, `saveplot`, `extrastring`, `HPC`, `earlystop`, `verbosity`:
  output and execution controls.
- `outputdir`: directory for posterior persistence and plots. `nothing`
  (default) performs no filesystem writes; an explicit directory receives
  the posterior object (`.jld2`) and — when `saveplot` is true — the ABC
  and measurement plots.

# Returns
`(res, dict)` where `res` is the sampler-specific result and `dict` contains
the optimal-loss target, posterior summaries, prior, and any sampler-specific
metadata (e.g. K, kernel for Turner).
"""
function run_abc(lossfunction::AbstractPELossFunction,
                 measurement::Vector{<:AbstractExperiment},
                 optimalpara::AbstractVector{<:Real}, prior,
                 nucleationfunction::AbstractNucleationFunction,
                 growthfunction::AbstractGrowthFunction,
                 aggregationfunction::AbstractAggregationFunction,
                 breakagefunction::AbstractBreakageFunction;
                 diss::AbstractDissolutionFunction = nodissolution(),
                 solver::AbstractSolver,
                 sampler::AbstractABCSampler = ABCDESampler(),
                 validation::Union{Nothing, Vector{<:AbstractExperiment}} = nothing,
                 extrastring::String = "Empty", nparticles::Int64 = 1024,
                 generations::Int64 = 128, saveplot::Bool = true,
                 confidenceinterval::Float64 = 0.95, HPC::Bool = false,
                 verbosity::Int64 = 1, earlystop::Bool = false, test::Symbol = :f,
                 outputdir::Union{Nothing, AbstractString} = nothing)

    dof = _dofcalculator(lossfunction, measurement) - length(optimalpara)

    loss_problem = _build_loss_problem(nucleationfunction, growthfunction,
                                       aggregationfunction, breakagefunction, solver;
                                       diss = diss)
    loss_setup = prepare_loss(loss_problem, measurement)

    optimallossfunction = loss(lossfunction, loss_setup, optimalpara)

    target = _abcde_target(optimallossfunction, dof, length(optimalpara),
                           confidenceinterval; test = test)

    confidenceinterval_str = string(Int(confidenceinterval * 100))
    optmle_round = round(optimallossfunction, sigdigits = 4)
    target_round = round(target, sigdigits = 4)

    panel_suffix = _panel_suffix(sampler)
    save_suffix = _save_suffix(sampler)
    label = _routine_label(sampler)

    start_content = build_abcde_start_content(optimalpara, prior, target,
                                              optimallossfunction,
                                              confidenceinterval, nparticles, generations,
                                              solver, nucleationfunction, growthfunction,
                                              aggregationfunction, breakagefunction;
                                              test = test,
                                              extrastring = extrastring * panel_suffix)

    print_start_panel(label, start_content; verbosity = verbosity)

    lossfn = x -> loss(lossfunction, loss_setup, collect(x))

    res, reached_ϵ = _runsampler(sampler, prior, lossfn, target, optimallossfunction;
                                 nparticles = nparticles, generations = generations,
                                 HPC = HPC, earlystop = earlystop)

    if verbosity > 1
        display(res)
    end

    has_particles = _abc_has_particles(res.P)
    if !has_particles
        @warn("$(label) returned no posterior particles; skipping posterior summaries and plots.")
    end
    meanpara_round = has_particles ? round.(pquantile.(res.P, 0.5), digits = 4) :
                     round.(optimalpara, digits = 4)

    now_str = Dates.format(now(), "yy-m-d HH-MM")
    output_dir = _resolve_abc_savedir(outputdir)
    sampler_meta = _sampler_metadata(sampler)

    if output_dir !== nothing
        objects_dir = joinpath(output_dir, "ABCDE Objects")
        mkpath(objects_dir)

        jldsave(joinpath(objects_dir, "$(now_str) $(extrastring)$(save_suffix).jld2");
                res = res, optmle = optimallossfunction, target = target,
                optimalpara = optimalpara, prior = prior, sampler_meta...)
    end

    if has_particles && saveplot && output_dir !== nothing
        ABCplot(res, optimalpara, lossfunction; prior = prior,
                title = Makie.rich("$(now_str) $(extrastring)$(panel_suffix)\n MLE",
                                   Makie.subscript("minimum"), " = $optmle_round MLE",
                                   Makie.subscript(confidenceinterval_str),
                                   " = $(target_round), CI = $confidenceinterval_str"),
                saveplot = saveplot,
                savestring = "$(now_str) $(extrastring)$(save_suffix)",
                savedir = output_dir,
                nucleationfunction = nucleationfunction,
                growthfunction = growthfunction,
                aggregationfunction = aggregationfunction,
                breakagefunction = breakagefunction,
                diss = diss)

        _ABCmeasurementplot(res, lossfunction, measurement, optimalpara, nucleationfunction,
                            growthfunction, aggregationfunction, breakagefunction, solver;
                            diss = diss,
                            saveplot = saveplot,
                            title = "$(now_str) $(extrastring)$(save_suffix)\nCI = $confidenceinterval_str",
                            HPC = HPC,
                            savestring = "$(now_str) $(extrastring)$(save_suffix)",
                            savedir = output_dir)

        if !isnothing(validation)
            _ABCmeasurementplot(res, lossfunction, validation, optimalpara,
                                nucleationfunction, growthfunction, aggregationfunction,
                                breakagefunction, solver;
                                diss = diss,
                                saveplot = saveplot,
                                title = "$(now_str) $(extrastring)$(save_suffix) Validation",
                                savestring = "$(now_str) $(extrastring)$(save_suffix) Validation",
                                colouroffset = 3, HPC = HPC, savedir = output_dir)
        end
    end

    end_content = build_abcde_end_content(meanpara_round, reached_ϵ)
    print_end_panel(label, end_content; verbosity = verbosity)

    result_dict = Dict{String, Any}("optmle" => optimallossfunction,
                                    "target" => target,
                                    "optimalparameters" => optimalpara,
                                    "meanparameters" => has_particles ? pmean.(res.P) :
                                                        optimalpara,
                                    "prior" => prior,
                                    "res" => res)
    for (k, v) in pairs(sampler_meta)
        result_dict[String(k)] = v
    end

    return res, result_dict
end

"""
    ABCDE_Routine(...)

Backward-compatible shim over `run_abc` that selects the standard
`ABCDESampler`. See `run_abc` for the full argument list.
"""
function ABCDE_Routine(lossfunction::AbstractPELossFunction,
                       measurement::Vector{<:AbstractExperiment},
                       optimalpara::AbstractVector{<:Real}, prior,
                       nucleationfunction::AbstractNucleationFunction,
                       growthfunction::AbstractGrowthFunction,
                       aggregationfunction::AbstractAggregationFunction,
                       breakagefunction::AbstractBreakageFunction;
                       diss::AbstractDissolutionFunction = nodissolution(),
                       solver::AbstractSolver,
                       validation::Union{Nothing, Vector{<:AbstractExperiment}} = nothing,
                       extrastring::String = "Empty", nparticles::Int64 = 1024,
                       generations::Int64 = 128, alpha::Int64 = 0, saveplot::Bool = true,
                       confidenceinterval::Float64 = 0.95, HPC::Bool = false,
                       verbosity::Int64 = 1,
                       earlystop::Bool = false, test::Symbol = :f,
                       outputdir::Union{Nothing, AbstractString} = nothing)
    return run_abc(lossfunction, measurement, optimalpara, prior, nucleationfunction,
                   growthfunction, aggregationfunction, breakagefunction;
                   diss = diss,
                   solver = solver, sampler = ABCDESampler(α = alpha),
                   validation = validation, extrastring = extrastring,
                   nparticles = nparticles, generations = generations, saveplot = saveplot,
                   confidenceinterval = confidenceinterval, HPC = HPC,
                   verbosity = verbosity, earlystop = earlystop, test = test,
                   outputdir = outputdir)
end

"""
    ABCDE_Turner_Routine(...)

Backward-compatible shim over `run_abc` that selects `ABCDETurnerSampler` with
the supplied Turner-specific kwargs. See `run_abc` for the full argument list.
"""
function ABCDE_Turner_Routine(lossfunction::AbstractPELossFunction,
                              measurement::Vector{<:AbstractExperiment},
                              optimalpara::AbstractVector{<:Real}, prior,
                              nucleationfunction::AbstractNucleationFunction,
                              growthfunction::AbstractGrowthFunction,
                              aggregationfunction::AbstractAggregationFunction,
                              breakagefunction::AbstractBreakageFunction;
                              diss::AbstractDissolutionFunction = nodissolution(),
                              solver::AbstractSolver,
                              validation::Union{Nothing, Vector{<:AbstractExperiment}} = nothing,
                              extrastring::String = "Empty",
                              nparticles::Int64 = 1024,
                              generations::Int64 = 128,
                              saveplot::Bool = true,
                              confidenceinterval::Float64 = 0.95,
                              HPC::Bool = false,
                              verbosity::Int64 = 1,
                              earlystop::Bool = false,
                              test::Symbol = :f,
                              K::Int = 8,
                              p_migration::Float64 = 0.10,
                              p_crossover::Float64 = 0.90,
                              κ::Float64 = 1.0,
                              γ2_burnin::Float64 = 0.5,
                              kernel::Symbol = :gaussian,
                              burnin_frac::Float64 = 0.3,
                              outputdir::Union{Nothing, AbstractString} = nothing)
    sampler = ABCDETurnerSampler(K = K, p_migration = p_migration,
                                 p_crossover = p_crossover, κ = κ,
                                 γ2_burnin = γ2_burnin, kernel = kernel,
                                 burnin_frac = burnin_frac)
    return run_abc(lossfunction, measurement, optimalpara, prior, nucleationfunction,
                   growthfunction, aggregationfunction, breakagefunction;
                   diss = diss,
                   solver = solver, sampler = sampler,
                   validation = validation, extrastring = extrastring,
                   nparticles = nparticles, generations = generations, saveplot = saveplot,
                   confidenceinterval = confidenceinterval, HPC = HPC,
                   verbosity = verbosity, earlystop = earlystop, test = test,
                   outputdir = outputdir)
end

function _append_symbols!(names::Vector{Symbol}, fn, prefix::String)
    fn === nothing && return
    nparams = hasproperty(fn, :nparams) ? getproperty(fn, :nparams) : 0
    syms = hasproperty(fn, :symbols) ? getproperty(fn, :symbols) : Symbol[]
    if nparams == 0
        return
    elseif length(syms) >= nparams
        append!(names, Symbol.(syms[1:nparams]))
    else
        append!(names, [Symbol("$(prefix)$i") for i in 1:nparams])
    end
end

function kinetic_parameter_symbols(nucleationfunction, growthfunction,
                                   aggregationfunction, breakagefunction;
                                   diss = nodissolution())
    names = Symbol[]
    _append_symbols!(names, nucleationfunction, "nu")
    _append_symbols!(names, growthfunction, "gr")
    _append_symbols!(names, diss, "diss")
    _append_symbols!(names, aggregationfunction, "agg")
    _append_symbols!(names, breakagefunction, "br")
    return names
end

function ABCplot(abcres, params::Vector{Float64}, lossfunction::AbstractPELossFunction;
                 prior::Union{Distributions.Distribution, Nothing} = nothing,
                 title = "",
                 show_title::Bool = true,
                 saveplot::Bool = false, savestring::String = "",
                 savedir::Union{Nothing, AbstractString} = nothing,
                 nucleationfunction::Union{AbstractNucleationFunction, Nothing} = nothing,
                 growthfunction::Union{AbstractGrowthFunction, Nothing} = nothing,
                 aggregationfunction::Union{AbstractAggregationFunction, Nothing} = nothing,
                 breakagefunction::Union{AbstractBreakageFunction, Nothing} = nothing,
                 diss::AbstractDissolutionFunction = nodissolution(),
                 showplot::Bool = true)
    samples_mat = _abc_particles_matrix(abcres.P)
    if isnothing(samples_mat)
        @warn("ABC posterior plot skipped because no posterior particles were retained.")
        return nothing
    end
    if abcres.reached_ϵ == false
        @warn("ABC did not reach the target")
    end

    mle = abcres.C
    df = DataFrame([samples_mat;; Vector(mle)], :auto)
    param_names = kinetic_parameter_symbols(nucleationfunction, growthfunction,
                                            aggregationfunction, breakagefunction;
                                            diss = diss)
    if length(param_names) < length(params)
        append!(param_names,
                [Symbol("θ_$i") for i in (length(param_names) + 1):length(params)])
    elseif length(param_names) > length(params)
        param_names = param_names[1:length(params)]
    end
    num_param_cols = min(length(propertynames(df)) - 1, length(param_names))
    rename!(df, Dict(propertynames(df)[i] => param_names[i] for i in 1:num_param_cols))
    mean_params = pmean.(abcres.P)
    param_columns = propertynames(df)[1:num_param_cols]
    truth_params = Dict(param_columns[i] => params[i] for i in 1:num_param_cols)
    mean_params_dict = Dict(param_columns[i] => mean_params[i]
                            for i in 1:num_param_cols)

    if prior !== nothing
        # Use standardized distribution_to_matrix function
        priordist_matrix = distribution_to_matrix(prior, CRISTOOL_PRIOR_PLOT_SAMPLES)
        prior_df = DataFrame(permutedims(priordist_matrix),
                             propertynames(df)[1:num_param_cols])
    else
        prior_df = DataFrame()
    end

    plot_posterior_pairplot(df, truth_params, mean_params_dict;
                            title = show_title ? title : "",
                            savename = savestring,
                            savedir = savedir,
                            lossfunction_string = lossfunction.string,
                            showplot = showplot)
end

function _ABCmeasurementplot(abcres, lossfunction::AbstractPELossFunction,
                             measurements::Vector{<:AbstractExperiment},
                             optimalparameters::Vector{Float64},
                             nucleationfunction::AbstractNucleationFunction,
                             growthfunction::AbstractGrowthFunction,
                             aggregationfunction::AbstractAggregationFunction,
                             breakagefunction::AbstractBreakageFunction,
                             solver::AbstractSolver;
                             diss::AbstractDissolutionFunction = nodissolution(),
                             saveplot::Bool = true, title = "",
                             savestring::String = "", colouroffset::Int64 = 0,
                             HPC::Bool = false, showtext::Bool = true,
                             savedir::Union{Nothing, AbstractString} = nothing)
    samples_mat = _abc_particles_matrix(abcres.P)
    if isnothing(samples_mat)
        @warn("ABC measurement plot skipped because no posterior particles were retained.")
        return nothing
    end
    samples = permutedims(samples_mat)
    ensembleresults = run_ensemble(samples, measurements, nucleationfunction,
                                   growthfunction, aggregationfunction, breakagefunction,
                                   solver; diss = diss, HPC = HPC)

    optimal_solutions = [(runsimulation(optimalparameters,
                                         nucl = nucleationfunction,
                                         gr = growthfunction,
                                         diss = diss,
                                         agg = aggregationfunction,
                                         br = breakagefunction,
                                         initial_concentration = initial_concentration(measurements[m]),
                                         solver = solver,
                                         save_idx = ensembleresults[m].time,
                                         temp_profile = ConstantTemperature(measurements[m].temperature),
                                         initial_crystals = measurements[m].initial_crystals))
                         for m in eachindex(measurements)]

    plot_measurements_vs_ensemble(measurements, ensembleresults, optimal_solutions;
                                  title = title, savename = savestring,
                                  savedir = savedir,
                                  colouroffset = colouroffset,
                                  parameters = optimalparameters,
                                  nucleationfunction = nucleationfunction,
                                  growthfunction = growthfunction,
                                  dissolutionfunction = diss,
                                  aggregationfunction = aggregationfunction,
                                  breakagefunction = breakagefunction,
                                  solver = solver,
                                  parameter_samples = samples, showtext = showtext)
end

function _ABCmeasurementplot_ps(abcres, lossfunction::AbstractPELossFunction,
                                measurements::Vector{<:AbstractExperiment},
                                optimalparameters::Vector{Float64},
                                nucleationfunction::AbstractNucleationFunction,
                                growthfunction::AbstractGrowthFunction,
                                aggregationfunction::AbstractAggregationFunction,
                                breakagefunction::AbstractBreakageFunction,
                                solver::AbstractSolver;
                                diss::AbstractDissolutionFunction = nodissolution(),
                                saveplot::Bool = true, title = "",
                                savestring::String = "", colouroffset::Int64 = 0,
                                HPC::Bool = false, showtext::Bool = true,
                                savedir::Union{Nothing, AbstractString} = nothing)
    samples_mat = _abc_particles_matrix(abcres.P)
    if isnothing(samples_mat)
        @warn("ABC measurement plot skipped because no posterior particles were retained.")
        return nothing
    end
    samples = permutedims(samples_mat)
    ensembleresults = run_ensemble(samples, measurements, nucleationfunction,
                                   growthfunction, aggregationfunction, breakagefunction,
                                   solver; diss = diss, HPC = HPC)

    optimal_solutions = [(runsimulation(optimalparameters,
                                         nucl = nucleationfunction,
                                         gr = growthfunction,
                                         diss = diss,
                                         agg = aggregationfunction,
                                         br = breakagefunction,
                                         initial_concentration = initial_concentration(measurements[m]),
                                         solver = solver,
                                         save_idx = ensembleresults[m].time,
                                         temp_profile = ConstantTemperature(measurements[m].temperature),
                                         initial_crystals = measurements[m].initial_crystals))
                         for m in eachindex(measurements)]

    plot_ps_measurements_vs_ensemble(measurements, ensembleresults, optimal_solutions;
                                     title = title, savename = savestring,
                                     savedir = savedir,
                                     colouroffset = colouroffset,
                                     parameters = optimalparameters,
                                     nucleationfunction = nucleationfunction,
                                     growthfunction = growthfunction,
                                     dissolutionfunction = diss,
                                     aggregationfunction = aggregationfunction,
                                     breakagefunction = breakagefunction,
                                     solver = solver,
                                     parameter_samples = samples, showtext = showtext)
end
