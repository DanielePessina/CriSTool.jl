"""
Approximate Bayesian computation routines for parameter inference in crystallisation models. Implements the ABCDE algorithm and related plotting helpers.
"""
# Count exactly the observations included by preparation, using global weights.
function _abc_observable_weight(lossfunction, index, name)
    hasproperty(lossfunction, :weighting) || return 1.0
    return _observable_weight(lossfunction, index, name)
end

function _dofcalculator(lossfunction::AbstractPELossFunction, setup::LossSetup)
    observable_names = setup.observable_names
    included_count = 0
    for (experiment_index, experiment) in enumerate(setup.experiments)
        for name in propertynames(experiment.observables)
            global_index = findfirst(==(name), observable_names)
            _abc_observable_weight(lossfunction, global_index, name) == 0 && continue
            included_count += length(_included_observation_indices(setup, experiment_index, name))
        end
    end
    return included_count
end

function _abc_unit_weight_likelihood(lossfunction, setup)
    lossfunction isa logMLE || return false
    return all(_abc_observable_weight(lossfunction, index, name) == 1
               for (index, name) in enumerate(setup.observable_names))
end

function _abc_target_policy(lossfunction, setup, test::Symbol, target)
    target !== nothing && return :explicit
    test in (:f, :fstat, :ftest) &&
        throw(ArgumentError("F targets require a residual-sum-of-squares discrepancy; " *
                            "CriSTool's logMLE and MAE losses do not provide one. Supply target explicitly."))
    test in (:auto, :wilks, :chisq, :chisqtest) ||
        throw(ArgumentError("Unknown ABC target policy :$test. Use :auto, :wilks, or target=<value>."))
    _abc_unit_weight_likelihood(lossfunction, setup) ||
        throw(ArgumentError("Wilks targets require a unit-weight logMLE likelihood. " *
                            "For MAE, custom, or weighted losses, supply target explicitly."))
    return :wilks
end

function _abcde_target(optimallossfunction::Real, dof::Integer, nparams::Integer,
                       confidenceinterval::Real; test::Symbol = :wilks,
                       target::Union{Nothing, Real} = nothing)
    isfinite(optimallossfunction) ||
        throw(ArgumentError("The ABC reference loss must be finite."))
    0 < confidenceinterval < 1 ||
        throw(ArgumentError("confidenceinterval must lie strictly between zero and one."))
    nparams >= 0 || throw(ArgumentError("The parameter count must be nonnegative."))
    resolved_target = if target !== nothing
        target
    elseif test in (:wilks, :chisq, :chisqtest)
        nparams == 0 ? optimallossfunction :
            optimallossfunction + 0.5 * quantile(Chisq(nparams), confidenceinterval)
    else
        throw(ArgumentError("Unsupported ABC target policy :$test. Use :wilks or target=<value>."))
    end
    isfinite(resolved_target) && resolved_target >= optimallossfunction ||
        throw(ArgumentError("ABC target must be finite and at least the reference loss ($optimallossfunction)."))
    return resolved_target
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

function _abc_particles_matrix(particles::MonteCarloMeasurements.AbstractParticles)
    scalar_draws = collect(particles.particles)
    return isempty(scalar_draws) ? nothing : reshape(scalar_draws, 1, :)
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

"""
    AbstractABCSampler

Abstract supertype for sampler configuration objects accepted by `run_abc`.
Implementations provide the algorithm-specific parameters and dispatch through
the internal sampler runner.
"""
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
    return res, res.reached_ϵ
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
            diss = nodissolution(),
            solver, sampler::AbstractABCSampler = ABCDESampler(),
            validation = nothing, extrastring = "Empty",
            nparticles = 1024, generations = 128, saveplot = true,
            confidenceinterval = 0.95, HPC = false, verbosity = 1,
            earlystop = false, test = :auto, target = nothing, outputdir = nothing)

Domain-level ABC inference for crystallisation kinetic parameters. Owns target
computation, the loss-function lambda, posterior persistence and plotting; the
choice of sampler (ABCDE, Turner ABCDE, …) is selected via `sampler`.

# Arguments
- `measurement`: experimental datasets used to compute the discrepancy.
- `optimalpara`: reference parameter vector in the compatibility order
  `[p_ν; p_g; p_agg; p_br]`, or in the canonical order
  `[p_ν; p_g; p_diss; p_agg; p_br]` when `diss` is supplied.
- `prior`: prior distribution (e.g. `Distributions.product_distribution([...])`).
- `solver`: numerical solver used by the loss function.
- `diss`: optional independent dissolution model; its parameter block follows
  the growth block.
- `sampler`: which ABC algorithm to run. Defaults to `ABCDESampler()`.
- `nparticles`, `generations`: ABC population size and iteration count.
- `confidenceinterval`, `test`: asymptotic Wilks target policy for unit-weight `logMLE`.
- `target`: explicit discrepancy threshold, required for MAE, weighted or custom losses.
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
                 validation::Union{Nothing, LossSetup, Vector{<:AbstractExperiment}} = nothing,
                 extrastring::String = "Empty", nparticles::Int64 = 1024,
                 generations::Int64 = 128, saveplot::Bool = true,
                 confidenceinterval::Float64 = 0.95, HPC::Bool = false,
                 verbosity::Int64 = 1, earlystop::Bool = false, test::Symbol = :auto,
                 target::Union{Nothing, Real} = nothing,
                 outputdir::Union{Nothing, AbstractString} = nothing)

    loss_problem = _build_loss_problem(nucleationfunction, growthfunction,
                                       aggregationfunction, breakagefunction, solver;
                                       diss = diss)
    loss_setup = prepare_loss(loss_problem, measurement)
    return run_abc(lossfunction, loss_setup, optimalpara, prior;
                   sampler, validation, extrastring, nparticles, generations, saveplot,
                   confidenceinterval, HPC, verbosity, earlystop, test, target, outputdir)
end

"""
    run_abc(lossfunction, setup::LossSetup, optimalpara, prior; test=:auto, target=nothing, ...)

Run ABC using an already configured system and its prepared experiments.
`:auto` uses a Wilks likelihood-ratio increment only for unit-weight `logMLE`.
MAE, weighted likelihoods, and custom discrepancies require an explicit target.
Wilks assumes a regular identifiable likelihood and an interior MLE; the
threshold is asymptotic, not a general finite-sample coverage guarantee.
"""
function run_abc(lossfunction::AbstractPELossFunction, loss_setup::LossSetup,
                 optimalpara::AbstractVector{<:Real}, prior;
                 sampler::AbstractABCSampler = ABCDESampler(),
                 validation::Union{Nothing, LossSetup, Vector{<:AbstractExperiment}} = nothing,
                 extrastring::String = "Empty", nparticles::Int64 = 1024,
                 generations::Int64 = 128, saveplot::Bool = true,
                 confidenceinterval::Float64 = 0.95, HPC::Bool = false,
                 verbosity::Int64 = 1, earlystop::Bool = false, test::Symbol = :auto,
                 target::Union{Nothing, Real} = nothing,
                 outputdir::Union{Nothing, AbstractString} = nothing)
    loss_problem = loss_setup.problem
    measurement = loss_setup.experiments
    solver = loss_problem.solver
    nucleationfunction = loss_problem.kinetics_nucleationfunction
    growthfunction = loss_problem.kinetics_growthfunction
    diss = loss_problem.kinetics_dissolutionfunction
    aggregationfunction = loss_problem.kinetics_aggregationfunction
    breakagefunction = loss_problem.kinetics_breakagefunction
    _validate_loss_weights(lossfunction, loss_setup)
    target_policy = _abc_target_policy(lossfunction, loss_setup, test, target)
    # A failed reference simulation must not become a plausible optimum.
    for prepared_experiment in loss_setup.prepared
        reference_solution = _solve_prepared(prepared_experiment, optimalpara)
        reference_solution.success ||
            throw(ArgumentError("The ABC reference parameters produce an unsuccessful simulation."))
    end
    included_observations = _dofcalculator(lossfunction, loss_setup)
    dof = included_observations - length(optimalpara)
    optimallossfunction = loss(lossfunction, loss_setup, optimalpara)
    target = _abcde_target(optimallossfunction, dof, length(optimalpara),
                           confidenceinterval; test = target_policy, target = target)

    confidenceinterval_str = string(Int(confidenceinterval * 100))
    threshold_caption = target_policy === :explicit ? "Target = $target" :
                        "CI = $confidenceinterval_str"
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
                                              test = target_policy,
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
                optimalpara = optimalpara, prior = prior,
                target_policy = target_policy,
                included_observations = included_observations, sampler_meta...)
    end

    if has_particles && saveplot && output_dir !== nothing
        ABCplot(res, optimalpara, lossfunction; prior = prior,
                title = Makie.rich("$(now_str) $(extrastring)$(panel_suffix)\n MLE",
                                   Makie.subscript("minimum"), " = $optmle_round MLE",
                                   Makie.subscript(confidenceinterval_str),
                                   " = $(target_round), $threshold_caption"),
                saveplot = saveplot,
                savestring = "$(now_str) $(extrastring)$(save_suffix)",
                savedir = output_dir,
                nucleationfunction = nucleationfunction,
                growthfunction = growthfunction,
                aggregationfunction = aggregationfunction,
                breakagefunction = breakagefunction,
                diss = diss)

        _ABCmeasurementplot(res, lossfunction, loss_setup, optimalpara;
                            saveplot = saveplot,
                            title = "$(now_str) $(extrastring)$(save_suffix)\n$threshold_caption",
                            HPC = HPC,
                            savestring = "$(now_str) $(extrastring)$(save_suffix)",
                            savedir = output_dir)

        if !isnothing(validation)
            validation_setup = validation isa LossSetup ? validation :
                               prepare_loss(loss_problem, validation)
            _ABCmeasurementplot(res, lossfunction, validation_setup, optimalpara;
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
                                    "target_policy" => target_policy,
                                    "included_observations" => included_observations,
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
                       validation::Union{Nothing, LossSetup, Vector{<:AbstractExperiment}} = nothing,
                       extrastring::String = "Empty", nparticles::Int64 = 1024,
                       generations::Int64 = 128, alpha::Int64 = 0, saveplot::Bool = true,
                       confidenceinterval::Float64 = 0.95, HPC::Bool = false,
                       verbosity::Int64 = 1,
                       earlystop::Bool = false, test::Symbol = :auto,
                       target::Union{Nothing, Real} = nothing,
                       outputdir::Union{Nothing, AbstractString} = nothing)
    return run_abc(lossfunction, measurement, optimalpara, prior, nucleationfunction,
                   growthfunction, aggregationfunction, breakagefunction;
                   diss = diss,
                   solver = solver, sampler = ABCDESampler(α = alpha),
                   validation = validation, extrastring = extrastring,
                   nparticles = nparticles, generations = generations, saveplot = saveplot,
                   confidenceinterval = confidenceinterval, HPC = HPC,
                   verbosity = verbosity, earlystop = earlystop, test = test, target = target,
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
                              validation::Union{Nothing, LossSetup, Vector{<:AbstractExperiment}} = nothing,
                              extrastring::String = "Empty",
                              nparticles::Int64 = 1024,
                              generations::Int64 = 128,
                              saveplot::Bool = true,
                              confidenceinterval::Float64 = 0.95,
                              HPC::Bool = false,
                              verbosity::Int64 = 1,
                              earlystop::Bool = false,
                              test::Symbol = :auto,
                              target::Union{Nothing, Real} = nothing,
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
                   verbosity = verbosity, earlystop = earlystop, test = test, target = target,
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

"""
    kinetic_parameter_symbols(nucleationfunction, growthfunction,
                              aggregationfunction, breakagefunction;
                              diss=nodissolution()) -> Vector{Symbol}

Return parameter symbols for the supplied kinetic models in composite
`paramaxis` order. Model-declared `symbols` are used when available; otherwise
the result contains `:nu1`, `:gr1`, `:diss1`, `:agg1`, or `:br1` placeholders.
"""
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
                             setup::LossSetup, optimalparameters::AbstractVector{<:Real};
                             saveplot::Bool = true, title = "",
                             savestring::String = "", colouroffset::Int64 = 0,
                             HPC::Bool = false, showtext::Bool = true,
                             savedir::Union{Nothing, AbstractString} = nothing)
    samples_mat = _abc_particles_matrix(abcres.P)
    isnothing(samples_mat) && return nothing
    samples = permutedims(samples_mat)
    ensembleresults = run_ensemble(samples, setup; HPC = HPC)
    optimal_solutions = [(prepared.problem, _solve_prepared(prepared, optimalparameters))
                         for prepared in setup.prepared]
    configured_problem = setup.problem
    return plot_measurements_vs_ensemble(setup.experiments, ensembleresults, optimal_solutions;
        title = title, savename = savestring, savedir = savedir,
        colouroffset = colouroffset, parameters = optimalparameters,
        nucleationfunction = configured_problem.kinetics_nucleationfunction,
        growthfunction = configured_problem.kinetics_growthfunction,
        dissolutionfunction = configured_problem.kinetics_dissolutionfunction,
        aggregationfunction = configured_problem.kinetics_aggregationfunction,
        breakagefunction = configured_problem.kinetics_breakagefunction,
        solver = configured_problem.solver, parameter_samples = samples, showtext = showtext)
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
