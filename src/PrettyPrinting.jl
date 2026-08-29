"""
Pretty-printing utilities for CriSTool using Term.jl.
Provides styled terminal output for parameter estimation and simulation workflows.
"""

import Term: Panel, Table, tprint

# ============================================================================ #
#                         GLOBAL COLOR PALETTE                                 #
# ============================================================================ #

"""
    CRISTOOL_PALETTE

Global mutable dictionary of colors used for terminal styling.
Modify this dictionary to customize the appearance of CriSTool output.

Keys:
- `:primary`   - Main accent color (default: soft green #A1D8B6)
- `:secondary` - Secondary color (default: warm tan #D2C48E)
- `:accent`    - Highlight color (default: vibrant red-orange #F45F40)
- `:highlight` - Emphasis color (default: peach #F9AE8D)
- `:info`      - Informational color (default: soft blue #80B9CE)
"""
const CRISTOOL_PALETTE = Dict{Symbol, String}(:primary => "#A1D8B6",
                                              :secondary => "#D2C48E",
                                              :accent => "#F45F40",
                                              :highlight => "#F9AE8D",
                                              :info => "#80B9CE")

# ============================================================================ #
#                         HELPER FUNCTIONS                                     #
# ============================================================================ #

"""
    format_solver_string(solver) -> String

Format a solver struct into a styled string.
"""
function format_solver_string(solver)
    return solver.string
end

"""
    format_kinetics_table(nucleationfunction, growthfunction, aggregationfunction, breakagefunction) -> Term.Table

Create a formatted table summarizing kinetic function settings.
"""
function format_kinetics_table(nucleationfunction, growthfunction,
                               aggregationfunction, breakagefunction)
    data = ["Nucleation" nucleationfunction.string;
            "Growth" growthfunction.string;
            "Aggregation" aggregationfunction.string;
            "Breakage" breakagefunction.string]
    return Table(data;
                 header = ["Function", "Type"],
                 box = :ROUNDED,
                 style = CRISTOOL_PALETTE[:info])
end

"""
    format_bounds_table(lb, ub) -> Term.Table

Create a formatted table showing parameter bounds.
"""
function format_bounds_table(lb::AbstractVector, ub::AbstractVector)
    n = length(lb)
    data = hcat(1:n, round.(lb, sigdigits = 4), round.(ub, sigdigits = 4))
    return Table(data;
                 header = ["Param", "Lower", "Upper"],
                 box = :SIMPLE,
                 style = CRISTOOL_PALETTE[:secondary])
end

"""
    format_distribution_summary(prior) -> String

Format a prior distribution into a summary string.
"""
function format_distribution_summary(prior)
    return string(prior)
end

"""
    format_parameters(params::AbstractVector; sigdigits=4) -> String

Format a parameter vector into a readable string.
"""
function format_parameters(params::AbstractVector; sigdigits::Int = 4)
    return "[" * join(round.(params, sigdigits = sigdigits), ", ") * "]"
end

# ============================================================================ #
#                         PANEL FUNCTIONS                                      #
# ============================================================================ #

"""
    print_start_panel(routine_name, settings_content; verbosity=1)

Print a styled start panel for a routine.

# Arguments
- `routine_name::String`: Name of the routine (e.g., "Parameter Estimation")
- `settings_content::String`: Content showing settings/configuration
- `verbosity::Int`: 0=silent, 1=normal, 2=verbose
"""
function print_start_panel(routine_name::String, settings_content::String;
                           verbosity::Int = 1)
    verbosity < 1 && return nothing

    primary = CRISTOOL_PALETTE[:primary]
    accent = CRISTOOL_PALETTE[:accent]

    panel = Panel(settings_content;
                  title = "{$accent bold}🚀 $routine_name Starting{/$accent bold}",
                  title_justify = :center,
                  style = primary,
                  fit = false,
                  width = 72)
    println(panel)
    return nothing
end

"""
    print_end_panel(routine_name, results_content; verbosity=1)

Print a styled end panel showing results.

# Arguments
- `routine_name::String`: Name of the routine
- `results_content::String`: Content showing results
- `verbosity::Int`: 0=silent, 1=normal, 2=verbose
"""
function print_end_panel(routine_name::String, results_content::String;
                         verbosity::Int = 1)
    verbosity < 1 && return nothing

    primary = CRISTOOL_PALETTE[:primary]
    accent = CRISTOOL_PALETTE[:highlight]

    panel = Panel(results_content;
                  title = "{$accent bold}✅ $routine_name Complete{/$accent bold}",
                  title_justify = :center,
                  style = primary,
                  fit = false,
                  width = 72)
    println(panel)
    return nothing
end

"""
    print_ensemble_diagnostics(samples::Matrix; verbosity=1)

Print parameter diagnostics table for ensemble samples.

# Arguments
- `samples::Matrix`: Parameter samples (parameters × samples)
- `verbosity::Int`: 0=silent, 1=normal, 2=verbose
"""
function print_ensemble_diagnostics(samples::Matrix; verbosity::Int = 1,
                                    param_names::Union{Nothing, Vector{String}} = nothing)
    verbosity < 2 && return nothing

    n_params = size(samples, 1)
    n_samples = size(samples, 2)

    names = isnothing(param_names) ? ["θ$i" for i in 1:n_params] : param_names

    stats = Matrix{Any}(undef, n_params, 5)
    for i in 1:n_params
        row = samples[i, :]
        stats[i, 1] = names[i]
        stats[i, 2] = round(minimum(row), sigdigits = 4)
        stats[i, 3] = round(maximum(row), sigdigits = 4)
        stats[i, 4] = round(mean(row), sigdigits = 4)
        stats[i, 5] = round(std(row), sigdigits = 4)
    end

    tbl = Table(stats;
                header = ["Param", "Min", "Max", "Mean", "Std"],
                box = :ROUNDED,
                style = CRISTOOL_PALETTE[:info])

    panel = Panel(tbl;
                  title = "{$(CRISTOOL_PALETTE[:secondary]) bold}📊 Parameter Diagnostics (n=$n_samples){/$(CRISTOOL_PALETTE[:secondary]) bold}",
                  title_justify = :left,
                  fit = true)
    println(panel)
    return nothing
end

# ============================================================================ #
#                     PE ROUTINE CONTENT BUILDERS                              #
# ============================================================================ #

"""
    build_pe_start_content(lb, ub, lossfunction, solver, nucleationfunction,
                           growthfunction, aggregationfunction, breakagefunction;
                           nparticles=nothing, generations=nothing, algorithm=nothing,
                           extrastring="") -> String

Build the content string for PE_Routine start panel.
"""
function build_pe_start_content(lb, ub, lossfunction, solver,
                                nucleationfunction, growthfunction,
                                aggregationfunction, breakagefunction;
                                nparticles::Union{Nothing, Int} = nothing,
                                generations::Union{Nothing, Int} = nothing,
                                algorithm = nothing,
                                extrastring::String = "")
    accent = CRISTOOL_PALETTE[:accent]
    info = CRISTOOL_PALETTE[:info]
    secondary = CRISTOOL_PALETTE[:secondary]

    lines = String[]
    push!(lines, "{$info bold}Bounds:{/$info bold}")
    push!(lines, "  LB: $(format_parameters(lb))")
    push!(lines, "  UB: $(format_parameters(ub))")
    push!(lines, "")
    push!(lines, "{$info bold}Configuration:{/$info bold}")
    push!(lines, "  Loss function: {$accent}$(lossfunction.string){/$accent}")
    push!(lines, "  Solver: {$secondary}$(solver.string){/$secondary}")
    push!(lines, "  Nucleation: $(nucleationfunction.string)")
    push!(lines, "  Growth: $(growthfunction.string)")
    push!(lines, "  Aggregation: $(aggregationfunction.string)")
    push!(lines, "  Breakage: $(breakagefunction.string)")

    if !isnothing(nparticles)
        push!(lines, "  Particles: $nparticles")
    end
    if !isnothing(generations)
        push!(lines, "  Generations: $generations")
    end
    if !isnothing(algorithm)
        push!(lines, "  Algorithm: $algorithm")
    end
    if !isempty(extrastring)
        push!(lines, "  ID: {dim}$extrastring{/dim}")
    end

    return join(lines, "\n")
end

"""
    build_pe_end_content(optimal_params, loss_value; stats=nothing) -> String

Build the content string for PE_Routine end panel.
"""
function build_pe_end_content(optimal_params::AbstractVector, loss_value::Real;
                              stats = nothing)
    accent = CRISTOOL_PALETTE[:accent]
    highlight = CRISTOOL_PALETTE[:highlight]

    lines = String[]
    push!(lines, "{$highlight bold}Optimal Parameters:{/$highlight bold}")
    push!(lines, "  $(format_parameters(optimal_params))")
    push!(lines, "")
    push!(lines,
          "{$accent bold}Loss Value: $(round(loss_value, sigdigits=5)){/$accent bold}")

    if !isnothing(stats)
        push!(lines, "")
        push!(lines, "{dim}Statistics:{/dim}")
        if hasproperty(stats, :iterations)
            push!(lines, "  Iterations: $(stats.iterations)")
        end
        if hasproperty(stats, :time)
            push!(lines, "  Time: $(round(stats.time, digits=2))s")
        end
        if hasproperty(stats, :fevals)
            push!(lines, "  Function evals: $(stats.fevals)")
        end
    end

    return join(lines, "\n")
end

# ============================================================================ #
#                     ABCDE ROUTINE CONTENT BUILDERS                           #
# ============================================================================ #

"""
    build_abcde_start_content(optimalpara, prior, target, optimallossfunction,
                              confidenceinterval, nparticles, generations,
                              solver, nucleationfunction, growthfunction,
                              aggregationfunction, breakagefunction;
                              test=:f, extrastring="") -> String

Build the content string for ABCDE_Routine start panel.
"""
function build_abcde_start_content(optimalpara, prior, target, optimallossfunction,
                                   confidenceinterval, nparticles, generations,
                                   solver, nucleationfunction, growthfunction,
                                   aggregationfunction, breakagefunction;
                                   test::Symbol = :f, extrastring::String = "")
    accent = CRISTOOL_PALETTE[:accent]
    info = CRISTOOL_PALETTE[:info]
    highlight = CRISTOOL_PALETTE[:highlight]
    secondary = CRISTOOL_PALETTE[:secondary]

    ci_pct = Int(confidenceinterval * 100)

    lines = String[]
    push!(lines, "{$highlight bold}Target Settings:{/$highlight bold}")
    push!(lines,
          "  Optimal MLE: {$accent}$(round(optimallossfunction, sigdigits=4)){/$accent}")
    push!(lines, "  Target: {$accent}$(round(target, sigdigits=4)){/$accent}")
    push!(lines, "  Confidence: $(ci_pct)%")
    push!(lines, "  Test: $test")
    push!(lines, "")
    push!(lines, "{$info bold}Algorithm:{/$info bold}")
    push!(lines, "  Particles: $nparticles")
    push!(lines, "  Generations: $generations")
    push!(lines, "")
    push!(lines, "{$info bold}Model:{/$info bold}")
    push!(lines, "  Solver: {$secondary}$(solver.string){/$secondary}")
    push!(lines, "  Nucleation: $(nucleationfunction.string)")
    push!(lines, "  Growth: $(growthfunction.string)")
    push!(lines, "")
    push!(lines, "{dim}Prior: $prior{/dim}")

    if !isempty(extrastring)
        push!(lines, "{dim}ID: $extrastring{/dim}")
    end

    return join(lines, "\n")
end

"""
    build_abcde_end_content(res, optimalpara) -> String

Build the content string for ABCDE_Routine end panel.
"""
function build_abcde_end_content(mean_params::AbstractVector, reached_target::Bool)
    accent = CRISTOOL_PALETTE[:accent]
    highlight = CRISTOOL_PALETTE[:highlight]

    status = reached_target ? "{green bold}✓ Target reached{/green bold}" :
             "{yellow bold}⚠ Target not reached{/yellow bold}"

    lines = String[]
    push!(lines, status)
    push!(lines, "")
    push!(lines, "{$highlight bold}Mean Parameters:{/$highlight bold}")
    push!(lines, "  $(format_parameters(mean_params))")

    return join(lines, "\n")
end

# ============================================================================ #
#                     MCMC ROUTINE CONTENT BUILDERS                            #
# ============================================================================ #

"""
    build_mcmc_start_content(prior, sampler, n_samples, n_chains,
                             lossfunction, solver, nucleationfunction,
                             growthfunction, aggregationfunction, breakagefunction;
                             extrastring="", symbols=nothing) -> String

Build the content string for `MCMC_Routine` start panel.
"""
function build_mcmc_start_content(prior, sampler, n_samples::Int, n_chains::Int,
                                  lossfunction, solver, nucleationfunction,
                                  growthfunction, aggregationfunction, breakagefunction;
                                  extrastring::String = "",
                                  symbols::Union{Nothing, Vector{Symbol}} = nothing)
    accent = CRISTOOL_PALETTE[:accent]
    info = CRISTOOL_PALETTE[:info]
    secondary = CRISTOOL_PALETTE[:secondary]

    lines = String[]
    push!(lines, "{$info bold}Sampler:{/$info bold}")
    push!(lines, "  $sampler")
    push!(lines, "  Samples per chain: $n_samples")
    push!(lines, "  Chains: $n_chains")
    if !isnothing(symbols)
        push!(lines, "  Parameters: {dim}$(join(symbols, ", ")){/dim}")
    end
    push!(lines, "")
    push!(lines, "{$info bold}Model:{/$info bold}")
    push!(lines, "  Loss function: {$accent}$(lossfunction.string){/$accent}")
    push!(lines, "  Solver: {$secondary}$(solver.string){/$secondary}")
    push!(lines, "  Nucleation: $(nucleationfunction.string)")
    push!(lines, "  Growth: $(growthfunction.string)")
    push!(lines, "  Aggregation: $(aggregationfunction.string)")
    push!(lines, "  Breakage: $(breakagefunction.string)")
    push!(lines, "")
    push!(lines, "{dim}Prior: $prior{/dim}")

    if !isempty(extrastring)
        push!(lines, "{dim}ID: $extrastring{/dim}")
    end

    return join(lines, "\n")
end

"""
    build_mcmc_end_content(chain, burnin::Int=0) -> String

Build the content string for `MCMC_Routine` end panel. Reports the posterior
mean per parameter (post-burnin) and the number of retained samples.
"""
function build_mcmc_end_content(chain::MCMCChains.Chains, burnin::Int = 0)
    accent = CRISTOOL_PALETTE[:accent]
    highlight = CRISTOOL_PALETTE[:highlight]

    chain_subset = burnin > 0 ? chain[(burnin + 1):end, :, :] : chain
    stats = mean(chain_subset).nt
    param_names = collect(stats.parameters)
    means = Dict(param_names[i] => stats.mean[i] for i in eachindex(stats.mean))

    lines = String[]
    push!(lines, "{$highlight bold}Posterior Means:{/$highlight bold}")
    for name in param_names
        push!(lines, "  $(name): {$accent}$(round(means[name], sigdigits=5)){/$accent}")
    end
    push!(lines, "")
    push!(lines, "{dim}Samples per chain: $(size(chain_subset.value, 1)){/dim}")

    return join(lines, "\n")
end
