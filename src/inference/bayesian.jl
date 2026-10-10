"""
Turing model construction for `fit(::PreparedFit{<:Any, <:BayesianSpec})`,
chain renaming, and native-chain trace/density diagnostics.
"""

"""Build a task-local memoized initializer of element type `T`.

Julia >= 1.12 ships `Base.OncePerTask`, so this returns the native object
directly (zero custom code, explicit `T`).  On the 1.10/1.11 LTS floor that
type is absent (no Compat backport either), so it falls back to an
identical port: Base's own implementation of `OncePerTask` is exactly
`get!(f, task_local_storage(), self)::T` keyed by the object.

`T` is passed explicitly (never inferred) so the model's
`loss_setup_per_task()` call site - which runs once per NUTS
log-probability evaluation - stays type stable on every supported Julia.
"""
function _cristool_once_per_task(initializer, ::Type{T}) where {T}
    if isdefined(Base, :OncePerTask)
        return Base.OncePerTask{T}(initializer)
    end
    return _CristoolOncePerTask{T, typeof(initializer)}(initializer)
end

"""Task-local memoized initializer, ported from `Base.OncePerTask` (1.12+)."""
struct _CristoolOncePerTask{T, F}
    initializer::F
end
@inline function (_opt::_CristoolOncePerTask{T, F})() where {T, F}
    get!(_opt.initializer, task_local_storage(), _opt)::T
end

# Keep the Turing model at module scope (DynamicPPL requirement). Its data
# arguments are captured by a zero-argument adapter below because the generated
# DynamicPPL 0.39 evaluator for nonempty model arguments references a global
# defined later in that dependency; Julia 1.12+ rejects that world-age access.
Turing.@model function _cristool_nuts_loss_model_with_data(prior, lossfunction,
                                                            loss_setup_per_task)
    n = length(prior)
    θ = Vector{Float64}(undef, n)
    for i in 1:n
        θ[i] ~ prior[i]
    end
    Turing.@addlogprob!(-loss(lossfunction, loss_setup_per_task(), θ))
end

struct _CristoolNUTSModelEvaluator{CompiledModel, ModelArguments, ModelDefaults}
    compiled_model::CompiledModel
    model_arguments::ModelArguments
    model_defaults::ModelDefaults
end

function (evaluator::_CristoolNUTSModelEvaluator)(outer_model, varinfo)
    # DynamicPPL's generated @model body reads its model context from this
    # first argument. Forward the active context installed on the public model
    # so logjoint, initialization, and sampling retain normal conditioning.
    compiled_model = Turing.DynamicPPL.contextualize(
        evaluator.compiled_model, outer_model.context)
    return compiled_model.f(compiled_model, varinfo, evaluator.model_arguments...;
                            evaluator.model_defaults...)
end

function _cristool_nuts_loss_model(prior, lossfunction, loss_setup_per_task)
    compiled_model = _cristool_nuts_loss_model_with_data(
        prior, lossfunction, loss_setup_per_task)
    evaluator = _CristoolNUTSModelEvaluator(
        compiled_model, Tuple(values(compiled_model.args)), compiled_model.defaults)
    return Turing.DynamicPPL.Model{false}(evaluator, NamedTuple(), NamedTuple())
end

"""
    rename_chain(chain::MCMCChains.Chains, symbols::Vector{Symbol}) -> Chains

Rename the parameter columns of an MCMC chain in place of a new vector of
symbols. The chain must
contain exactly `length(symbols)` parameters.
"""
function rename_chain(chain::MCMCChains.Chains, symbols::Vector{Symbol})
    old_names = collect(chain.name_map.parameters)
    length(old_names) == length(symbols) ||
        throw(ArgumentError("rename_chain: chain has $(length(old_names)) parameters but " *
                            "$(length(symbols)) symbols were given."))
    rename_dict = Dict(old_names[i] => symbols[i] for i in eachindex(old_names))
    return MCMCChains.replacenames(chain, rename_dict)
end

function _mcmcchain_to_flexichain(chain::MCMCChains.Chains)
    params = collect(names(chain, :parameters))
    niters = size(chain.value, 1)
    nchains = size(chain.value, 3)
    dict = Dict{FlexiChains.ParameterOrExtra{Symbol}, Matrix{Float64}}()
    for (j, p) in enumerate(params)
        dict[FlexiChains.Parameter(p)] = chain.value[:, j, :]
    end
    return FlexiChains.FlexiChain{Symbol}(niters, nchains, dict)
end

"""
    ChainStatsPlots(chain; burnin=0, title="", show_title=true, saveplot=false,
                    savestring="", savedir=nothing, showplot=true)

Create trace and density plots for MCMC chain diagnostics (Makie backend).

# Arguments
- `chain`: `MCMCChains.Chains` object.
- `burnin`: number of initial iterations to discard.
- `title`, `show_title`: figure title controls.
- `saveplot`, `savestring`, `savedir`: save the figure to
  `joinpath(savedir, savestring * " ChDensity.png")`. With `saveplot=true`
  and `savedir=nothing` the figure is not saved.
- `showplot`: display the figure.

# Returns
The generated `Makie.Figure`.
"""
function ChainStatsPlots(chain::MCMCChains.Chains;
                         burnin::Int = 0,
                         title = "",
                         show_title::Bool = true,
                         saveplot::Bool = false,
                         savestring::String = "",
                         savedir::Union{Nothing, AbstractString} = nothing,
                         showplot::Bool = true)

    chain_subset = burnin > 0 ? chain[(burnin + 1):end, :, :] : chain
    fc = _mcmcchain_to_flexichain(chain_subset)
    param_names = collect(names(chain_subset, :parameters))
    n_params = length(param_names)

    fig = Makie.Figure(size = (1000, 260 * n_params))
    if show_title && !isempty(title)
        Makie.Label(fig[0, :], title)
    end

    for (i, param) in enumerate(param_names)
        ax_trace = Makie.Axis(fig[i, 1])
        FlexiChains.mtraceplot!(ax_trace, fc, param)
        if i < n_params
            Makie.hidexdecorations!(ax_trace; grid = false)
        else
            ax_trace.xlabel = "Iteration"
        end

        ax_density = Makie.Axis(fig[i, 2])
        Makie.density!(ax_density, fc, param)
        Makie.hideydecorations!(ax_density)
        if i < n_params
            Makie.hidexdecorations!(ax_density; grid = false)
        else
            ax_density.xlabel = "Parameter estimate"
        end
    end

    if saveplot && savedir !== nothing
        outdir = String(savedir)
        mkpath(outdir)
        Makie.save(joinpath(outdir, "$(savestring) ChDensity.png"), fig, px_per_unit = 3)
    end

    if showplot
        display(fig)
    end

    return fig
end
