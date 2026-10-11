"""
Turing model construction for `fit(::PreparedFit{<:Any, <:BayesianSpec})`
and chain renaming. Chain trace/density plots (`ChainStatsPlots`) live in
the Makie extension.
"""

"""Build a task-local memoized initializer of element type `T`.

Julia >= 1.12 ships `Base.OncePerTask`, so this returns the native object
directly (zero custom code, explicit `T`).  On the 1.10/1.11 LTS floor that
type is absent (no Compat backport either), so it falls back to an
identical port: Base's own implementation of `OncePerTask` is exactly
`get!(f, task_local_storage(), self)::T` keyed by the object.

`T` is passed explicitly (never inferred) so the model's
`prepared_fit_per_task()` call site - which runs once per NUTS
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
Turing.@model function _cristool_nuts_loss_model_with_data(prior, prepared_fit_per_task)
    n = length(prior)
    θ = Vector{Float64}(undef, n)
    for i in 1:n
        θ[i] ~ prior[i]
    end
    Turing.@addlogprob!(-loss(prepared_fit_per_task(), θ))
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

function _cristool_nuts_loss_model(prior, prepared_fit_per_task)
    compiled_model = _cristool_nuts_loss_model_with_data(prior, prepared_fit_per_task)
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
