"""
Task-local state, the Turing-extension entry points of
`fit(::PreparedFit{<:Any, <:BayesianSpec})`, and chain renaming. Chain trace/density plots (`plot_chain_diagnostics`) live in
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

"""A bounded pool of `copy_count` (at least one) private deep copies of `template`."""
function _private_copy_pool(template::T, copy_count::Integer) where {T}
    pool = Channel{T}(max(copy_count, 1))
    for _ in 1:copy_count
        put!(pool, deepcopy(template))
    end
    return pool
end

"""Run `evaluate(private_copy)` on a copy borrowed from `pool`, returning it afterwards."""
function _with_private_copy(evaluate, pool::Channel)
    private_copy = take!(pool)
    try
        return evaluate(private_copy)
    finally
        put!(pool, private_copy)
    end
end

"""Task-local memoized initializer, ported from `Base.OncePerTask` (1.12+)."""
struct _CristoolOncePerTask{T, F}
    initializer::F
end
@inline function (_opt::_CristoolOncePerTask{T, F})() where {T, F}
    get!(_opt.initializer, task_local_storage(), _opt)::T
end

"""Build the Turing model with explicit priors for a Bayesian prepared fit; an optional
second argument selects the chain ensemble (`MCMCSerial()` or `MCMCThreads()`).

Methods are provided by the `CriSToolTuringExt` package extension; load it with
`using Turing`.
"""
function nuts_model end

"""Throw a loading instruction when the Turing extension is absent."""
function _require_turing_extension()
    Base.get_extension(@__MODULE__, :CriSToolTuringExt) === nothing &&
        throw(ArgumentError("Sampling a BayesianSpec requires Turing; run `using Turing` " *
                            "to load the CriSToolTuringExt extension."))
    return nothing
end

"""Sample the NUTS posterior chain of a Bayesian prepared fit (`CriSToolTuringExt`)."""
function _nuts_posterior_chain end

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
