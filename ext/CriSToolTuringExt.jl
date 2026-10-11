"""
Turing-backed NUTS sampling for `BayesianSpec`. Loaded when Turing is: it
supplies the DynamicPPL loss model, the `nuts_model` and NUTS chain methods
owned by CriSTool, the `nothing` sampler default (`Turing.NUTS()`), and the
guard against passing a DynamicPPL model to the vendored AIS sampler.
"""
module CriSToolTuringExt

import AbstractMCMC
import Random
import Turing
using CriSTool: CriSTool, BayesianSpec, PreparedFit, loss
using CriSTool: nuts_model, _nuts_posterior_chain, _cristool_once_per_task,
                _named_parameter_leaves
using CriSTool.KissABC: AIS

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


"""Resolve the sampler of a `BayesianSpec`: `nothing` means `Turing.NUTS()`."""
_nuts_sampler(::Nothing) = Turing.NUTS()
_nuts_sampler(sampler) = sampler

# Serial sampling evaluates on the caller's task, so the fit is used directly and no
# copy stays pinned in the caller's task-local storage. Threaded chains each take a
# private per-task copy of the mutable solve templates.
_fit_per_task(prepared_fit, ::Turing.MCMCSerial) = Base.Returns(prepared_fit)
_fit_per_task(prepared_fit, ::Turing.MCMCThreads) =
    _cristool_once_per_task(typeof(prepared_fit)) do
        deepcopy(prepared_fit)
    end

_chain_ensemble(threaded::Bool) = threaded ? Turing.MCMCThreads() : Turing.MCMCSerial()

CriSTool.nuts_model(prepared_fit::PreparedFit{<:Any, <:BayesianSpec}) =
    nuts_model(prepared_fit, Turing.MCMCSerial())

function CriSTool.nuts_model(prepared_fit::PreparedFit{<:Any, <:BayesianSpec},
        ensemble::Union{Turing.MCMCSerial, Turing.MCMCThreads})
    prior_values = last.(_named_parameter_leaves(prepared_fit.specification.priors))
    return _cristool_nuts_loss_model(prior_values, _fit_per_task(prepared_fit, ensemble))
end

function CriSTool._nuts_posterior_chain(prepared_fit::PreparedFit{<:Any, <:BayesianSpec},
        rng::Random.AbstractRNG, threaded::Bool; kwargs...)
    fit_specification = prepared_fit.specification
    ensemble = _chain_ensemble(threaded)
    return Turing.sample(rng, nuts_model(prepared_fit, ensemble), _nuts_sampler(fit_specification.sampler),
        ensemble, fit_specification.n_samples, fit_specification.n_chains; kwargs...)
end

# Resolve the method ambiguity between the vendored AIS step and DynamicPPL's
# `step(::AbstractRNG, ::DynamicPPL.Model, ::AbstractMCMC.AbstractSampler)`.
# The vendored ABC machinery only supports `AbstractDensity` models, so a
# DynamicPPL/Turing model is a hard misuse: fail loudly instead of dispatching
# ambiguously into Turing's sampler path.
function AbstractMCMC.step(rng::Random.AbstractRNG,
                           model::Turing.DynamicPPL.Model,
                           spl::AIS;
                           kwargs...)
    error("AIS sampling requires a KissABC density (ApproxPosterior / " *
          "CommonLogDensity), not a DynamicPPL/Turing model. Use Turing's own " *
          "MCMC samplers for DynamicPPL models.")
end

end
