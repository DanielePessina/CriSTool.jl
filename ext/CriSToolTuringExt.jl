"""
Turing-backed posterior sampling of an `EstimationProblem`. Loaded when Turing
is: it supplies the DynamicPPL loss model, the `nuts_model` method owned by
CriSTool, `solve(problem, sampler)` for Turing samplers such as `NUTS()`, and
the guard against passing a DynamicPPL model to the vendored AIS sampler.
"""
module CriSToolTuringExt

import AbstractMCMC
import Random
import Turing
using CriSTool: CriSTool, EstimationProblem, loss, EnsembleSerial, EnsembleThreads
using CriSTool.SciMLBase: SciMLBase
using CriSTool: nuts_model, _cristool_once_per_task, _prior_distributions
using CriSTool.KissABC: AIS

# Keep the Turing model at module scope (DynamicPPL requirement). Its data
# arguments are captured by a zero-argument adapter below because the generated
# DynamicPPL 0.39 evaluator for nonempty model arguments references a global
# defined later in that dependency; Julia 1.12+ rejects that world-age access.
Turing.@model function _cristool_nuts_loss_model_with_data(prior, estimation_problem_per_task)
    n = length(prior)
    θ = Vector{Float64}(undef, n)
    for i in 1:n
        θ[i] ~ prior[i]
    end
    Turing.@addlogprob!(-loss(estimation_problem_per_task(), θ))
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

function _cristool_nuts_loss_model(prior, estimation_problem_per_task)
    compiled_model = _cristool_nuts_loss_model_with_data(prior, estimation_problem_per_task)
    evaluator = _CristoolNUTSModelEvaluator(
        compiled_model, Tuple(values(compiled_model.args)), compiled_model.defaults)
    return Turing.DynamicPPL.Model{false}(evaluator, NamedTuple(), NamedTuple())
end


# Serial sampling evaluates on the caller's task, so the problem is used directly and no
# copy stays pinned in the caller's task-local storage. Threaded chains each take a
# private per-task copy of the mutable solve templates.
_problem_per_task(estimation_problem, ::Turing.MCMCSerial) = Base.Returns(estimation_problem)
_problem_per_task(estimation_problem, ::Turing.MCMCThreads) =
    _cristool_once_per_task(typeof(estimation_problem)) do
        deepcopy(estimation_problem)
    end

_chain_ensemble(::EnsembleSerial) = Turing.MCMCSerial()
_chain_ensemble(::EnsembleThreads) = Turing.MCMCThreads()
_chain_ensemble(unsupported::SciMLBase.BasicEnsembleAlgorithm) = CriSTool._unsupported_parallel(unsupported)

CriSTool.nuts_model(estimation_problem::EstimationProblem) =
    nuts_model(estimation_problem, Turing.MCMCSerial())

function CriSTool.nuts_model(estimation_problem::EstimationProblem,
        ensemble::Union{Turing.MCMCSerial, Turing.MCMCThreads})
    return _cristool_nuts_loss_model(_prior_distributions(estimation_problem.priors),
        _problem_per_task(estimation_problem, ensemble))
end

"""
    solve(problem::EstimationProblem, sampler::AbstractMCMC.AbstractSampler;
          n_samples=1000, n_chains=1, rng=default_rng(), parallel=EnsembleSerial(), kwargs...)
        -> InferenceResult

Sample the posterior of the problem's priors and likelihood loss with a Turing
sampler such as `NUTS()`: `n_chains` chains of `n_samples` draws each. With
`n_chains > 1`, `parallel = EnsembleThreads()` runs the chains on threads
(`MCMCThreads`); with one chain it solves the experiments inside each
log-density evaluation on threads instead. Other keywords go to `Turing.sample`
(e.g. `progress`). The likelihood uses supplied measurement variances; noise is
never fitted.
"""
function SciMLBase.solve(estimation_problem::EstimationProblem, sampler::AbstractMCMC.AbstractSampler;
        n_samples::Integer = 1000, n_chains::Integer = 1, rng::Random.AbstractRNG = Random.default_rng(),
        parallel::SciMLBase.BasicEnsembleAlgorithm = EnsembleSerial(), sample_options...)
    CriSTool._validate_posterior_sampling(estimation_problem, n_samples, n_chains)
    chain_parallel, split_problem = CriSTool._split_parallel(estimation_problem,
        CriSTool._chain_parallel_level(n_chains), parallel)
    ensemble = _chain_ensemble(chain_parallel)
    posterior_chain = Turing.sample(rng, nuts_model(split_problem, ensemble), sampler, ensemble,
        n_samples, n_chains; sample_options...)
    return CriSTool._chain_inference_result(split_problem, sampler, posterior_chain)
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
