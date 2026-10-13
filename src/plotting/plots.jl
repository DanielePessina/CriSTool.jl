"""
Plotting entry points. The methods live in package extensions so that
`using CriSTool` does not load Makie: `using CairoMakie` (or another Makie
backend) loads `CriSToolMakieExt` (`plot_predictions`, `plot_chain_diagnostics`),
and `using PairPlots` additionally loads `CriSToolPairPlotsExt`
(`plot_posterior_pairplot`). Every plotting function returns a `Makie.Figure`
and never saves or displays it; save with Makie's `save(path, figure)`.
"""

"""
    plot_predictions(predictions::PredictionResult, experiments = nothing;
                     observables = (:concentration, :d43), quantiles = (0.05, 0.95)) -> Makie.Figure

Plot predicted trajectories with one row per predicted run and one column per
observable. Each panel shows the median over successful draws and, when there
is more than one successful draw, the band between the two `quantiles`. Failed
draws are excluded and counted in the panel title. When `experiments` is given,
runs are matched by `exp_id` and measured means are overlaid, with
± one standard deviation error bars where variances are supplied. Time is shown
in minutes and particle sizes in µm. Save the returned figure with
`save(path, figure)`.

Requires a Makie backend: run `using CairoMakie` first.
"""
function plot_predictions end

"""
    plot_posterior_pairplot(samples::ParameterSamples; prior = nothing, reference = nothing,
                            title = "") -> Makie.Figure
    plot_posterior_pairplot(result::InferenceResult; kwargs...)

Pair plot of joint parameter draws labelled by their named paths (for example
`nucleation.lnA`). `prior` is a joint distribution in the same order whose
marginals are overlaid; for an `InferenceResult` it defaults to the
specification's priors. `reference` is a vector of values in the same order,
drawn as reference lines (for example the true values of a synthetic study).
An empty `title` draws no title. Save the returned figure with
`save(path, figure)`.

Requires PairPlots and a Makie backend: run `using CairoMakie, PairPlots` first.
"""
function plot_posterior_pairplot end

plot_posterior_pairplot(result::InferenceResult; kwargs...) =
    plot_posterior_pairplot(result.samples;
        prior = product_distribution(last.(_named_parameter_leaves(result.preparation.specification.priors))),
        kwargs...)

"""
    plot_chain_diagnostics(chain; burnin = 0, title = "") -> Makie.Figure

Trace and density plots of every parameter of an `MCMCChains.Chains`, one row
per parameter and one line per chain. The first `burnin` iterations are
discarded; an empty `title` draws no title. Save the returned figure with
`save(path, figure)`.

Requires a Makie backend: run `using CairoMakie` first.
"""
function plot_chain_diagnostics end

# Extension that defines each plotting function's methods, and how to load it.
_plotting_extension(plotting_function) = nothing
_plotting_extension(::typeof(plot_predictions)) =
    (:CriSToolMakieExt, "`using CairoMakie` (or another Makie backend)")
_plotting_extension(::typeof(plot_chain_diagnostics)) =
    (:CriSToolMakieExt, "`using CairoMakie` (or another Makie backend)")
_plotting_extension(::typeof(plot_posterior_pairplot)) =
    (:CriSToolPairPlotsExt, "`using CairoMakie, PairPlots`")

"""MethodError hint naming the packages to load when a plotting extension is missing."""
function _plotting_method_error_hint(io::IO, method_error::MethodError, argument_types, keyword_arguments)
    plotting_extension = _plotting_extension(method_error.f)
    plotting_extension === nothing && return nothing
    extension_name, load_instruction = plotting_extension
    Base.get_extension(@__MODULE__, extension_name) === nothing || return nothing
    print(io, "\nCriSTool plotting is provided by a package extension; run ", load_instruction,
          " to load it.")
    return nothing
end
