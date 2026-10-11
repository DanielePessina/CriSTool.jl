"""
Posterior pair plots through PairPlots.jl. Loaded when PairPlots (and hence
Makie) is loaded alongside CriSTool.
"""
module CriSToolPairPlotsExt

using CriSTool
using CriSTool: CRISTOOL_PRIOR_PLOT_SAMPLES
import CriSTool: plot_posterior_pairplot
import Makie
import PairPlots

_named_sample_columns(column_names, sample_matrix) =
    NamedTuple{column_names}(Tuple(collect.(eachrow(sample_matrix))))

# Optional overlays are empty tuples when not requested, so the series list stays a tuple.
_prior_pairplot_series(::Nothing, column_names) = ()
function _prior_pairplot_series(prior, column_names)
    # A univariate prior draws a vector; both shapes become dimensions x draws.
    prior_draws = reshape(rand(prior, CRISTOOL_PRIOR_PLOT_SAMPLES), :, CRISTOOL_PRIOR_PLOT_SAMPLES)
    size(prior_draws, 1) == length(column_names) ||
        throw(DimensionMismatch("The prior must have one dimension per sampled parameter."))
    return (PairPlots.Series(_named_sample_columns(column_names, prior_draws); label = "Prior",
        color = :black) => (PairPlots.MarginDensity(),),)
end

_reference_pairplot_series(::Nothing, column_names) = ()
function _reference_pairplot_series(reference, column_names)
    length(reference) == length(column_names) ||
        throw(DimensionMismatch("Supply one reference value per sampled parameter."))
    return (PairPlots.Truth(NamedTuple{column_names}(Tuple(reference));
        label = "Reference", color = Makie.wong_colors()[2]),)
end

function plot_posterior_pairplot(samples::ParameterSamples; prior = nothing, reference = nothing,
                                 title::AbstractString = "")
    column_names = Tuple(Symbol.(samples.names))
    posterior_series = PairPlots.Series(_named_sample_columns(column_names, samples.values);
        label = "Posterior", color = Makie.wong_colors()[1]) =>
        (PairPlots.Scatter(), PairPlots.Contour(), PairPlots.MarginDensity(), PairPlots.MarginQuantileLines())
    figure = Makie.Figure(size = (220 * length(column_names) + 200, 220 * length(column_names) + 120))
    PairPlots.pairplot(figure[1, 1], posterior_series, _prior_pairplot_series(prior, column_names)...,
        _reference_pairplot_series(reference, column_names)...;
        labels = Dict(column_name => string(column_name) for column_name in column_names))
    isempty(title) || Makie.Label(figure[0, 1], title; tellwidth = false)
    return figure
end

end
