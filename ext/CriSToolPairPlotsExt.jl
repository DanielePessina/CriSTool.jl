"""
Posterior pair plots through PairPlots.jl. Loaded when PairPlots (and hence
Makie) is loaded alongside CriSTool.
"""
module CriSToolPairPlotsExt

using CriSTool
using CriSTool: _plot_output_path, CRISTOOL_PRIOR_PLOT_SAMPLES
import CriSTool: plot_posterior_pairplot
import Makie
import PairPlots

function plot_posterior_pairplot(samples::ParameterSamples; prior = nothing, reference = nothing,
                                 title::AbstractString = "", savename::AbstractString = "",
                                 savedir::Union{Nothing, AbstractString} = nothing,
                                 showplot::Bool = true)
    column_names = Tuple(Symbol.(samples.names))
    named_columns(sample_matrix) = NamedTuple{column_names}(Tuple(collect.(eachrow(sample_matrix))))
    plotted_series = Any[PairPlots.Series(named_columns(samples.values); label = "Posterior",
        color = Makie.wong_colors()[1]) =>
        (PairPlots.Scatter(), PairPlots.Contour(), PairPlots.MarginDensity(), PairPlots.MarginQuantileLines())]
    if prior !== nothing
        prior_draws = rand(prior, CRISTOOL_PRIOR_PLOT_SAMPLES)
        prior_draws isa AbstractVector && (prior_draws = reshape(prior_draws, 1, :))
        size(prior_draws, 1) == length(column_names) ||
            throw(DimensionMismatch("The prior must have one dimension per sampled parameter."))
        push!(plotted_series, PairPlots.Series(named_columns(prior_draws); label = "Prior",
            color = :black) => (PairPlots.MarginDensity(),))
    end
    if reference !== nothing
        length(reference) == length(column_names) ||
            throw(DimensionMismatch("Supply one reference value per sampled parameter."))
        push!(plotted_series, PairPlots.Truth(NamedTuple{column_names}(Tuple(reference));
            label = "Reference", color = Makie.wong_colors()[2]))
    end
    figure = Makie.Figure(size = (220 * length(column_names) + 200, 220 * length(column_names) + 120))
    PairPlots.pairplot(figure[1, 1], plotted_series...;
        labels = Dict(column_name => string(column_name) for column_name in column_names))
    isempty(title) || Makie.Label(figure[0, 1], title; tellwidth = false)
    output_path = _plot_output_path(savename, savedir, " Pairplot")
    if output_path !== nothing
        mkpath(dirname(output_path))
        Makie.save(output_path, figure; px_per_unit = 3)
    end
    showplot && display(figure)
    return figure
end

end
