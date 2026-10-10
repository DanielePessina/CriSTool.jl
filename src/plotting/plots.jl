"""
Plots of prediction ensembles against measurements and of posterior parameter
draws. Both consume the model-workflow results (`PredictionResult`,
`InferenceResult`, `ParameterSamples`).
"""

_plot_time_minutes(values) = values ./ 60
_plot_size_micrometres(values) = values .* 1e6

# Size metrics (d10, d32, d43, d50q, ...) are stored in metres and plotted in µm.
_is_size_observable(observable_name::Symbol) = occursin(r"^d\d+q?$", string(observable_name))
_observable_axis_label(observable_name::Symbol) =
    _is_size_observable(observable_name) ? "$(observable_name) [µm]" : string(observable_name)

function _save_plot(figure, savename::AbstractString,
                    savedir::Union{Nothing, AbstractString}, suffix::AbstractString = "")
    (isempty(savename) || savedir === nothing) && return nothing
    mkpath(savedir)
    Makie.save(joinpath(savedir, "$(savename)$(suffix).png"), figure; px_per_unit = 3)
    return nothing
end

"""
    plot_predictions(predictions::PredictionResult, experiments = nothing;
                     observables = (:concentration, :d43), quantiles = (0.05, 0.95),
                     savename = "", savedir = nothing, showplot = true) -> Makie.Figure

Plot predicted trajectories with one row per predicted run and one column per
observable. Each panel shows the median over successful draws and, when there
is more than one successful draw, the band between the two `quantiles`. Failed
draws are excluded and counted in the panel title. When `experiments` is given,
runs are matched by `exp_id` and measured means are overlaid, with
± one standard deviation error bars where variances are supplied. Time is shown
in minutes and particle sizes in µm. The figure is saved to
`joinpath(savedir, savename * " Predictions.png")` when both are set.
"""
function plot_predictions(predictions::PredictionResult, experiments = nothing;
                          observables = (:concentration, :d43), quantiles = (0.05, 0.95),
                          savename::AbstractString = "",
                          savedir::Union{Nothing, AbstractString} = nothing,
                          showplot::Bool = true)
    plotted_observables = observables isa Symbol ? (observables,) : Tuple(Symbol.(observables))
    lower_quantile, upper_quantile = quantiles
    0 <= lower_quantile < upper_quantile <= 1 ||
        throw(ArgumentError("quantiles must be an increasing pair within [0, 1]."))
    measured_runs = experiments === nothing ? Dict{Int, CrystallisationExperiment}() :
        Dict(measured_run.exp_id => measured_run for measured_run in experiments)
    prediction_colour, measurement_colour = Makie.wong_colors()[1], Makie.wong_colors()[2]
    figure = Makie.Figure(size = (380 * length(plotted_observables), 280 * length(predictions.ensembles)))
    for (row_index, (experiment_id, ensemble)) in
        enumerate(zip(predictions.experiments, predictions.ensembles))
        successful_draws = ensemble.success
        any(successful_draws) ||
            throw(ArgumentError("Experiment $experiment_id has no successful prediction draws."))
        failed_count = count(!, successful_draws)
        panel_title = failed_count == 0 ? "Experiment $experiment_id" :
            "Experiment $experiment_id ($failed_count failed draws excluded)"
        measured_run = get(measured_runs, experiment_id, nothing)
        time_minutes = _plot_time_minutes(ensemble.time)
        for (column_index, observable_name) in enumerate(plotted_observables)
            hasproperty(ensemble.predictions, observable_name) ||
                throw(ArgumentError("Experiment $experiment_id has no predicted :$observable_name."))
            unit_scale = _is_size_observable(observable_name) ? _plot_size_micrometres : identity
            prediction_axis = Makie.Axis(figure[row_index, column_index]; title = panel_title,
                xlabel = "Time [min]", ylabel = _observable_axis_label(observable_name))
            draw_values = unit_scale(observable_values(ensemble, observable_name)[successful_draws, :])
            if size(draw_values, 1) > 1
                Makie.band!(prediction_axis, time_minutes,
                    [quantile(saved_values, lower_quantile) for saved_values in eachcol(draw_values)],
                    [quantile(saved_values, upper_quantile) for saved_values in eachcol(draw_values)];
                    color = (prediction_colour, 0.25))
            end
            Makie.lines!(prediction_axis, time_minutes, [median(saved_values) for saved_values in eachcol(draw_values)];
                color = prediction_colour, linewidth = 2)
            measured_run === nothing && continue
            hasproperty(measured_run.observables, observable_name) || continue
            measured_series = getproperty(measured_run.observables, observable_name)
            measured_time = _plot_time_minutes(measured_series.time)
            measured_mean = unit_scale(measured_series.mean)
            measured_series.variance === nothing ||
                Makie.errorbars!(prediction_axis, measured_time, measured_mean,
                    unit_scale(sqrt.(measured_series.variance)); color = measurement_colour, whiskerwidth = 6)
            Makie.scatter!(prediction_axis, measured_time, measured_mean; color = measurement_colour)
        end
    end
    _save_plot(figure, savename, savedir, " Predictions")
    showplot && display(figure)
    return figure
end

"""
    plot_posterior_pairplot(samples::ParameterSamples; prior = nothing, reference = nothing,
                            title = "", savename = "", savedir = nothing, showplot = true)
    plot_posterior_pairplot(result::InferenceResult; kwargs...)

Pair plot of joint parameter draws labelled by their named paths (for example
`nucleation.lnA`). `prior` is a joint distribution in the same order whose
marginals are overlaid; for an `InferenceResult` it defaults to the
specification's priors. `reference` is a vector of values in the same order,
drawn as reference lines (for example the true values of a synthetic study).
The figure is saved to `joinpath(savedir, savename * " Pairplot.png")` when both
are set.
"""
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
    _save_plot(figure, savename, savedir, " Pairplot")
    showplot && display(figure)
    return figure
end

plot_posterior_pairplot(result::InferenceResult; kwargs...) =
    plot_posterior_pairplot(result.samples;
        prior = product_distribution(last.(_named_parameter_leaves(result.preparation.specification.priors))),
        kwargs...)
