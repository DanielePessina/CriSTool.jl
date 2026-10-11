"""
Makie plot conversions for solutions, plots of prediction ensembles against measurements and MCMC chain
diagnostics. Loaded when a Makie backend (for example `using CairoMakie`) is
loaded alongside CriSTool.
"""
module CriSToolMakieExt

using CriSTool
using CriSTool: AbstractDiscretisedSolver
import CriSTool: plot_predictions, plot_chain_diagnostics
import MCMCChains
import Makie
using Statistics: median, quantile

_plot_time_minutes(values) = values ./ 60
_plot_size_micrometres(values) = values .* 1e6

# Size metrics (d10, d32, d43, d50q, ...) are stored in metres and plotted in µm.
_is_size_observable(observable_name::Symbol) = occursin(r"^d\d+q?$", string(observable_name))
_observable_axis_label(observable_name::Symbol) =
    _is_size_observable(observable_name) ? "$(observable_name) [µm]" : string(observable_name)

_plotted_observables(observable_name::Symbol) = (observable_name,)
_plotted_observables(observable_names) = Tuple(Symbol.(observable_names))

function plot_predictions(predictions::PredictionResult, experiments = nothing;
                          observables = (:concentration, :d43), quantiles = (0.05, 0.95))
    plotted_observables = _plotted_observables(observables)
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
        time_minutes = _plot_time_minutes(ensemble.t)
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
    return figure
end

# --- Plot conversion traits ---------------------------------------------------
# `lines(solution)`, `lines(solution, :d43)` and the same calls with `scatter`,
# `stairs`, `scatterlines` ... convert a solution to (time, observable) points in
# SI units (s, kg/m³, m). `plot(solution)` defaults to a line. The mesh form
# `lines(solution, solver, time_index)` draws the number density at one saved
# time against the cell centres of the discretised `solver` that produced it.

Makie.plottype(::CrystallisationSolution) = Makie.Lines

function _solution_time_series_points(solution::CrystallisationSolution, observable_name::Symbol)
    observable_series = observable_values(solution, observable_name)
    observable_series isa AbstractVector && length(observable_series) == length(solution.t) ||
        throw(ArgumentError("Observable :$observable_name is not a per-time scalar series and " *
                            "cannot be plotted against time."))
    return Makie.Point2d.(solution.t, observable_series)
end

Makie.convert_arguments(::Makie.PointBased, solution::CrystallisationSolution) =
    (_solution_time_series_points(solution, :concentration),)
Makie.convert_arguments(::Makie.PointBased, solution::CrystallisationSolution, observable_name::Symbol) =
    (_solution_time_series_points(solution, observable_name),)

function Makie.convert_arguments(::Makie.PointBased, solution::CrystallisationSolution{<:MeshPopulation},
                                 solver::AbstractDiscretisedSolver, time_index::Integer)
    number_density = solution.population.numberdensity
    length(solver.cell_centre) == size(number_density, 1) ||
        throw(DimensionMismatch("solver has $(length(solver.cell_centre)) cells but the solution " *
                                "has $(size(number_density, 1))."))
    time_index in axes(number_density, 2) ||
        throw(BoundsError(solution.t, time_index))
    return (Makie.Point2d.(solver.cell_centre, view(number_density, :, time_index)),)
end

_chain_colour(chain_index) = Makie.wong_colors()[mod1(chain_index, length(Makie.wong_colors()))]

function plot_chain_diagnostics(chain::MCMCChains.Chains; burnin::Integer = 0,
                                title::AbstractString = "")
    chain_subset = burnin > 0 ? chain[(burnin + 1):end, :, :] : chain
    parameter_names = collect(names(chain_subset, :parameters))
    parameter_count = length(parameter_names)
    chain_iterations = collect(range(chain_subset))

    figure = Makie.Figure(size = (1000, 260 * parameter_count))
    isempty(title) || Makie.Label(figure[0, :], title)
    for (row_index, parameter_name) in enumerate(parameter_names)
        # Iterations x chains for this parameter.
        parameter_draws = Array(chain_subset[:, parameter_name, :])
        is_last_row = row_index == parameter_count

        trace_axis = Makie.Axis(figure[row_index, 1]; ylabel = string(parameter_name))
        for (chain_index, chain_draws) in enumerate(eachcol(parameter_draws))
            Makie.lines!(trace_axis, chain_iterations, chain_draws; color = _chain_colour(chain_index))
        end
        is_last_row ? (trace_axis.xlabel = "Iteration") :
            Makie.hidexdecorations!(trace_axis; grid = false)

        density_axis = Makie.Axis(figure[row_index, 2])
        for (chain_index, chain_draws) in enumerate(eachcol(parameter_draws))
            Makie.density!(density_axis, chain_draws; color = (_chain_colour(chain_index), 0.3),
                strokewidth = 1.5, strokecolor = _chain_colour(chain_index))
        end
        Makie.hideydecorations!(density_axis)
        is_last_row ? (density_axis.xlabel = "Parameter estimate") :
            Makie.hidexdecorations!(density_axis; grid = false)
    end

    return figure
end

end
