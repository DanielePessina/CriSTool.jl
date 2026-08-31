# Plotting functions


## Set Font

function _resolve_plot_savedir(savedir::Union{Nothing, AbstractString},
                               default_subdir::AbstractString)
    return isnothing(savedir) ? nothing : String(savedir)
end

_palette(palette_name::Symbol; alpha::Real = 1.0) =
    alpha == 1.0 ? Makie.to_colormap(palette_name) :
    [Makie.RGBAf(c, alpha) for c in Makie.to_colormap(palette_name)]

function _save_plot(figure, savename::AbstractString,
                    savedir::Union{Nothing, AbstractString},
                    suffix::AbstractString = "")
    savename != "" || return nothing
    output_dir = _resolve_plot_savedir(savedir, "Saved Plots")
    output_dir === nothing && return nothing
    mkpath(output_dir)
    save(joinpath(output_dir, "$(savename)$(suffix).png"), figure, px_per_unit = 3)
    return nothing
end

"""
    plot_posterior_pairplot(posterior_df[, prior_df], truth_params,
                            mean_params; title = "", savename = "",
                            savedir = nothing, lossfunction_string = "logMLE")

Plot pairwise parameter distributions from posterior samples. When a
`prior_df` is supplied the prior densities are overlayed.

# Arguments
- `posterior_df::DataFrame`: columns correspond to parameters; the last
  column typically contains the loss value. Each column should have
  `n_samples` rows.
- `prior_df::DataFrame`: optional DataFrame with the same column layout
  used to plot prior densities.
- `truth_params`, `mean_params`: dictionaries mapping parameter symbols
  to reference values.
- `title`, `savename`, `lossfunction_string`: optional plot metadata.

# Returns
The generated `Makie.Figure`. If `savename` is non-empty and `savedir` is
set the figure is saved to `savedir`.
"""
function plot_posterior_pairplot(posterior_df::DataFrame, prior_df::DataFrame,
                                 truth_params::Dict, mean_params::Dict; title = "",
                                 savename = "", lossfunction_string = "logMLE",
                                 savedir::Union{Nothing, AbstractString} = nothing,
                                 showplot::Bool = true)

    fontfile = joinpath(@__DIR__, "..", "..", "PaperMono-Regular.ttf")
    Makie.with_theme(Makie.Theme(fonts = (; regular = fontfile, bold = fontfile,
                                          italic = fontfile, bolditalic = fontfile))) do

        ms_cmap = Makie.cgrad([Makie.wong_colors(0.1)[3], Makie.wong_colors(0.9)[3]],
                              [
                                  minimum(posterior_df[!, end]),
                                  maximum(posterior_df[!, end])
                              ])
        figure = Makie.Figure(size = (750, 790))
        axis_limits = Dict{Symbol,NamedTuple}()
        for name in propertynames(posterior_df)
            col_values = collect(skipmissing(posterior_df[!, name]))
            isempty(col_values) && continue
            low, high = extrema(col_values)
            axis_limits[name] = (; lims=(; low = low, high = high))
        end

        PairPlots.pairplot(figure[1, 1],
                           posterior_df => (PairPlots.Scatter(),
                                            PairPlots.HexBin(colormap = ms_cmap),
                                            PairPlots.Contour(),
                                            PairPlots.MarginHist(color = (Makie.wong_colors()[1],
                                                                          0.4)),
                                            PairPlots.MarginDensity(color = Makie.wong_colors()[1],
                                                                    linewidth = 3),
                                            PairPlots.MarginQuantileText(),
                                            PairPlots.MarginQuantileLines()),
                           prior_df => (PairPlots.MarginDensity(color = (:black, 0.95)),),
                           labels = Dict(name => string(name)
                                         for name in propertynames(posterior_df)),
                           axis = axis_limits,
                           PairPlots.Truth(truth_params, color = Makie.wong_colors()[2],
                                           linewidth = 3),
                           PairPlots.Truth(mean_params, color = Makie.wong_colors()[4],
                                           linewidth = 2), fullgrid = false,
                           bodyaxis = (; xgridvisible = true, ygridvisible = true,
                                       xminorgridvisible = true,))

        cblimits = [minimum(posterior_df[!, end]), maximum(posterior_df[!, end])]
        Makie.Colorbar(figure[3, 1], colormap = ms_cmap, vertical = false, labelsize = 16,
                       ticklabelsize = 16, limits = cblimits,
                       ticks = (cblimits,
                                [
                                    Makie.rich("Min."),
                                    Makie.rich(lossfunction_string,
                                               Makie.subscript("C.I."))
                                ]), width = 500, tellwidth = false)
        Makie.Legend(figure[2, 1],
                     [
                         Makie.PolyElement(color = Makie.wong_colors()[1]),
                         Makie.LineElement(color = Makie.wong_colors()[2],
                                           linestyle = :solid,
                                           linewidth = 3),
                         Makie.LineElement(color = Makie.wong_colors()[4],
                                           linestyle = :solid,
                                           linewidth = 2),
                         Makie.LineElement(color = :black, linestyle = :solid,
                                           linewidth = 2)
                     ],
                     [
                         "Parameter Posteriors",
                         "Optimal Parameters",
                         "Mean Parameters",
                         "Prior"
                     ],
                     orientation = :horizontal, framevisible = false)

        Makie.Label(figure[1, 1][0, :], title)

        if savename != ""
            _save_plot(figure, savename, savedir, " Pairplot")
        end

        if showplot
            display(figure)
        end

        return figure
    end
end
"""
Variant of [`plot_posterior_pairplot`](@ref) without a prior dataset.
"""
function plot_posterior_pairplot(posterior_df::DataFrame,
                                 truth_params::Dict, mean_params::Dict; title = "",
                                 savename = "", lossfunction_string = "logMLE",
                                 savedir::Union{Nothing, AbstractString} = nothing,
                                 showplot::Bool = true)


    color_palette = :Set2_8
    fontfile = joinpath(@__DIR__, "..", "..", "PaperMono-Regular.ttf")
    ms_cmap = Makie.cgrad([
                              _palette(color_palette; alpha = 0.1)[3],
                              _palette(color_palette; alpha = 0.9)[3]
                          ],
                          [minimum(posterior_df[!, end]), maximum(posterior_df[!, end])])

    Makie.with_theme(Makie.Theme(fonts = (; regular = fontfile, bold = fontfile,
                                          italic = fontfile, bolditalic = fontfile))) do
        figure = Makie.Figure(size = (1000, 1000))

        PairPlots.pairplot(figure[1, 1],
                           posterior_df => (PairPlots.Scatter(),
                                            PairPlots.HexBin(colormap = ms_cmap),
                                            PairPlots.Contour(),
                                            PairPlots.MarginHist(color = (_palette(color_palette;
                                                                                        alpha = 0.4)[1])),
                                            PairPlots.MarginDensity(color = _palette(color_palette;
                                                                                          alpha = 0.9)[1],
                                                                    linewidth = 3),
                                            PairPlots.MarginQuantileText(),
                                            PairPlots.MarginQuantileLines()),
                           #    prior_df => (PairPlots.MarginDensity(color = (:black, 0.95)),),
                           labels = Dict(name => string(name)
                                         for name in propertynames(posterior_df)),
                           PairPlots.Truth(truth_params,
                                           color = _palette(color_palette)[2],
                                           linewidth = 3),
                           PairPlots.Truth(mean_params,
                                           color = _palette(color_palette)[4],
                                           linewidth = 3), fullgrid = false,
                           bodyaxis = (; xgridvisible = true, ygridvisible = true,
                                       xminorgridvisible = true,))

        cblimits = [minimum(posterior_df[!, end]), maximum(posterior_df[!, end])]
        Makie.Colorbar(figure[3, 1], colormap = ms_cmap, vertical = false, labelsize = 16,
                       ticklabelsize = 16, limits = cblimits,
                       ticks = (cblimits,
                                [
                                    Makie.rich("Min."),
                                    Makie.rich(lossfunction_string,
                                               Makie.subscript("C.I."))
                                ]), width = 500, tellwidth = false)
        Makie.Legend(figure[2, 1],
                     [
                         Makie.PolyElement(color = _palette(color_palette)[1]),
                         Makie.LineElement(color = _palette(color_palette)[2],
                                           linestyle = :solid,
                                           linewidth = 3),
                         Makie.LineElement(color = _palette(color_palette)[4],
                                           linestyle = :solid,
                                           linewidth = 3),
                         Makie.LineElement(color = :black, linestyle = :solid,
                                           linewidth = 3)
                     ],
                     [
                         "Parameter Posteriors",
                         "Optimal Parameters",
                         "Mean Parameters",
                         "Prior"
                     ],
                     orientation = :horizontal, framevisible = false)

        Makie.Label(figure[1, 1][0, :], title)

        if savename != ""
            _save_plot(figure, savename, savedir, " Pairplot")
        end

        if showplot
            display(figure)
        end

        return figure
    end
end
"""
    plot_measurements_vs_ensemble(measurements, ensemble_results,
                                  optimal_solutions; kwargs...)

Compare experimental measurements against ensemble simulation results.

# Arguments
- `measurements::Vector{<:AbstractExperiment}`: experimental datasets,
  one per batch.
- `ensemble_results::Vector{<:Union{EnsembleFVSolution, EnsembleMoMSolution}}`:
  simulation outputs where each `concentration` field is a matrix of size
  `(ensemble_size, n_tsteps)`. QMOM uses the moment-based
  `EnsembleMoMSolution` representation.
- `optimal_solutions`: vector of `(problem, solution)` tuples for the
  mean or optimal parameter set.
- Keyword arguments control plot appearance and may include the kinetic
  functions and sampled parameters for annotation.

# Returns
A `Makie.Figure` with concentration and particle-size plots. If
`savename` is provided and `savedir` is set the figure is saved to
`savedir`.
"""
function linear_interpolate(x::AbstractVector, y::AbstractVector, t::Real)
    if t <= x[begin]
        return y[begin]
    elseif t >= x[end]
        return y[end]
    end

    idx = searchsortedlast(x, t)
    if idx >= length(x)
        return y[end]
    end

    x1 = x[idx]
    x2 = x[idx + 1]
    y1 = y[idx]
    y2 = y[idx + 1]
    return y1 + (y2 - y1) * (t - x1) / (x2 - x1)
end

function linear_interpolate(x::AbstractVector, y::AbstractVector, t::AbstractVector)
    return [linear_interpolate(x, y, ti) for ti in t]
end

"""
    resolve_experiment_color(colors, idx, color_palette, colouroffset)

Resolve the color for experiment at index `idx`. If `colors` is provided,
use it (either as a Vector indexed by position, or Dict keyed by experiment ID).
Falls back to `color_palette[idx + colouroffset]` if no explicit color.
"""
function resolve_experiment_color(colors, idx::Int, color_palette, colouroffset::Int)
    if colors === nothing
        return color_palette[idx + colouroffset]
    elseif colors isa AbstractDict
        return get(colors, idx, color_palette[idx + colouroffset])
    elseif colors isa AbstractVector && idx <= length(colors)
        return colors[idx]
    else
        return color_palette[idx + colouroffset]
    end
end

function build_parameter_summary_table(param_samples::AbstractMatrix,
                                       param_symbols::Vector{Symbol};
                                       sigdigits = 3)
    size(param_samples, 2) == 0 && return ""
    means = vec(mean(param_samples, dims = 2))
    stds = vec(std(param_samples, dims = 2))
    n_rows = min(length(param_symbols), length(means))

    rows = [[string(param_symbols[i]),
             "$(round(means[i], sigdigits = sigdigits)) ± $(round(stds[i], sigdigits = sigdigits))"]
            for i in 1:n_rows]
    data = permutedims(reduce(hcat, rows))
    header = (["Parameter", "Value"], ["", ""])

    io = IOBuffer()
    PrettyTables.pretty_table(io, data;
                              header = header,
                              tf = PrettyTables.tf_ascii_rounded,
                              alignment = :c,
                              linebreaks = true)
    return chomp(String(take!(io)))
end

"""
    format_plot_measurement_value(mean_value, variance_value;
                                  show_uncertainty::Bool = true,
                                  mean_sigdigits::Int = 3,
                                  std_sigdigits::Int = 2) -> String

Format a measured size value for thesis-table display, optionally suppressing the
reported uncertainty term.
"""
function format_plot_measurement_value(mean_value,
                                       variance_value;
                                       show_uncertainty::Bool = true,
                                       mean_sigdigits::Int = 3,
                                       std_sigdigits::Int = 2)
    mean_text = string(round(mean_value, sigdigits = mean_sigdigits))
    if !show_uncertainty
        return mean_text
    end

    std_value = sqrt(variance_value)
    return "$(mean_text) ± $(round(std_value, sigdigits = std_sigdigits))"
end

function _measured_size_observable(expt::CrystallisationExperiment,
                                   solution::AbstractSolution)
    preferred_names = solution isa CrystallisationFVSolution ? (:d50q, :d43) :
                      (:d43, :d50q)
    for name in preferred_names
        hasproperty(expt.observables, name) &&
            return name, getproperty(expt.observables, name)
    end
    return nothing, nothing
end

function _solution_observable_trajectory(solution::AbstractSolution, name::Symbol)
    trajectory = observable_values(solution, name)
    trajectory isa AbstractVector ||
        throw(ArgumentError("Simulated observable :$name must be a trajectory."))
    return trajectory
end

function _ensemble_size_fields(solution::EnsembleMoMSolution, ::Val{:d43})
    return solution.d43, solution.d43_mean, solution.d43_std, "D43"
end

function _ensemble_size_fields(solution::EnsembleFVSolution, ::Val{:d43})
    return solution.d43, solution.d43_mean, solution.d43_std, "D43"
end

function _ensemble_size_fields(solution::EnsembleFVSolution, ::Val{:d50q})
    return solution.d50q, solution.d50q_mean, solution.d50q_std, "D50"
end

function _ensemble_size_fields(solution::EnsembleMoMSolution, ::Val{:d50q})
    throw(ArgumentError("MoM ensembles do not provide d50q."))
end

function _ensemble_size_fields(solution::AbstractSolution, ::Nothing)
    throw(ArgumentError("No particle-size metric is available for $(typeof(solution))."))
end

function _ensemble_size_fields_for_experiment(ensemble_solution,
                                              experiment::CrystallisationExperiment,
                                              optimal_solution::AbstractSolution)
    measured_name, _ = _measured_size_observable(experiment, optimal_solution)
    size_name = measured_name === nothing ?
                (ensemble_solution isa EnsembleFVSolution ? :d50q : :d43) : measured_name
    return _ensemble_size_fields(ensemble_solution, Val(size_name))
end

"""
    build_parameter_value_table(parameters::AbstractVector{<:Real},
                                param_symbols::Vector{Symbol};
                                sigdigits::Int = 3) -> String

Build a PrettyTables string for a deterministic parameter vector.
"""
function build_parameter_value_table(parameters::AbstractVector{<:Real},
                                     param_symbols::Vector{Symbol};
                                     sigdigits::Int = 3)
    isempty(param_symbols) && return ""

    n_rows = min(length(param_symbols), length(parameters))
    rows = [[string(param_symbols[i]),
             string(round(parameters[i], sigdigits = sigdigits))]
            for i in 1:n_rows]
    data = permutedims(reduce(hcat, rows))
    header = (["Parameter", "Value"], ["", ""])

    io = IOBuffer()
    PrettyTables.pretty_table(io, data;
                              header = header,
                              tf = PrettyTables.tf_ascii_rounded,
                              alignment = :c,
                              linebreaks = true)
    return chomp(String(take!(io)))
end

"""
    build_simulation_thesis_table_data(measurements, solutions;
                                       show_measurement_uncertainty::Bool = true)
                                       -> NamedTuple

Build row data and labels for the simulation thesis table used by
`plot_measurements_vs_simulation`.
"""
function build_simulation_thesis_table_data(measurements,
                                            solutions;
                                            show_measurement_uncertainty::Bool = true)
    thesis_rows = Vector{Vector{String}}()
    size_label = "size"

    for m in eachindex(measurements)
        sol = solutions[m]
        pred_text = "NA"
        meas_text = "NA"

        size_name, measured_size = _measured_size_observable(measurements[m], sol)
        if measured_size !== nothing
            size_label = size_name === :d50q ? "d50" : string(size_name)
            predicted_size = _solution_observable_trajectory(sol, size_name)[end]
            pred_text = string(round(predicted_size, sigdigits = 3))
            final_variance = measured_size.variance === nothing ? nothing :
                             measured_size.variance[end]
            meas_text = final_variance === nothing ?
                        string(round(measured_size.mean[end], sigdigits = 3)) :
                        format_plot_measurement_value(measured_size.mean[end], final_variance;
                                                      show_uncertainty = show_measurement_uncertainty)
        end

        temp_str = string(round(measurements[m].temperature - 273, digits = 2))
        exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m
        push!(thesis_rows, [string(exp_id), temp_str, pred_text, meas_text])
    end

    return (; rows = thesis_rows, size_label = size_label)
end

"""
    plot_measurements_vs_ensemble(measurements, ensemble_results, optimal_solutions; kwargs...)

Plot experimental measurements against ensemble simulation results, overlaying the
optimal (best-fit) simulation trajectory and the ensemble uncertainty band for each
experiment.

# Arguments
- `measurements::Vector{<:AbstractExperiment}`: experimental measurement sets.
- `ensemble_results::Vector{<:Union{EnsembleFVSolution, EnsembleMoMSolution}}`: ensemble
  simulation results (one per measurement), as returned by [`run_ensemble`](@ref).
- `optimal_solutions`: optimal simulation solutions (one per measurement), typically
  from [`runsimulation`](@ref) with best-fit parameters.

# Keyword Arguments
- `title::String`: figure title (default `""`).
- `savename::String`: file path (without extension) to save the figure; empty to skip.
- `colors`: per-experiment colors. Can be `nothing` (auto palette), a `Vector`, or a
  `Dict{Int,String}` keyed by experiment ID.
- `colouroffset::Int`: offset into the default colour palette (default `0`).
- `parameters`: best-fit parameter vector, used for the parameter summary table.
- `nucleationfunction`: nucleation model struct, used for symbol labels in the table.
- `growthfunction`: growth model struct, used for symbol labels in the table.
- `aggregationfunction`: aggregation model struct.
- `breakagefunction`: breakage model struct.
- `solver`: solver used for the simulations.
- `parameter_samples::AbstractMatrix`: parameter sample matrix (params × samples), used
  to compute and display parameter statistics in the table.
- `showtext::Bool`: show inline text annotations (default `true`).
- `show_thesistext::Bool`: show the thesis-style summary table below the plot
  (default `false`).
- `show_title::Bool`: display the figure title (default `true`).
- `showplot::Bool`: call `display` on the figure (default `true`).
- `figure_kwargs::NamedTuple`: extra keyword arguments forwarded to `Makie.Figure`.
- `axis_kwargs::NamedTuple`: extra keyword arguments forwarded to `Makie.Axis`.

# Returns
A `Makie.Figure` object. If `savename` is non-empty the figure is also saved as PNG.
"""
