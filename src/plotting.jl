# Plotting functions


## Set Font

function _resolve_plot_savedir(savedir::Union{Nothing, AbstractString},
                               default_subdir::AbstractString)
    return isnothing(savedir) ? joinpath(pwd(), default_subdir) : String(savedir)
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
The generated `Makie.Figure`. If `savename` is non-empty the figure is
saved to the `R - ABCDE Plots` directory.
"""
function plot_posterior_pairplot(posterior_df::DataFrame, prior_df::DataFrame,
                                 truth_params::Dict, mean_params::Dict; title = "",
                                 savename = "", lossfunction_string = "logMLE",
                                 savedir::Union{Nothing, AbstractString} = nothing,
                                 showplot::Bool = true)

    fontfile = joinpath(@__DIR__, "..", "PaperMono-Regular.ttf")
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
            output_dir = _resolve_plot_savedir(savedir, "R - ABCDE Plots")
            mkpath(output_dir)
            save(joinpath(output_dir, "$(savename) Pairplot.png"), figure, px_per_unit = 3)
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
    fontfile = joinpath(@__DIR__, "..", "PaperMono-Regular.ttf")
    ms_cmap = Makie.cgrad([
                              Plots.palette(color_palette, alpha = 0.1)[3],
                              Plots.palette(color_palette, alpha = 0.9)[3]
                          ],
                          [minimum(posterior_df[!, end]), maximum(posterior_df[!, end])])

    Makie.with_theme(Makie.Theme(fonts = (; regular = fontfile, bold = fontfile,
                                          italic = fontfile, bolditalic = fontfile))) do
        figure = Makie.Figure(size = (1000, 1000))

        PairPlots.pairplot(figure[1, 1],
                           posterior_df => (PairPlots.Scatter(),
                                            PairPlots.HexBin(colormap = ms_cmap),
                                            PairPlots.Contour(),
                                            PairPlots.MarginHist(color = (Plots.palette(color_palette,
                                                                                        alpha = 0.4)[1])),
                                            PairPlots.MarginDensity(color = Plots.palette(color_palette,
                                                                                          alpha = 0.9)[1],
                                                                    linewidth = 3),
                                            PairPlots.MarginQuantileText(),
                                            PairPlots.MarginQuantileLines()),
                           #    prior_df => (PairPlots.MarginDensity(color = (:black, 0.95)),),
                           labels = Dict(name => string(name)
                                         for name in propertynames(posterior_df)),
                           PairPlots.Truth(truth_params,
                                           color = Plots.palette(color_palette)[2],
                                           linewidth = 3),
                           PairPlots.Truth(mean_params,
                                           color = Plots.palette(color_palette)[4],
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
                         Makie.PolyElement(color = Plots.palette(color_palette)[1]),
                         Makie.LineElement(color = Plots.palette(color_palette)[2],
                                           linestyle = :solid,
                                           linewidth = 3),
                         Makie.LineElement(color = Plots.palette(color_palette)[4],
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
            output_dir = _resolve_plot_savedir(savedir, "R - ABCDE Plots")
            mkpath(output_dir)
            save(joinpath(output_dir, "$(savename) Pairplot.png"), figure, px_per_unit = 3)
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
- `measurements::Vector{<:AbstractMeasurements}`: experimental datasets,
  one per batch.
- `ensemble_results::Vector{<:Union{EnsembleFVSolution, EnsembleMoMSolution}}`:
  simulation outputs where each `concentration` field is a matrix of size
  `(ensemble_size, n_tsteps)`.
- `optimal_solutions`: vector of `(problem, solution)` tuples for the
  mean or optimal parameter set.
- Keyword arguments control plot appearance and may include the kinetic
  functions and sampled parameters for annotation.

# Returns
A `Makie.Figure` with concentration and particle-size plots. If
`savename` is provided the figure is saved to `R - ABCDE Plots`.
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

        if sol isa CrystallisationMoMSolution
            size_label = "d43"
            pred_text = string(round(get_characteristic_size(sol), sigdigits = 3))
            if hasproperty(measurements[m], :d43) && !isempty(measurements[m].d43)
                meas_text = format_plot_measurement_value(measurements[m].d43,
                                                          measurements[m].d43var;
                                                          show_uncertainty = show_measurement_uncertainty)
            end
        elseif sol isa CrystallisationFVSolution
            size_label = "d50"
            pred_text = string(round(get_characteristic_size(sol), sigdigits = 3))
            if hasproperty(measurements[m], :quantilemean) &&
               !isempty(measurements[m].quantilemean)
                meas_text = format_plot_measurement_value(measurements[m].quantilemean,
                                                          measurements[m].quantilevariance;
                                                          show_uncertainty = show_measurement_uncertainty)
            end
        end

        loading_str = string(round(measurements[m].loading, sigdigits = 3))
        temp_str = string(round(measurements[m].temperature - 273, digits = 2))
        exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m
        push!(thesis_rows, [string(exp_id), loading_str, temp_str, pred_text, meas_text])
    end

    return (; rows = thesis_rows, size_label = size_label)
end

"""
    plot_measurements_vs_ensemble(measurements, ensemble_results, optimal_solutions; kwargs...)

Plot experimental measurements against ensemble simulation results, overlaying the
optimal (best-fit) simulation trajectory and the ensemble uncertainty band for each
experiment.

# Arguments
- `measurements::Vector{<:AbstractMeasurements}`: experimental measurement sets.
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
function plot_measurements_vs_ensemble(measurements::Vector{<:AbstractMeasurements},
                                       ensemble_results::Vector{<:Union{EnsembleFVSolution,
                                                                        EnsembleMoMSolution}},
                                       optimal_solutions; title = "", savename = "",
                                       colors = nothing, colouroffset = 0, parameters = nothing,
                                       nucleationfunction = nothing,
                                       growthfunction = nothing,
                                       aggregationfunction = nothing,
                                       breakagefunction = nothing,
                                       solver = nothing,
                                       parameter_samples = nothing, showtext = true,
                                       show_thesistext = false, show_title::Bool = true,
                                       showplot::Bool = true,
                                       savedir::Union{Nothing, AbstractString} = nothing,
                                       figure_kwargs = (;), axis_kwargs = (;))


    thesis_rows = Vector{Vector{String}}()
    if show_thesistext
        for m in eachindex(measurements)
            ensemble_sol = ensemble_results[m]
            pred_label = ""
            pred_text = "NA"
            meas_text = "NA"

            if hasproperty(ensemble_sol, :d43_mean)
                pred_label = "D43"
                mean_pred_size = ensemble_sol.d43_mean[end]
                std_pred_size = ensemble_sol.d43_std[end]
                pred_text = "$(round(mean_pred_size, sigdigits=3)) ± $(round(std_pred_size, sigdigits=2))"

                if hasproperty(measurements[m], :d43) && !isempty(measurements[m].d43)
                    meas_mean = measurements[m].d43
                    meas_std = sqrt(measurements[m].d43var)
                    meas_text = "$(round(meas_mean, sigdigits=3)) ± $(round(meas_std, sigdigits=2))"
                end
            elseif hasproperty(ensemble_sol, :d50q_mean)
                pred_label = "D50"
                mean_pred_size = ensemble_sol.d50q_mean[end]
                std_pred_size = ensemble_sol.d50q_std[end]
                pred_text = "$(round(mean_pred_size, sigdigits=3)) ± $(round(std_pred_size, sigdigits=2))"

                if hasproperty(measurements[m], :quantilemean) &&
                   !isempty(measurements[m].quantilemean)
                    meas_mean = measurements[m].quantilemean
                    meas_std = sqrt(measurements[m].quantilevariance)
                    meas_text = "$(round(meas_mean, sigdigits=3)) ± $(round(meas_std, sigdigits=2))"
                end
            end

            loading_str = string(round(measurements[m].loading, sigdigits=3))
            temp_str = string(round(measurements[m].temperature - 273, digits = 2))
            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m
            push!(thesis_rows, [string(exp_id), loading_str, temp_str, pred_text, meas_text])
        end
    end

    fontfile = joinpath(@__DIR__, "..", "PaperMono-Regular.ttf")
    Makie.with_theme(fonts = (; regular = fontfile, bold = fontfile, italic = fontfile,
                              bold_italic = fontfile)) do

        # Calculate dynamic figure height based on content
        axis_height = 350        # Fixed - main plot stays consistent
        title_height = 80        # For figure[0, 1]
        legend_height = 50       # For figure[2, 1]
        table_row_height = 24    # Per row in thesis table (monospace text)
        table_header_height = 80 # Table borders + header

        thesis_table_height = show_thesistext && !isempty(thesis_rows) ?
            table_header_height + length(measurements) * table_row_height : 0

        param_table_height = 0
        if show_thesistext && parameter_samples !== nothing &&
           nucleationfunction !== nothing && growthfunction !== nothing
            n_params = length(vcat(nucleationfunction.symbols, growthfunction.symbols))
            param_table_height = table_header_height + n_params * table_row_height
        end

        total_height = title_height + axis_height + legend_height + thesis_table_height +
                       param_table_height + 40  # 40 for gaps

        figure_defaults = (; size = (700, total_height))
        figure = Makie.Figure(; merge(figure_defaults, figure_kwargs)...)

        # try
        #     fontfile = joinpath(pwd(), "PaperMono-Regular.ttf")
        #     Makie.set_theme!(fonts = (; regular = fontfile, bold = fontfile, italic = fontfile,
        #                               bold_italic = fontfile))
        # catch err
        #     @warn "RobotoSlab font not found, using default font."
        #     println("pwd is: ", pwd())
        # end

        ms_legendfontsize = 18
        ms_xtickfontsize = 18
        ms_ytickfontsize = 18
        ms_ylabelfontsize = 20
        ms_xlabelfontsize = 20
        ms_titlesize = 20

        ms_markersize = 24
        ms_linewidth = 5
        ms_whiskerwidth = 10
        ms_linewidtheb = 2

        color_palette = Plots.palette(:Set2_8)
        color_palette = repeat(collect(color_palette), 10)

        axis_defaults = (; xlabel = "Batch Time [min]",
                           ylabel = "Lysozyme concentration [mg/mL]",
                           titlesize = ms_titlesize,
                           xticklabelsize = ms_xtickfontsize,
                           yticklabelsize = ms_ytickfontsize,
                           ylabelsize = ms_ylabelfontsize,
                           xlabelsize = ms_xlabelfontsize,
                           yminorgridvisible = true,
                           xminorgridvisible = true,
                           yminorticks = Makie.IntervalsBetween(4),
                           xminorticks = Makie.IntervalsBetween(4),
                           limits = ((0,
                                      maximum([measurements[m].time[end]
                                               for m in eachindex(measurements)]) + 20),
                                     nothing))
        ax1 = Makie.Axis(figure[1, 1]; merge(axis_defaults, axis_kwargs)...)

        delta = 0.499
        alpha = 0.3

        for m in eachindex(measurements)
            particles = [Particles(Vector{Float64}(col))
                         for col in eachcol(ensemble_results[m].concentration)]
            Makie.band!(ax1, ensemble_results[m].time,
                        pquantile.(particles, 0.5 - delta),
                        pquantile.(particles, 0.5 + delta),
                        color = (resolve_experiment_color(colors, m, color_palette, colouroffset), alpha))
        end

        for m in eachindex(measurements)
            Makie.lines!(ax1, optimal_solutions[m][2].time,
                         optimal_solutions[m][2].concentration,
                         color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                         linewidth = ms_linewidth,
                         linestyle = :dash)
        end

        plot_elements = []
        labels = []

        param_string = ""

        # Collect particle size information
        size_info = String[]

        for m in eachindex(measurements)
            Makie.errorbars!(ax1, measurements[m].time, measurements[m].concentrationmean,
                             sqrt.(measurements[m].concentrationvariance),
                             color = :black,
                             whiskerwidth = ms_whiskerwidth,
                             linewidth = ms_linewidtheb)

            p = Makie.scatter!(ax1, measurements[m].time, measurements[m].concentrationmean,
                               color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                               markersize = ms_markersize,
                               strokewidth = 2)
            push!(plot_elements, p)

            ensemble_sol = ensemble_results[m]

            measured_size_str = ""
            predicted_size_str = ""

            if hasproperty(ensemble_sol, :d43_mean) # MoM solution
                mean_pred_size = ensemble_sol.d43_mean[end]
                std_pred_size = ensemble_sol.d43_std[end]
                predicted_size_str = "T = $(round(measurements[m].temperature-273,digits = 2) )°C, Load = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. D43 = $(round(mean_pred_size, sigdigits=3)) ± $(round(std_pred_size, sigdigits=2)) μm"

                if hasproperty(measurements[m], :d43) && !isempty(measurements[m].d43)
                    measured_size_str = "Meas. = $(round(measurements[m].d43, sigdigits=3)) ± $(round(sqrt(measurements[m].d43var), sigdigits=2)) μm"
                end
            elseif hasproperty(ensemble_sol, :d50q_mean) # FV solution
                mean_pred_size = ensemble_sol.d50q_mean[end]
                std_pred_size = ensemble_sol.d50q_std[end]
                predicted_size_str = "T = $(round(measurements[m].temperature-273,digits = 2) )°C, Load = $(round(measurements[m].loading, sigdigits=2)) g/L,Pred. D50 = $(round(mean_pred_size, sigdigits=3)) ± $(round(std_pred_size, sigdigits=2)) μm"

                if hasproperty(measurements[m], :quantilemean) &&
                   !isempty(measurements[m].quantilemean)
                    measured_size_str = "Meas. = $(round(measurements[m].quantilemean, sigdigits=3)) ± $(round(sqrt(measurements[m].quantilevariance), sigdigits=2)) μm"
                end
            end

            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m

            # Collect size information for text box
            if !isempty(predicted_size_str) && !isempty(measured_size_str)
                combined_str = "Exp. $(exp_id) $predicted_size_str, $measured_size_str"
                push!(size_info, combined_str)
            elseif !isempty(predicted_size_str)
                push!(size_info, "Exp. $(exp_id) $predicted_size_str")
            end

            new_label = "Exp. $(exp_id)"
            push!(labels, new_label)
        end

        # Add particle size information to parameter string
        if showtext && !isempty(size_info)
            if !isempty(param_string)
                param_string *= "\nParticle Sizes:\n"
            else
                param_string = "Particle Sizes:\n"
            end
            param_string *= join(size_info, "\n")
        end

        # Add text box with all information if we have any
        if !isempty(param_string) && showtext
            max_time = maximum([m.time[end] for m in measurements])
            max_conc = maximum([m.concentrationmean[1] for m in measurements])

            Makie.text!(ax1, max_time * 0.75, max_conc * 1.05, text = param_string,
                        align = (:center, :top),
                        fontsize = 12)
        end

        nbanks = length(labels) > 6 ? 2 : 1

        Makie.Legend(figure[2, 1], plot_elements, labels, orientation = :horizontal,
                     tellwidth = false, tellheight = true, nbanks = nbanks)

        if show_thesistext && !isempty(thesis_rows)
            thesis_io = IOBuffer()
            thesis_data = permutedims(reduce(hcat, thesis_rows))
            thesis_header = (["Exp ID", "Loading", "T", "Pred. d43", "Meas. d43"],
                             ["", "[g/L]", "[°C]", "[μm]", "[μm]"])
            PrettyTables.pretty_table(thesis_io, thesis_data;
                                      header = thesis_header,
                                      tf = PrettyTables.tf_ascii_rounded,
                                      alignment = :c,
                                      linebreaks = true)
            thesis_string = chomp(String(take!(thesis_io)))

            Makie.Label(figure[3, 1], thesis_string;
                        tellwidth = false,
                        tellheight = true,
                        halign = :center,
                        valign = :top,
                        fontsize = 16)
        end

        if show_thesistext && parameter_samples !== nothing &&
           nucleationfunction !== nothing && growthfunction !== nothing
            param_symbols = vcat(nucleationfunction.symbols, growthfunction.symbols)
            params_string = build_parameter_summary_table(parameter_samples,
                                                          param_symbols)
            if !isempty(params_string)
                Makie.Label(figure[4, 1], params_string;
                            tellwidth = false,
                            tellheight = true,
                            halign = :center,
                            valign = :top,
                            fontsize = 16)
            end
        end

        Makie.rowsize!(figure.layout, 1, Makie.Fixed(axis_height))
        Makie.rowsize!(figure.layout, 2, Makie.Auto())
        if show_thesistext && !isempty(thesis_rows)
            Makie.rowsize!(figure.layout, 3, Makie.Auto())
        end
        if show_thesistext && parameter_samples !== nothing &&
           nucleationfunction !== nothing && growthfunction !== nothing
            Makie.rowsize!(figure.layout, 4, Makie.Auto())
        end
        Makie.rowgap!(figure.layout, 10)

        if show_title
            Makie.Label(figure[0, 1], (title), tellwidth = false, tellheight = true)
        end

        if savename != ""
            output_dir = _resolve_plot_savedir(savedir, "Saved Plots")
            mkpath(output_dir)
            save(joinpath(output_dir, "$(savename).png"), figure, px_per_unit = 3)
        end

        if showplot
            display(figure)
        end

        return figure
    end
end

"""
    plot_ps_measurements_vs_ensemble(measurements, ensemble_results,
                                     optimal_solutions; kwargs...)

Compare measured particle sizes against ensemble simulation trajectories.

# Arguments
- `measurements::Vector{<:AbstractMeasurements}`: experimental datasets, one per
  batch.
- `ensemble_results::Vector{<:Union{EnsembleFVSolution, EnsembleMoMSolution}}`:
  ensemble simulation outputs containing particle-size trajectories.
- `optimal_solutions`: vector of `(problem, solution)` tuples for the nominal
  parameter set.
- `showtext`: toggle annotation textbox.

# Returns
A `Makie.Figure` showing particle-size ensembles, optimal simulations, and
measurements. Saves to `Saved Plots` when `savename` is provided.
"""
function plot_ps_measurements_vs_ensemble(measurements::Vector{<:AbstractMeasurements},
                                          ensemble_results::Vector{<:Union{EnsembleFVSolution,
                                                                           EnsembleMoMSolution}},
                                          optimal_solutions; title = "", savename = "",
                                          colors = nothing, colouroffset = 0, parameters = nothing,
                                          nucleationfunction = nothing,
                                          growthfunction = nothing,
                                          aggregationfunction = nothing,
                                          breakagefunction = nothing,
                                          solver = nothing,
                                          parameter_samples = nothing,
                                          showtext = true, showplot::Bool = true,
                                          savedir::Union{Nothing, AbstractString} = nothing,
                                          figure_kwargs = (;), axis_kwargs = (;))

    fontfile = joinpath(@__DIR__, "..", "PaperMono-Regular.ttf")
    Makie.with_theme(fonts = (; regular = fontfile, bold = fontfile, italic = fontfile,
                              bold_italic = fontfile)) do

        figure_defaults = (; size = (850, 700))
        figure = Makie.Figure(; merge(figure_defaults, figure_kwargs)...)

        ms_legendfontsize = 18
        ms_xtickfontsize = 18
        ms_ytickfontsize = 18
        ms_ylabelfontsize = 20
        ms_xlabelfontsize = 20
        ms_titlesize = 20

        ms_markersize = 24
        ms_linewidth = 5
        ms_whiskerwidth = 10
        ms_linewidtheb = 2

        color_palette = Plots.palette(:Set2_8)
        color_palette = repeat(collect(color_palette), 10)

        axis_defaults = (; xlabel = "Batch Time [min]",
                           ylabel = "Particle size [μm]",
                           titlesize = ms_titlesize,
                           xticklabelsize = ms_xtickfontsize,
                           yticklabelsize = ms_ytickfontsize,
                           ylabelsize = ms_ylabelfontsize,
                           xlabelsize = ms_xlabelfontsize,
                           yminorgridvisible = true,
                           xminorgridvisible = true,
                           yminorticks = Makie.IntervalsBetween(4),
                           xminorticks = Makie.IntervalsBetween(4),
                           limits = ((0,
                                      maximum([measurements[m].time[end]
                                               for m in eachindex(measurements)]) + 20),
                                     nothing))
        ax1 = Makie.Axis(figure[1, 1]; merge(axis_defaults, axis_kwargs)...)

        # Select the particle-size trajectories and summary stats
        get_size_fields(sol) = sol isa EnsembleMoMSolution ?
                               (sol.d43, sol.d43_mean, sol.d43_std, "D43") :
                               (sol.d50q, sol.d50q_mean, sol.d50q_std, "D50")

        delta = 0.499
        alpha = 0.3

        for m in eachindex(measurements)
            size_matrix, _, _, _ = get_size_fields(ensemble_results[m])
            particles = [Particles(Vector{Float64}(col))
                         for col in eachcol(size_matrix)]
            Makie.band!(ax1, ensemble_results[m].time,
                        pquantile.(particles, 0.5 - delta),
                        pquantile.(particles, 0.5 + delta),
                        color = (resolve_experiment_color(colors, m, color_palette, colouroffset), alpha))
        end

        for m in eachindex(measurements)
            sol = optimal_solutions[m][2]
            size_traj = sol isa CrystallisationMoMSolution ? sol.d43 : sol.d50q
            Makie.lines!(ax1, sol.time, size_traj, color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                         linewidth = ms_linewidth,
                         linestyle = :dash)
        end

        plot_elements = []
        labels = []

        # Build parameter string if functions and parameters are provided
        param_string = ""
        if !isnothing(parameters) && !isnothing(nucleationfunction) &&
           !isnothing(growthfunction) &&
           !isnothing(aggregationfunction) && !isnothing(breakagefunction)

            param_groups = [
                (nucleationfunction.symbols, "Nucleation"),
                (growthfunction.symbols, "Growth"),
                (aggregationfunction.symbols, "Aggregation"),
                (breakagefunction.symbols, "Breakage")
            ]

            idx = 1
            param_string = "Parameters:\n"
            for (symbols, groupname) in param_groups
                if !isempty(symbols)
                    param_string *= "$groupname: "
                    group_params = []
                    for s in symbols
                        if idx <= length(parameters)
                            if !isnothing(parameter_samples) &&
                               size(parameter_samples, 1) >= idx
                                param_mean = mean(parameter_samples[idx, :])
                                param_std = std(parameter_samples[idx, :])
                                push!(group_params,
                                      "$(s) = $(round(param_mean, sigdigits=2)) ± $(round(param_std, sigdigits=2))")
                            else
                                push!(group_params,
                                      "$(s) = $(round(parameters[idx], sigdigits=2))")
                            end
                            idx += 1
                        end
                    end
                    param_string *= join(group_params, ", ") * "\n"
                end
            end

            if !isnothing(solver)
                param_string *= "\nSolver: $(string(typeof(solver)))\n"
            end
        end

        # Collect particle size information
        size_info = String[]
        measured_sizes = Float64[]

        for m in eachindex(measurements)
            ensemble_sol = ensemble_results[m]
            size_matrix, size_mean, size_std, label_symbol = get_size_fields(ensemble_sol)

            # Measurement values (prefer d43; fallback to quantile)
            if hasproperty(measurements[m], :d43) && !isnothing(measurements[m].d43)
                meas_size = measurements[m].d43
                meas_std = hasproperty(measurements[m], :d43var) ?
                           sqrt.(measurements[m].d43var) : nothing
            elseif hasproperty(measurements[m], :quantilemean) &&
                   !isnothing(measurements[m].quantilemean)
                meas_size = measurements[m].quantilemean
                meas_std = hasproperty(measurements[m], :quantilevariance) ?
                           sqrt.(measurements[m].quantilevariance) : nothing
            else
                meas_size = nothing
                meas_std = nothing
            end

            if !isnothing(meas_size)
                push!(measured_sizes, meas_size)
                if !isnothing(meas_std)
                    Makie.errorbars!(ax1, [measurements[m].time[end]], [meas_size],
                                     [meas_std],
                                     color = :black,
                                     whiskerwidth = ms_whiskerwidth,
                                     linewidth = ms_linewidtheb)
                end

                p = Makie.scatter!(ax1, measurements[m].time[end], meas_size,
                                   color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                                   markersize = ms_markersize,
                                   strokewidth = 2)
                push!(plot_elements, p)
                # Get the actual experiment ID if available, otherwise use the loop index
                exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m
                push!(labels, "Exp. $(exp_id)")
            end

            # Collect size information strings
            predicted_size = size_mean[end]
            predicted_std = size_std[end]

            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m

            predicted_size_str = "T = $(round(measurements[m].temperature - 273, digits = 2))°C, Load = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. $(label_symbol) = $(round(predicted_size, sigdigits=3)) ± $(round(predicted_std, sigdigits=2)) μm"

            if !isnothing(meas_size)
                measured_size_str = "Meas. = $(round(meas_size, sigdigits=3))"
                if !isnothing(meas_std)
                    measured_size_str *= " ± $(round(meas_std, sigdigits=2))"
                end
                measured_size_str *= " μm"
                push!(size_info, "Exp. $(exp_id) $predicted_size_str, $measured_size_str")
            else
                push!(size_info, "Exp. $(exp_id) $predicted_size_str")
            end
        end

        # Add particle size information to parameter string
        if !isempty(size_info)
            if !isempty(param_string)
                param_string *= "\nParticle Sizes:\n"
            else
                param_string = "Particle Sizes:\n"
            end
            param_string *= join(size_info, "\n")
        end

        # Add text box with all information if desired
        if !isempty(param_string) && showtext
            max_time = maximum([m.time[end] for m in measurements])
            max_size = isempty(measured_sizes) ?
                       maximum([maximum(get_size_fields(ensemble_results[m])[2])
                                for m in eachindex(ensemble_results)]) :
                       maximum(measured_sizes)

            Makie.text!(ax1, max_time * 0.75, max_size * 1.05, text = param_string,
                        align = (:center, :top),
                        fontsize = 12)
        end

        Makie.Legend(figure[2, 1], plot_elements, labels, orientation = :horizontal,
                     tellwidth = false, tellheight = true)

        Makie.Label(figure[0, 1], (title), tellwidth = false)

        if savename != ""
            output_dir = _resolve_plot_savedir(savedir, "Saved Plots")
            mkpath(output_dir)
            save(joinpath(output_dir, "$(savename).png"), figure, px_per_unit = 3)
        end

        if showplot
            display(figure)
        end

        return figure
    end
end

"""
    plot_measurements_vs_simulation(measurements, parameters, nucleationfunction,
                                    growthfunction, aggregationfunction, breakagefunction,
                                    solver; kwargs...)

Plot experimental measurements against a single deterministic simulation using the
given kinetic parameters. Runs [`runsimulation`](@ref) internally for each measurement
and overlays the simulated trajectories on the experimental data.

# Arguments
- `measurements::Vector{<:AbstractMeasurements}`: experimental measurement sets.
- `parameters::AbstractVector{<:Real}`: kinetic parameter vector.
- `nucleationfunction::AbstractNucleationFunction`: nucleation kinetic model.
- `growthfunction::AbstractGrowthFunction`: growth kinetic model.
- `aggregationfunction::AbstractAggregationFunction`: aggregation kinetic model.
- `breakagefunction::AbstractBreakageFunction`: breakage kinetic model.
- `solver::AbstractSolver`: population balance solver ([`MoM`](@ref), [`FiniteVol`](@ref),
  or [`WENO`](@ref)).

# Keyword Arguments
- `title::String`: figure title (default `""`).
- `savename::String`: file path (without extension) to save the figure; empty to skip.
- `colors`: per-experiment colors (`nothing` for auto palette, `Vector`, or `Dict`).
- `colouroffset::Int`: offset into the default colour palette (default `0`).
- `showplot::Bool`: call `display` on the figure (default `true`).
- `showtext::Bool`: show inline text annotations (default `true`).
- `show_thesistext::Bool`: show the thesis-style summary table below the plot
  (default `false`).
- `show_title::Bool`: display the figure title (default `true`).
- `show_parameter_table::Bool`: display a parameter summary table (default `true`).
- `show_measurement_uncertainty::Bool`: show measurement error bars (default `true`).
- `figure_kwargs::NamedTuple`: extra keyword arguments forwarded to `Makie.Figure`.
- `axis_kwargs::NamedTuple`: extra keyword arguments forwarded to `Makie.Axis`.
- `kwargs...`: additional keyword arguments forwarded to [`runsimulation`](@ref).

# Returns
A `Makie.Figure` object. If `savename` is non-empty the figure is also saved as PNG.
"""
function plot_measurements_vs_simulation(measurements::Vector{<:AbstractMeasurements},
                                         parameters::AbstractVector{<:Real},
                                         nucleationfunction::AbstractNucleationFunction,
                                         growthfunction::AbstractGrowthFunction,
                                         aggregationfunction::AbstractAggregationFunction,
                                         breakagefunction::AbstractBreakageFunction,
                                         solver::AbstractSolver;
                                         title = "", savename = "", colors = nothing, colouroffset = 0,
                                         showplot::Bool = true, showtext = true,
                                         show_thesistext = false, show_title::Bool = true,
                                         show_parameter_table::Bool = true,
                                         show_measurement_uncertainty::Bool = true,
                                         savedir::Union{Nothing, AbstractString} = nothing,
                                         figure_kwargs = (;), axis_kwargs = (;),
                                         kwargs...)

    # Pre-run all simulations so we can build thesis rows before the plot
    solutions = Vector{Any}(undef, length(measurements))
    for m in eachindex(measurements)
        prob, sol = runsimulation(parameters,
                                  nucl = nucleationfunction,
                                  gr = growthfunction,
                                  agg = aggregationfunction,
                                  br = breakagefunction,
                                  initial_concentration = measurements[m].concentrationmean[1],
                                  save_idx = LinRange(measurements[m].time[1],
                                                      measurements[m].time[end], 150),
                                  solver = solver,
                                  temp_profile = ConstantTemperature(measurements[m].temperature),
                                  loading = measurements[m].loading,
                                  kwargs...)
        solutions[m] = sol
    end

    # Build thesis table rows (no ± on predictions — single parameter set, not ensemble)
    thesis_rows = Vector{Vector{String}}()
    size_label = "size"
    if show_thesistext
        thesis_data = build_simulation_thesis_table_data(measurements,
                                                         solutions;
                                                         show_measurement_uncertainty = show_measurement_uncertainty)
        thesis_rows = thesis_data.rows
        size_label = thesis_data.size_label
    end

    fontfile = joinpath(@__DIR__, "..", "PaperMono-Regular.ttf")

    Makie.with_theme(fonts = (; regular = fontfile, bold = fontfile,
                              italic = fontfile, bold_italic = fontfile)) do

        # Calculate dynamic figure height based on content
        axis_height = 350
        title_height = 80
        legend_height = 50
        table_row_height = 24
        table_header_height = 80

        thesis_table_height = show_thesistext && !isempty(thesis_rows) ?
            table_header_height + length(measurements) * table_row_height : 0

        param_table_height = 0
        if show_thesistext && show_parameter_table
            n_params = length(vcat(nucleationfunction.symbols, growthfunction.symbols))
            if n_params > 0
                param_table_height = table_header_height + n_params * table_row_height
            end
        end

        total_height = title_height + axis_height + legend_height + thesis_table_height +
                       param_table_height + 40

        figure_defaults = show_thesistext ? (; size = (700, total_height)) : (; size = (850, 700))
        figure = Makie.Figure(; merge(figure_defaults, figure_kwargs)...)

        ms_legendfontsize = 18
        ms_xtickfontsize = 18
        ms_ytickfontsize = 18
        ms_ylabelfontsize = 20
        ms_xlabelfontsize = 20
        ms_titlesize = 20

        ms_markersize = 24
        ms_linewidth = 5
        ms_whiskerwidth = 10
        ms_linewidtheb = 2

        color_palette = Plots.palette(:Set2_8)
        color_palette = repeat(collect(color_palette), 10)

        axis_defaults = (; xlabel = "Batch Time [min]",
                           ylabel = "Lysozyme concentration [mg/mL]",
                           titlesize = ms_titlesize,
                           xticklabelsize = ms_xtickfontsize,
                           yticklabelsize = ms_ytickfontsize,
                           ylabelsize = ms_ylabelfontsize,
                           xlabelsize = ms_xlabelfontsize,
                           yminorgridvisible = true,
                           xminorgridvisible = true,
                           yminorticks = Makie.IntervalsBetween(4),
                           xminorticks = Makie.IntervalsBetween(4),
                           limits = ((0,
                                      maximum([measurements[m].time[end]
                                               for m in eachindex(measurements)]) + 20),
                                     nothing))
        ax1 = Makie.Axis(figure[1, 1]; merge(axis_defaults, axis_kwargs)...)

        plot_elements = []
        labels = []

        # Build parameter string (grouped by function, separated by newlines)
        param_groups = [
            (nucleationfunction.symbols, "Nucleation"),
            (growthfunction.symbols, "Growth"),
            (aggregationfunction.symbols, "Aggregation"),
            (breakagefunction.symbols, "Breakage")
        ]

        idx = 1
        param_string = "Parameters:\n"
        for (symbols, groupname) in param_groups
            if !isempty(symbols)
                param_string *= "$groupname: "
                group_params = []
                for s in symbols
                    if idx <= length(parameters)
                        push!(group_params, "$(s) = $(round(parameters[idx], sigdigits=2))")
                        idx += 1
                    end
                end
                param_string *= join(group_params, ", ") * "\n"
            end
        end

        # Add solver information
        param_string *= "\nSolver: $(string(typeof(solver)))\n"

        # Collect particle size information
        size_info = String[]

        for m in eachindex(measurements)
            sol = solutions[m]

            Makie.lines!(ax1, sol.time, sol.concentration,
                         color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                         linewidth = ms_linewidth,
                         linestyle = :dash)

            Makie.errorbars!(ax1, measurements[m].time, measurements[m].concentrationmean,
                             sqrt.(measurements[m].concentrationvariance),
                             color = :black,
                             whiskerwidth = ms_whiskerwidth,
                             linewidth = ms_linewidtheb)

            p = Makie.scatter!(ax1, measurements[m].time, measurements[m].concentrationmean,
                               color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                               markersize = ms_markersize,
                               strokewidth = 2)
            push!(plot_elements, p)

            predicted_size = get_characteristic_size(sol)

            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m

            # Collect size information for text box
            if sol isa CrystallisationMoMSolution
                pred_str = "Exp. $(exp_id) T = $(round(measurements[m].temperature-273,digits = 2) )°C, L = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. d43 = $(round(predicted_size, sigdigits=2)) μm"
                if hasproperty(measurements[m], :d43) && !isempty(measurements[m].d43)
                    meas_str = "Meas. = $(round(measurements[m].d43, sigdigits=3)) ± $(round(sqrt(measurements[m].d43var), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                    push!(size_info, combined_str)
                else
                    push!(size_info, pred_str)
                end
            elseif sol isa CrystallisationFVSolution
                pred_str = "Exp. $(exp_id) T = $(round(measurements[m].temperature-273,digits = 2) )°C, L = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. D50 = $(round(predicted_size, sigdigits=2)) μm"
                if hasproperty(measurements[m], :quantilemean) &&
                   !isempty(measurements[m].quantilemean)
                    meas_str = "Meas. = $(round(measurements[m].quantilemean, sigdigits=3)) ± $(round(sqrt(measurements[m].quantilevariance), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                    push!(size_info, combined_str)
                else
                    push!(size_info, pred_str)
                end
            end
            new_label = "Exp. $(exp_id)"
            push!(labels, new_label)
        end

        # Add particle size information to parameter string
        if showtext && !show_thesistext && !isempty(size_info)
            param_string *= "\nParticle Sizes:\n"
            param_string *= join(size_info, "\n")

            max_time = maximum([m.time[end] for m in measurements])
            max_conc = maximum([m.concentrationmean[1] for m in measurements])

            Makie.text!(ax1, max_time * 0.75, max_conc, text = param_string,
                        align = (:center, :top),
                        fontsize = 14)
        end

        nbanks = length(labels) > 6 ? 2 : 1

        Makie.Legend(figure[2, 1], plot_elements, labels, orientation = :horizontal,
                     tellwidth = false, tellheight = true, nbanks = nbanks)

        if show_thesistext && !isempty(thesis_rows)
            thesis_io = IOBuffer()
            thesis_data = permutedims(reduce(hcat, thesis_rows))
            thesis_header = (["Exp ID", "Loading", "T", "Pred. $(size_label)", "Meas. $(size_label)"],
                             ["", "[g/L]", "[°C]", "[μm]", "[μm]"])
            PrettyTables.pretty_table(thesis_io, thesis_data;
                                      header = thesis_header,
                                      tf = PrettyTables.tf_ascii_rounded,
                                      alignment = :c,
                                      linebreaks = true)
            thesis_string = chomp(String(take!(thesis_io)))

            Makie.Label(figure[3, 1], thesis_string;
                        tellwidth = false,
                        tellheight = true,
                        halign = :center,
                        valign = :top,
                        fontsize = 16)
        end

        if show_thesistext && show_parameter_table
            # Build parameter table from the single parameter vector
            param_symbols = vcat(nucleationfunction.symbols, growthfunction.symbols)
            if !isempty(param_symbols)
                params_string = build_parameter_value_table(parameters, param_symbols)
                Makie.Label(figure[4, 1], params_string;
                            tellwidth = false,
                            tellheight = true,
                            halign = :center,
                            valign = :top,
                            fontsize = 16)
            end
        end

        if show_thesistext
            Makie.rowsize!(figure.layout, 1, Makie.Fixed(axis_height))
            Makie.rowsize!(figure.layout, 2, Makie.Auto())
            if !isempty(thesis_rows)
                Makie.rowsize!(figure.layout, 3, Makie.Auto())
            end
            if show_parameter_table &&
               !isempty(vcat(nucleationfunction.symbols, growthfunction.symbols))
                Makie.rowsize!(figure.layout, 4, Makie.Auto())
            end
            Makie.rowgap!(figure.layout, 10)
        end

        if show_title
            Makie.Label(figure[0, 1], (title), tellwidth = false, tellheight = true)
        end

        if savename != ""
            output_dir = _resolve_plot_savedir(savedir, "Saved Plots")
            mkpath(output_dir)
            save(joinpath(output_dir, "$(savename).png"), figure, px_per_unit = 3)
        end

        if showplot
            display(figure)
        end

        return figure
    end
end

function plot_ps_measurements_vs_simulation(measurements::Vector{<:AbstractMeasurements},
                                            parameters::AbstractVector{<:Real},
                                            nucleationfunction::AbstractNucleationFunction,
                                            growthfunction::AbstractGrowthFunction,
                                            aggregationfunction::AbstractAggregationFunction,
                                            breakagefunction::AbstractBreakageFunction,
                                            solver::AbstractSolver;
                                            title = "", savename = "", colors = nothing, colouroffset = 0,
                                            showplot::Bool = true,
                                            savedir::Union{Nothing, AbstractString} = nothing,
                                            kwargs...)

    fontfile = joinpath(@__DIR__, "..", "PaperMono-Regular.ttf")

    Makie.with_theme(Makie.Theme(fonts = (; regular = fontfile, bold = fontfile,
                                          italic = fontfile, bold_italic = fontfile))) do

        figure = Makie.Figure(size = (850, 700))


        ms_legendfontsize = 18
        ms_xtickfontsize = 18
        ms_ytickfontsize = 18
        ms_ylabelfontsize = 20
        ms_xlabelfontsize = 20
        ms_titlesize = 20

        ms_markersize = 24
        ms_linewidth = 5
        ms_whiskerwidth = 10
        ms_linewidtheb = 2

        color_palette = Plots.palette(:Set2_8)
        color_palette = repeat(collect(color_palette), 10)

        ax1 = Makie.Axis(figure[1, 1], xlabel = "Batch Time [min]",
                         ylabel = "Particle size [μm]",
                         titlesize = ms_titlesize, title = title,
                         xticklabelsize = ms_xtickfontsize,
                         yticklabelsize = ms_ytickfontsize,
                         ylabelsize = ms_ylabelfontsize, xlabelsize = ms_xlabelfontsize,
                         yminorgridvisible = true, xminorgridvisible = true,
                         yminorticks = Makie.IntervalsBetween(4),
                         xminorticks = Makie.IntervalsBetween(4))

        plot_elements = []
        labels = []

        # Build parameter string (grouped by function, separated by newlines)
        param_groups = [
            (nucleationfunction.symbols, "Nucleation"),
            (growthfunction.symbols, "Growth"),
            (aggregationfunction.symbols, "Aggregation"),
            (breakagefunction.symbols, "Breakage")
        ]

        idx = 1
        param_string = "Parameters:\n"
        for (symbols, groupname) in param_groups
            if !isempty(symbols)
                param_string *= "$groupname: "
                group_params = []
                for s in symbols
                    if idx <= length(parameters)
                        push!(group_params, "$(s) = $(round(parameters[idx], sigdigits=2))")
                        idx += 1
                    end
                end
                param_string *= join(group_params, ", ") * "\n"
            end
        end

        # Add solver information
        param_string *= "\nSolver: $(string(typeof(solver)))\n"

        # Collect particle size information
        size_info = String[]

        for m in eachindex(measurements)
            prob,
            sol = runsimulation(parameters,
                                nucl = nucleationfunction,
                                gr = growthfunction,
                                agg = aggregationfunction,
                                br = breakagefunction,
                                initial_concentration = measurements[m].concentrationmean[1],
                                save_idx = LinRange(measurements[m].time[1],
                                                    measurements[m].time[end], 150),
                                solver = solver,
                                temp_profile = ConstantTemperature(measurements[m].temperature),
                                loading = measurements[m].loading,
                                kwargs...)

            function get_characteristic_size(sol)
                if sol isa CrystallisationMoMSolution
                    return sol.d43
                elseif sol isa CrystallisationFVSolution
                    return sol.d50q
                else
                    return NaN
                end
            end

            particlessizes = get_characteristic_size(sol)

            Makie.lines!(ax1, sol.time, particlessizes,
                         color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                         linewidth = ms_linewidth,
                         linestyle = :dash)

            # Makie.errorbars!(ax1, measurements[m].time[end], measurements[m].d43,
            #                  sqrt.(measurements[m].d43var),
            #                  color = :black,
            #                  whiskerwidth = ms_whiskerwidth,
            #                  linewidth = ms_linewidtheb)

            p = Makie.scatter!(ax1, measurements[m].time[end], measurements[m].d43,
                               color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                               markersize = ms_markersize,
                               strokewidth = 2)

            push!(plot_elements, p)

            predicted_size = get_characteristic_size(sol)[end]

            temp_str = "T = $(round(measurements[m].temperature-273.15, digits=2)) °C"

            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m

            # Collect size information for text box
            if sol isa CrystallisationMoMSolution
                pred_str = "Exp. $(exp_id) T = $(round(measurements[m].temperature-273,digits = 2) )°C, L = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. d43 = $(round(predicted_size, sigdigits=2)) μm"
                if hasproperty(measurements[m], :d43) && !isempty(measurements[m].d43)
                    meas_str = "Meas. = $(round(measurements[m].d43, sigdigits=3)) ± $(round(sqrt(measurements[m].d43var), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                    push!(size_info, combined_str)
                else
                    push!(size_info, pred_str)
                end
            elseif sol isa CrystallisationFVSolution
                pred_str = "Exp. $(exp_id) T = $(round(measurements[m].temperature-273,digits = 2) )°C, L = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. D50 = $(round(predicted_size, sigdigits=2)) μm"
                if hasproperty(measurements[m], :quantilemean) &&
                   !isempty(measurements[m].quantilemean)
                    meas_str = "Meas. = $(round(measurements[m].quantilemean, sigdigits=3)) ± $(round(sqrt(measurements[m].quantilevariance), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                    push!(size_info, combined_str)
                else
                    push!(size_info, pred_str)
                end
            end
            new_label = "Exp. $(exp_id)"
            push!(labels, new_label)
        end

        # Add particle size information to parameter string
        if !isempty(size_info)
            param_string *= "\nParticle Sizes:\n"
            param_string *= join(size_info, "\n")
        end

        # Add text box with all information

        max_time = maximum([m.time[end] for m in measurements])
        max_conc = maximum([m.concentrationmean[1] for m in measurements])

        Makie.text!(ax1, max_time * 0.75, max_conc, text = param_string,
                    align = (:center, :top),
                    fontsize = 14)

        Makie.Legend(figure[2, 1], plot_elements, labels, orientation = :horizontal,
                     tellwidth = false, tellheight = true)

        # Makie.Label(figure[0, 1], (title), tellwidth = false)

        if savename != ""
            output_dir = _resolve_plot_savedir(savedir, "Saved Plots")
            mkpath(output_dir)
            save(joinpath(output_dir, "$(savename).png"), figure, px_per_unit = 3)
        end

        if showplot
            display(figure)
        end

        return figure
    end
end
