"""
    plot_measurements_vs_ensemble(measurements, ensemble_results,
                                  optimal_solutions; kwargs...) -> Makie.Figure

Compare experimental concentration and particle-size measurements with
ensemble simulation results and nominal solutions.
"""
function plot_measurements_vs_ensemble(measurements::Vector{<:AbstractExperiment},
                                       ensemble_results::Vector{<:Union{EnsembleFVSolution,
                                                                        EnsembleMoMSolution}},
                                       optimal_solutions; title = "", savename = "",
                                       colors = nothing, colouroffset = 0, parameters = nothing,
                                       nucleationfunction = nothing,
                                       growthfunction = nothing,
                                       dissolutionfunction = nothing,
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

            size_name, measured_size = _measured_size_observable(
                measurements[m], optimal_solutions[m][2])
            if size_name === nothing
                size_name = ensemble_sol isa EnsembleFVSolution ? :d50q : :d43
            end

            size_matrix, size_mean, size_std, pred_label =
                _ensemble_size_fields(ensemble_sol, Val(size_name))
            pred_text = "$(round(1e6 * size_mean[end], sigdigits=3)) ± " *
                        "$(round(1e6 * size_std[end], sigdigits=2))"
            if measured_size !== nothing
                meas_mean = 1e6 * measured_size.mean[end]
                meas_std = measured_size.variance === nothing ? nothing :
                            1e6 * sqrt(measured_size.variance[end])
                meas_text = meas_std === nothing ? "$(round(meas_mean, sigdigits=3))" :
                            "$(round(meas_mean, sigdigits=3)) ± $(round(meas_std, sigdigits=2))"
            end

            temp_str = string(round(measurements[m].temperature - 273, digits = 2))
            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m
            push!(thesis_rows, [string(exp_id), temp_str, pred_text, meas_text])
        end
    end

    fontfile = joinpath(@__DIR__, "..", "..", "PaperMono-Regular.ttf")
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
            dissolution_symbols = dissolutionfunction === nothing ? Symbol[] :
                                  dissolutionfunction.symbols
            n_params = length(vcat(nucleationfunction.symbols, growthfunction.symbols,
                                   dissolution_symbols))
            param_table_height = table_header_height + n_params * table_row_height
        end

        total_height = title_height + axis_height + legend_height + thesis_table_height +
                       param_table_height + 40  # 40 for gaps

        figure_defaults = (; size = (700, total_height))
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

        color_palette = repeat(_palette(:Set2_8), 10)

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
                                      maximum([_experiment_time_span(measurements[m])[2] / 60
                                               for m in eachindex(measurements)]) + 20),
                                     nothing))
        ax1 = Makie.Axis(figure[1, 1]; merge(axis_defaults, axis_kwargs)...)

        delta = 0.499
        alpha = 0.3

        for m in eachindex(measurements)
            particles = [Particles(Vector{Float64}(col))
                         for col in eachcol(ensemble_results[m].concentration)]
            Makie.band!(ax1, _plot_time_minutes(ensemble_results[m].time),
                        pquantile.(particles, 0.5 - delta),
                        pquantile.(particles, 0.5 + delta),
                        color = (resolve_experiment_color(colors, m, color_palette, colouroffset), alpha))
        end

        for m in eachindex(measurements)
            Makie.lines!(ax1, _plot_time_minutes(optimal_solutions[m][2].time),
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
            concentration = measurements[m].observables.concentration
            if concentration.variance !== nothing
                Makie.errorbars!(ax1, _plot_time_minutes(concentration.time), concentration.mean,
                                 sqrt.(concentration.variance),
                                 color = :black,
                                 whiskerwidth = ms_whiskerwidth,
                                 linewidth = ms_linewidtheb)
            end

            p = Makie.scatter!(ax1, _plot_time_minutes(concentration.time), concentration.mean,
                               color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                               markersize = ms_markersize,
                               strokewidth = 2)
            push!(plot_elements, p)

            ensemble_sol = ensemble_results[m]

            measured_size_str = ""
            predicted_size_str = ""

            if ensemble_sol isa EnsembleMoMSolution
                size_name = :d43
            elseif ensemble_sol isa EnsembleFVSolution
                size_name = hasproperty(measurements[m].observables, :d50q) ? :d50q : :d43
            end
            size_matrix, size_mean, size_std, label_symbol =
                _ensemble_size_fields(ensemble_sol, Val(size_name))
            predicted_size_str = "T = $(round(measurements[m].temperature-273,digits = 2) )°C, " *
                                "Pred. $(label_symbol) = $(round(1e6 * size_mean[end], sigdigits=3)) ± " *
                                "$(round(1e6 * size_std[end], sigdigits=2)) μm"

            _, measured_size = _measured_size_observable(
                measurements[m], optimal_solutions[m][2])
            if measured_size !== nothing
                measured_size_str = measured_size.variance === nothing ?
                                    "Meas. = $(round(1e6 * measured_size.mean[end], sigdigits=3)) μm" :
                                    "Meas. = $(round(1e6 * measured_size.mean[end], sigdigits=3)) ± " *
                                    "$(round(1e6 * sqrt(measured_size.variance[end]), sigdigits=2)) μm"
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
            max_time = maximum([_experiment_time_span(m)[2] / 60 for m in measurements])
            max_conc = maximum([initial_concentration(m) for m in measurements])

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
            thesis_column_labels = [
                ["Exp ID", "T", "Pred. d43", "Meas. d43"],
                ["", "[°C]", "[μm]", "[μm]"],
            ]
            PrettyTables.pretty_table(thesis_io, thesis_data;
                                      column_labels = thesis_column_labels,
                                      table_format = PrettyTables.TextTableFormat(
                                          borders = PrettyTables.text_table_borders__ascii_rounded),
                                      alignment = :c,
                                      line_breaks = true)
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
            dissolution_symbols = dissolutionfunction === nothing ? Symbol[] :
                                  dissolutionfunction.symbols
            param_symbols = vcat(nucleationfunction.symbols, growthfunction.symbols,
                                 dissolution_symbols)
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
            _save_plot(figure, savename, savedir)
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
- `measurements::Vector{<:AbstractExperiment}`: experimental datasets, one per
  batch.
- `ensemble_results::Vector{<:Union{EnsembleFVSolution, EnsembleMoMSolution}}`:
  ensemble simulation outputs containing particle-size trajectories. QMOM
  supplies its d43 trajectory through the moment-based representation.
- `optimal_solutions`: vector of `(problem, solution)` tuples for the nominal
  parameter set.
- `showtext`: toggle annotation textbox.

# Returns
A `Makie.Figure` showing particle-size ensembles, optimal simulations, and
measurements. Saves to `Saved Plots` when `savename` is provided.
"""
function plot_ps_measurements_vs_ensemble(measurements::Vector{<:AbstractExperiment},
                                          ensemble_results::Vector{<:Union{EnsembleFVSolution,
                                                                           EnsembleMoMSolution}},
                                          optimal_solutions; title = "", savename = "",
                                          colors = nothing, colouroffset = 0, parameters = nothing,
                                          nucleationfunction = nothing,
                                          growthfunction = nothing,
                                          dissolutionfunction = nothing,
                                          aggregationfunction = nothing,
                                          breakagefunction = nothing,
                                          solver = nothing,
                                          parameter_samples = nothing,
                                          showtext = true, showplot::Bool = true,
                                          savedir::Union{Nothing, AbstractString} = nothing,
                                          figure_kwargs = (;), axis_kwargs = (;))

    fontfile = joinpath(@__DIR__, "..", "..", "PaperMono-Regular.ttf")
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

        color_palette = repeat(_palette(:Set2_8), 10)

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
                                      maximum([_experiment_time_span(measurements[m])[2] / 60
                                               for m in eachindex(measurements)]) + 20),
                                     nothing))
        ax1 = Makie.Axis(figure[1, 1]; merge(axis_defaults, axis_kwargs)...)

        delta = 0.499
        alpha = 0.3

        for m in eachindex(measurements)
            size_matrix, _, _, _ = _ensemble_size_fields_for_experiment(
                ensemble_results[m], measurements[m], optimal_solutions[m][2])
            particles = [Particles(Vector{Float64}(col))
                         for col in eachcol(size_matrix)]
            Makie.band!(ax1, _plot_time_minutes(ensemble_results[m].time),
                        1e6 .* pquantile.(particles, 0.5 - delta),
                        1e6 .* pquantile.(particles, 0.5 + delta),
                        color = (resolve_experiment_color(colors, m, color_palette, colouroffset), alpha))
        end

        for m in eachindex(measurements)
            sol = optimal_solutions[m][2]
            size_name, measured_size = _measured_size_observable(measurements[m], sol)
            size_traj = size_name === nothing ? _size_trajectory(sol) :
                        _solution_observable_trajectory(sol, size_name)
            Makie.lines!(ax1, _plot_time_minutes(sol.time),
                         _plot_size_micrometres(size_traj),
                         color = resolve_experiment_color(colors, m, color_palette, colouroffset),
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
                dissolutionfunction === nothing ? (Symbol[], "Dissolution") :
                                                  (dissolutionfunction.symbols, "Dissolution"),
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
            _, measured_size = _measured_size_observable(
                measurements[m], optimal_solutions[m][2])
            size_matrix, size_mean, size_std, label_symbol =
                _ensemble_size_fields_for_experiment(ensemble_sol, measurements[m],
                                                     optimal_solutions[m][2])

            # Select the measured metric matching the optimal solution.
            measured_size === nothing && continue
            meas_size = _plot_size_micrometres(measured_size.mean)
            meas_std = measured_size.variance === nothing ? nothing :
                        _plot_size_micrometres(sqrt.(measured_size.variance))

            append!(measured_sizes, meas_size)
            if meas_std !== nothing
                Makie.errorbars!(ax1, _plot_time_minutes(measured_size.time), meas_size,
                                 meas_std,
                                 color = :black,
                                 whiskerwidth = ms_whiskerwidth,
                                 linewidth = ms_linewidtheb)
            end

            p = Makie.scatter!(ax1, _plot_time_minutes(measured_size.time), meas_size,
                               color = resolve_experiment_color(colors, m, color_palette, colouroffset),
                               markersize = ms_markersize,
                               strokewidth = 2)
            push!(plot_elements, p)
            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m
            push!(labels, "Exp. $(exp_id)")

            # Collect size information strings
            predicted_size = 1e6 * size_mean[end]
            predicted_std = 1e6 * size_std[end]

            # Get the actual experiment ID if available, otherwise use the loop index
            exp_id = hasproperty(measurements[m], :exp_id) ? measurements[m].exp_id : m

            predicted_size_str = "T = $(round(measurements[m].temperature - 273, digits = 2))°C, Pred. $(label_symbol) = $(round(predicted_size, sigdigits=3)) ± $(round(predicted_std, sigdigits=2)) μm"

            if !isempty(meas_size)
                final_meas_size = meas_size[end]
                measured_size_str = "Meas. = $(round(final_meas_size, sigdigits=3))"
                if meas_std !== nothing
                    measured_size_str *= " ± $(round(meas_std[end], sigdigits=2))"
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
            max_time = maximum([_experiment_time_span(m)[2] / 60 for m in measurements])
            max_size = isempty(measured_sizes) ?
                       1e6 * maximum([maximum(_ensemble_size_fields_for_experiment(
                                                        ensemble_results[m],
                                                        measurements[m],
                                                        optimal_solutions[m][2])[2])
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
            _save_plot(figure, savename, savedir)
        end

        if showplot
            display(figure)
        end

        return figure
    end
end
