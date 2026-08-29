function plot_measurements_vs_simulation(measurements::Vector{<:AbstractExperiment},
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
                                  initial_concentration = initial_concentration(measurements[m]),
                                  save_idx = LinRange(measurements[m].observables.concentration.time[1],
                                                      measurements[m].observables.concentration.time[end], 150),
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

    fontfile = joinpath(@__DIR__, "..", "..", "PaperMono-Regular.ttf")

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
                                      maximum([measurements[m].observables.concentration.time[end]
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

            Makie.errorbars!(ax1, measurements[m].observables.concentration.time, measurements[m].observables.concentration.mean,
                             sqrt.(measurements[m].observables.concentration.variance),
                             color = :black,
                             whiskerwidth = ms_whiskerwidth,
                             linewidth = ms_linewidtheb)

            p = Makie.scatter!(ax1, measurements[m].observables.concentration.time, measurements[m].observables.concentration.mean,
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
                                    meas_str = "Meas. = $(round(measurements[m].observables.d43.mean, sigdigits=3)) ± $(round(sqrt(measurements[m].observables.d43.variance), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                push!(size_info, combined_str)
            elseif sol isa CrystallisationFVSolution
                pred_str = "Exp. $(exp_id) T = $(round(measurements[m].temperature-273,digits = 2) )°C, L = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. D50 = $(round(predicted_size, sigdigits=2)) μm"
                                    meas_str = "Meas. = $(round(measurements[m].observables.d50q.mean, sigdigits=3)) ± $(round(sqrt(measurements[m].observables.d50q.variance), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                    push!(size_info, combined_str)
            end
            new_label = "Exp. $(exp_id)"
            push!(labels, new_label)
        end

        # Add particle size information to parameter string
        if showtext && !show_thesistext && !isempty(size_info)
            param_string *= "\nParticle Sizes:\n"
            param_string *= join(size_info, "\n")

            max_time = maximum([m.observables.concentration.time[end] for m in measurements])
            max_conc = maximum([initial_concentration(m) for m in measurements])

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
            _save_plot(figure, savename, savedir)
        end

        if showplot
            display(figure)
        end

        return figure
    end
end

function plot_ps_measurements_vs_simulation(measurements::Vector{<:AbstractExperiment},
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

    fontfile = joinpath(@__DIR__, "..", "..", "PaperMono-Regular.ttf")

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

        color_palette = repeat(_palette(:Set2_8), 10)

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
                                initial_concentration = initial_concentration(measurements[m]),
                                save_idx = LinRange(measurements[m].observables.concentration.time[1],
                                                    measurements[m].observables.concentration.time[end], 150),
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

            # Makie.errorbars!(ax1, measurements[m].observables.concentration.time[end], measurements[m].observables.d43.mean,
            #                  sqrt.(measurements[m].observables.d43.variance),
            #                  color = :black,
            #                  whiskerwidth = ms_whiskerwidth,
            #                  linewidth = ms_linewidtheb)

            p = Makie.scatter!(ax1, measurements[m].observables.concentration.time[end], measurements[m].observables.d43.mean,
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
                                    meas_str = "Meas. = $(round(measurements[m].observables.d43.mean, sigdigits=3)) ± $(round(sqrt(measurements[m].observables.d43.variance), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                push!(size_info, combined_str)
            elseif sol isa CrystallisationFVSolution
                pred_str = "Exp. $(exp_id) T = $(round(measurements[m].temperature-273,digits = 2) )°C, L = $(round(measurements[m].loading, sigdigits=2)) g/L, Pred. D50 = $(round(predicted_size, sigdigits=2)) μm"
                                    meas_str = "Meas. = $(round(measurements[m].observables.d50q.mean, sigdigits=3)) ± $(round(sqrt(measurements[m].observables.d50q.variance), sigdigits=2)) μm"
                    combined_str = "$pred_str, $meas_str"
                    push!(size_info, combined_str)
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

        max_time = maximum([m.observables.concentration.time[end] for m in measurements])
        max_conc = maximum([initial_concentration(m) for m in measurements])

        Makie.text!(ax1, max_time * 0.75, max_conc, text = param_string,
                    align = (:center, :top),
                    fontsize = 14)

        Makie.Legend(figure[2, 1], plot_elements, labels, orientation = :horizontal,
                     tellwidth = false, tellheight = true)

        # Makie.Label(figure[0, 1], (title), tellwidth = false)

        if savename != ""
            _save_plot(figure, savename, savedir)
        end

        if showplot
            display(figure)
        end

        return figure
    end
end
