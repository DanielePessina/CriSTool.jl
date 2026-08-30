function plot_measurements_vs_ensemble(measurements::Vector{<:AbstractExperiment},
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

                                    meas_mean = measurements[m].observables.d43.mean
                    meas_std = sqrt(measurements[m].observables.d43.variance)
                    meas_text = "$(round(meas_mean, sigdigits=3)) ± $(round(meas_std, sigdigits=2))"
            elseif hasproperty(ensemble_sol, :d50q_mean)
                pred_label = "D50"
                mean_pred_size = ensemble_sol.d50q_mean[end]
                std_pred_size = ensemble_sol.d50q_std[end]
                pred_text = "$(round(mean_pred_size, sigdigits=3)) ± $(round(std_pred_size, sigdigits=2))"

                                    meas_mean = measurements[m].observables.d50q.mean
                    meas_std = sqrt(measurements[m].observables.d50q.variance)
                    meas_text = "$(round(meas_mean, sigdigits=3)) ± $(round(meas_std, sigdigits=2))"
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

            ensemble_sol = ensemble_results[m]

            measured_size_str = ""
            predicted_size_str = ""

            if hasproperty(ensemble_sol, :d43_mean) # MoM solution
                mean_pred_size = ensemble_sol.d43_mean[end]
                std_pred_size = ensemble_sol.d43_std[end]
                predicted_size_str = "T = $(round(measurements[m].temperature-273,digits = 2) )°C, Pred. D43 = $(round(mean_pred_size, sigdigits=3)) ± $(round(std_pred_size, sigdigits=2)) μm"

                                    measured_size_str = "Meas. = $(round(measurements[m].observables.d43.mean, sigdigits=3)) ± $(round(sqrt(measurements[m].observables.d43.variance), sigdigits=2)) μm"
            elseif hasproperty(ensemble_sol, :d50q_mean) # FV solution
                mean_pred_size = ensemble_sol.d50q_mean[end]
                std_pred_size = ensemble_sol.d50q_std[end]
                predicted_size_str = "T = $(round(measurements[m].temperature-273,digits = 2) )°C, Pred. D50 = $(round(mean_pred_size, sigdigits=3)) ± $(round(std_pred_size, sigdigits=2)) μm"

                                    measured_size_str = "Meas. = $(round(measurements[m].observables.d50q.mean, sigdigits=3)) ± $(round(sqrt(measurements[m].observables.d50q.variance), sigdigits=2)) μm"
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
            max_time = maximum([m.observables.concentration.time[end] for m in measurements])
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
            thesis_header = (["Exp ID", "T", "Pred. d43", "Meas. d43"],
                             ["", "[°C]", "[μm]", "[μm]"])
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
                                      maximum([measurements[m].observables.concentration.time[end]
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
            size_traj = _size_trajectory(sol)
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

            # Measurement values (d43; all loaders also populate d50q with the same value)
            meas_size = measurements[m].observables.d43.mean
            meas_std = sqrt.(measurements[m].observables.d43.variance)

            if !isnothing(meas_size)
                push!(measured_sizes, meas_size)
                if !isnothing(meas_std)
                    Makie.errorbars!(ax1, [measurements[m].observables.concentration.time[end]], [meas_size],
                                     [meas_std],
                                     color = :black,
                                     whiskerwidth = ms_whiskerwidth,
                                     linewidth = ms_linewidtheb)
                end

                p = Makie.scatter!(ax1, measurements[m].observables.concentration.time[end], meas_size,
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

            predicted_size_str = "T = $(round(measurements[m].temperature - 273, digits = 2))°C, Pred. $(label_symbol) = $(round(predicted_size, sigdigits=3)) ± $(round(predicted_std, sigdigits=2)) μm"

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
            max_time = maximum([m.observables.concentration.time[end] for m in measurements])
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
            _save_plot(figure, savename, savedir)
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
- `measurements::Vector{<:AbstractExperiment}`: experimental measurement sets.
- `parameters::AbstractVector{<:Real}`: kinetic parameter vector.
- `nucleationfunction::AbstractNucleationFunction`: nucleation kinetic model.
- `growthfunction::AbstractGrowthFunction`: growth kinetic model.
- `aggregationfunction::AbstractAggregationFunction`: aggregation kinetic model.
- `breakagefunction::AbstractBreakageFunction`: breakage kinetic model.
- `solver::AbstractSolver`: population balance solver ([`MoM`](@ref), [`QMOM`](@ref),
  [`FiniteVol`](@ref), or [`WENO`](@ref)).

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
