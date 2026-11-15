# Plotting lives in package extensions: CairoMakie loads CriSToolMakieExt and
# PairPlots loads CriSToolPairPlotsExt.
using Random
using MCMCChains: Chains
using CairoMakie: CairoMakie
using PairPlots: PairPlots

@testset "Prediction and posterior plots" begin
    PlotMakie = CairoMakie.Makie
    @test Base.get_extension(CriSTool, :CriSToolMakieExt) !== nothing
    @test Base.get_extension(CriSTool, :CriSToolPairPlotsExt) !== nothing
    plotting_model = fixture_translation_model()
    plotting_seed = fixture_uniform_seed()
    plotting_experiments = [CrystallisationExperiment(exp_id = run_id,
        observables = (concentration = Observable(time = [0.0, 60.0, 120.0],
                           mean = [20.0, 19.9, 19.8], variance = [0.0, 0.01, 0.01]),
                       d43 = Observable(time = [60.0, 120.0], mean = [16e-6, 17e-6], variance = [1e-14, 1e-14])),
        conditions = RunConditions(temperature = 293.15, crystals = plotting_seed)) for run_id in (7, 8)]
    plotted_observables = (:concentration, :d43, :d32)
    panel_axes(figure) = [figure_block for figure_block in figure.content if figure_block isa PlotMakie.Axis]
    has_plot_type(axis, plot_type) = any(axis_plot -> axis_plot isa plot_type, axis.scene.plots)

    draw_predictions = predict(plotting_model, plotting_experiments,
        ParameterSamples(growth = (speed = [0.005, 0.01, 0.02],)); solver = MoM())
    draw_figure = plot_predictions(draw_predictions, plotting_experiments; observables = plotted_observables)
    @test draw_figure isa PlotMakie.Figure
    draw_axes = panel_axes(draw_figure)
    @test length(draw_axes) == length(plotting_experiments) * length(plotted_observables)
    @test all(axis -> has_plot_type(axis, PlotMakie.Band), draw_axes)
    # Measured data exist for concentration and d43 only, never for d32.
    @test [has_plot_type(axis, PlotMakie.Scatter) for axis in draw_axes] == repeat([true, true, false], 2)

    # One draw is a line without a band; its size column is the simulated d43 in µm.
    single_predictions = predict(plotting_model, plotting_experiments[1:1],
        ParameterSamples(growth = (speed = [0.01],)); solver = MoM())
    single_figure = plot_predictions(single_predictions; observables = (:d43,))
    single_axis = only(panel_axes(single_figure))
    @test !has_plot_type(single_axis, PlotMakie.Band)
    @test !has_plot_type(single_axis, PlotMakie.Scatter)
    single_line = only(filter(axis_plot -> axis_plot isa PlotMakie.Lines, single_axis.scene.plots))
    saved_times = only(single_predictions.ensembles).time
    @test [point[1] for point in single_line[1][]] ≈ saved_times ./ 60
    @test [point[2] for point in single_line[1][]] ≈ fixture_translation_d43.(saved_times) .* 1e6 rtol = 1e-6
    @test_throws ArgumentError plot_predictions(single_predictions; observables = (:pH,))
    @test_throws ArgumentError plot_predictions(single_predictions; quantiles = (0.9, 0.1))

    posterior_draws = ParameterSamples(growth = (speed = 0.01 .+ 0.001 .* randn(MersenneTwister(4), 40),
        order = 2.0 .+ 0.1 .* randn(MersenneTwister(5), 40)))
    pairplot_figure = plot_posterior_pairplot(posterior_draws; reference = [0.01, 2.0],
        prior = product_distribution([Uniform(0.0, 0.02), Uniform(1.0, 3.0)]), title = "Posterior draws")
    @test pairplot_figure isa PlotMakie.Figure
    pairplot_labels = [figure_block.text[] for figure_block in pairplot_figure.content if figure_block isa PlotMakie.Label]
    @test "Posterior draws" in pairplot_labels
    untitled_pairplot = plot_posterior_pairplot(posterior_draws)
    @test !any(figure_block -> figure_block isa PlotMakie.Label && figure_block.text[] == "Posterior draws",
        untitled_pairplot.content)
    @test_throws DimensionMismatch plot_posterior_pairplot(posterior_draws; reference = [0.01])
    @test_throws DimensionMismatch plot_posterior_pairplot(posterior_draws; prior = Uniform(0.0, 0.02))

    chain_draws = rand(MersenneTwister(7), 30, 2, 1)
    statistics_figure = plot_chain_diagnostics(Chains(chain_draws, [:speed, :order]); burnin = 5,
        title = "NUTS chain")
    @test statistics_figure isa PlotMakie.Figure
    @test [figure_block.text[] for figure_block in statistics_figure.content if figure_block isa PlotMakie.Label] ==
          ["NUTS chain"]
    untitled_statistics = plot_chain_diagnostics(Chains(chain_draws, [:speed, :order]))
    @test !any(figure_block -> figure_block isa PlotMakie.Label, untitled_statistics.content)
    # Axes alternate trace, density per parameter; the :order trace is the post-burn-in draws.
    statistics_axes = panel_axes(statistics_figure)
    @test length(statistics_axes) == 4
    order_trace = only(filter(axis_plot -> axis_plot isa PlotMakie.Lines, statistics_axes[3].scene.plots))
    @test [point[1] for point in order_trace[1][]] == 6:30
    @test [point[2] for point in order_trace[1][]] ≈ chain_draws[6:end, 2, 1] rtol = 1e-6
    @test has_plot_type(statistics_axes[4], PlotMakie.Density)
    # Plotting functions return the figure; saving keywords no longer exist.
    @test_throws MethodError plot_predictions(single_predictions; savename = "draws", savedir = mktempdir())
    @test_throws MethodError plot_chain_diagnostics(Chains(chain_draws, [:speed, :order]); showplot = false)
end

@testset "Solution plot conversions" begin
    PlotMakie = CairoMakie.Makie
    recipe_times = [0.0, 600.0, 1800.0, 3600.0]
    recipe_solution = simulate(fixture_translation_model(), fixture_seeded_conditions();
        solver = MoM(), saveat = recipe_times)
    plotted_points(plot_object) = plot_object[1][]
    # Converted data equal the solution's own vectors, and the closed form.
    default_points = PlotMakie.convert_arguments(PlotMakie.PointBased(), recipe_solution)[1]
    @test [point[1] for point in default_points] == recipe_solution.time
    @test [point[2] for point in default_points] == recipe_solution.concentration
    @test [point[2] for point in default_points] ≈ fixture_translation_concentration.(recipe_times) rtol = 1e-6
    for observable_name in (:concentration, :d10, :d32, :d43, :moment2)
        observable_points = PlotMakie.convert_arguments(PlotMakie.PointBased(), recipe_solution, observable_name)[1]
        @test [point[1] for point in observable_points] == recipe_solution.time
        @test [point[2] for point in observable_points] == observable_values(recipe_solution, observable_name)
    end
    d43_points = PlotMakie.convert_arguments(PlotMakie.PointBased(), recipe_solution, :d43)[1]
    @test [point[2] for point in d43_points] ≈ fixture_translation_d43.(recipe_times) rtol = 1e-6
    @test_throws ArgumentError PlotMakie.convert_arguments(PlotMakie.PointBased(), recipe_solution, :pH)

    # Rendering through several plot types and the default `plot`.
    recipe_figure = PlotMakie.Figure()
    recipe_axis = PlotMakie.Axis(recipe_figure[1, 1])
    line_plot = PlotMakie.lines!(recipe_axis, recipe_solution, :d43)
    @test [point[2] for point in plotted_points(line_plot)] ≈ recipe_solution.d43 rtol = 1e-6
    @test PlotMakie.scatter!(recipe_axis, recipe_solution) isa PlotMakie.Scatter
    @test PlotMakie.stairs!(recipe_axis, recipe_solution, :d32) isa PlotMakie.Stairs
    @test PlotMakie.plottype(recipe_solution) === PlotMakie.Lines
    default_plot = PlotMakie.plot(recipe_solution)
    @test default_plot.plot isa PlotMakie.Lines
    @test [point[2] for point in plotted_points(default_plot.plot)] == recipe_solution.concentration
    @test PlotMakie.lines(recipe_solution, :d43).figure isa PlotMakie.Figure
    mktempdir() do output_dir
        PlotMakie.save(joinpath(output_dir, "recipe.png"), recipe_figure)
        @test isfile(joinpath(output_dir, "recipe.png"))
    end

    # Mesh size distribution: cell centres against the saved number density.
    mesh_solver = FiniteVol(meshsize = 40, lmax = 40e-6)
    mesh_solution = simulate(fixture_translation_model(), fixture_seeded_conditions();
        solver = mesh_solver, saveat = recipe_times)
    distribution_points = PlotMakie.convert_arguments(PlotMakie.PointBased(), mesh_solution, mesh_solver, 3)[1]
    @test [point[1] for point in distribution_points] == mesh_solver.cell_centre
    @test [point[2] for point in distribution_points] == mesh_solution.population.numberdensity[:, 3]
    # Integrating n(L) over the uniform seed recovers the seed number: mass / (rho kv E[L^3]).
    initial_points = PlotMakie.convert_arguments(PlotMakie.PointBased(), mesh_solution, mesh_solver, 1)[1]
    seed_number = fixture_seed_mass / 1000.0 / fixture_uniform_moment(3)
    @test sum(point[2] for point in initial_points) * step(mesh_solver.cell_centre) ≈ seed_number rtol = 2e-2
    distribution_figure = PlotMakie.Figure()
    distribution_axis = PlotMakie.Axis(distribution_figure[1, 1])
    @test PlotMakie.lines!(distribution_axis, mesh_solution, mesh_solver, 2) isa PlotMakie.Lines
    @test_throws BoundsError PlotMakie.convert_arguments(PlotMakie.PointBased(), mesh_solution, mesh_solver, 9)
    @test_throws DimensionMismatch PlotMakie.convert_arguments(PlotMakie.PointBased(), mesh_solution,
        FiniteVol(meshsize = 20, lmax = 40e-6), 1)
end
