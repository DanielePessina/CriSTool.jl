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
    output_dir = mktempdir()

    draw_predictions = predict(plotting_model, plotting_experiments,
        ParameterSamples(growth = (speed = [0.005, 0.01, 0.02],)); solver = MoM())
    draw_figure = plot_predictions(draw_predictions, plotting_experiments; observables = plotted_observables,
        savename = "draws", savedir = output_dir, showplot = false)
    @test isfile(joinpath(output_dir, "draws Predictions.png"))
    draw_axes = panel_axes(draw_figure)
    @test length(draw_axes) == length(plotting_experiments) * length(plotted_observables)
    @test all(axis -> has_plot_type(axis, PlotMakie.Band), draw_axes)
    # Measured data exist for concentration and d43 only, never for d32.
    @test [has_plot_type(axis, PlotMakie.Scatter) for axis in draw_axes] == repeat([true, true, false], 2)

    # One draw is a line without a band; its size column is the simulated d43 in µm.
    single_predictions = predict(plotting_model, plotting_experiments[1:1],
        ParameterSamples(growth = (speed = [0.01],)); solver = MoM())
    single_figure = plot_predictions(single_predictions; observables = (:d43,), showplot = false)
    single_axis = only(panel_axes(single_figure))
    @test !has_plot_type(single_axis, PlotMakie.Band)
    @test !has_plot_type(single_axis, PlotMakie.Scatter)
    single_line = only(filter(axis_plot -> axis_plot isa PlotMakie.Lines, single_axis.scene.plots))
    saved_times = only(single_predictions.ensembles).time
    @test [point[1] for point in single_line[1][]] ≈ saved_times ./ 60
    @test [point[2] for point in single_line[1][]] ≈ fixture_translation_d43.(saved_times) .* 1e6 rtol = 1e-6
    @test_throws ArgumentError plot_predictions(single_predictions; observables = (:pH,), showplot = false)
    @test_throws ArgumentError plot_predictions(single_predictions; quantiles = (0.9, 0.1), showplot = false)

    posterior_draws = ParameterSamples(growth = (speed = 0.01 .+ 0.001 .* randn(MersenneTwister(4), 40),
        order = 2.0 .+ 0.1 .* randn(MersenneTwister(5), 40)))
    pairplot_figure = plot_posterior_pairplot(posterior_draws; reference = [0.01, 2.0],
        prior = product_distribution([Uniform(0.0, 0.02), Uniform(1.0, 3.0)]),
        savename = "posterior", savedir = output_dir, showplot = false)
    @test pairplot_figure isa PlotMakie.Figure
    @test isfile(joinpath(output_dir, "posterior Pairplot.png"))
    @test_throws DimensionMismatch plot_posterior_pairplot(posterior_draws; reference = [0.01], showplot = false)

    chain_draws = rand(MersenneTwister(7), 30, 2, 1)
    statistics_figure = ChainStatsPlots(Chains(chain_draws, [:speed, :order]); burnin = 5,
        saveplot = true, savestring = "stats", savedir = output_dir, showplot = false)
    @test statistics_figure isa PlotMakie.Figure
    # Axes alternate trace, density per parameter; the :order trace is the post-burn-in draws.
    statistics_axes = panel_axes(statistics_figure)
    @test length(statistics_axes) == 4
    order_trace = only(filter(axis_plot -> axis_plot isa PlotMakie.Lines, statistics_axes[3].scene.plots))
    @test [point[1] for point in order_trace[1][]] == 6:30
    @test [point[2] for point in order_trace[1][]] ≈ chain_draws[6:end, 2, 1] rtol = 1e-6
    @test has_plot_type(statistics_axes[4], PlotMakie.Density)
    @test isfile(joinpath(output_dir, "stats ChDensity.png"))
end
