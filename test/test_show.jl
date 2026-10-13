# REPL display (src/core/show.jl). Hand-built objects with deterministic values are
# compared against frozen strings, so a change to the display is a deliberate edit here.
using Metaheuristics
import OptimizationOptimJL
import SciMLBase

show_plain(displayed_object) = sprint(show, MIME"text/plain"(), displayed_object)

const show_seed = LogNormalInitialCrystals(mass_concentration = 0.25, d43 = 12.0e-6, geometric_std = 1.25)
const show_conditions = RunConditions(temperature = 293.15, concentration = 18.0, crystals = show_seed)

@testset "Display: run conditions and seeds" begin
    @test show_plain(show_conditions) == """
        RunConditions
          temperature:   ConstantTemperature(293.15 K)
          concentration: 18.0 kg/m³
          crystals:      LogNormalInitialCrystals(mass_concentration = 0.25 kg/m³, d43 = 12.0 µm, geometric_std = 1.25)
          operation:     BatchOperation()"""
    @test sprint(show, show_conditions) ==
          "RunConditions(ConstantTemperature(293.15 K), 18.0 kg/m³, seeded, BatchOperation())"
    # Placeholders stay visible; extra solvent values get their own row; no seed is "none".
    placeholder_conditions = RunConditions(temperature = 300.0, solvent = (pH = 7.0,))
    @test show_plain(placeholder_conditions) == """
        RunConditions
          temperature:   ConstantTemperature(300.0 K)
          concentration: FromObservation(:concentration)
          pH:            7.0
          crystals:      none
          operation:     BatchOperation()"""
    seed_placeholder = GaussianInitialCrystals(mass_concentration = FromObservation(:seed_mass), d43 = 10.0e-6,
        standard_deviation = 2.0e-6)
    @test sprint(show, seed_placeholder) == "GaussianInitialCrystals(mass_concentration = FromObservation(:seed_mass), " *
          "d43 = 10.0 µm, standard_deviation = 2.0 µm)"
    @test show_plain(seed_placeholder) == sprint(show, seed_placeholder)
    # d43 of the uniform [10, 20] µm number distribution is (20⁵-10⁵)/(20⁴-10⁴)/5*4 = 16.533… µm.
    @test sprint(show, fixture_uniform_seed()) == "DistributionInitialCrystals(Uniform{Float64}(a=1.0e-5, b=2.0e-5), " *
          "mass_concentration = 0.25 kg/m³, weighting = NumberWeighted(), d43 = 16.53 µm)"
    # A dual temperature shows its primal value.
    dual_conditions = RunConditions(temperature = ConstantTemperature(ForwardDiff.Dual(293.15, 1.0)), concentration = 18.0)
    @test occursin("ConstantTemperature(293.15 K)", show_plain(dual_conditions))
    @test sprint(show, LinearTemperature(300.0, -0.01)) == "LinearTemperature(T0 = 300.0 K, slope = -0.01 K/s)"
end

@testset "Display: operations and feeds" begin
    @test sprint(show, BatchOperation()) == "BatchOperation()"
    @test sprint(show, BatchOperation(volume = 2.5)) == "BatchOperation(volume = 2.5 m³)"
    @test sprint(show, fixture_clear_feed()) == "CrystallisationFeed(concentration = 2.0 kg/m³)"
    seeded_feed = CrystallisationFeed(concentration = time -> 1.0 + time, crystals = fixture_uniform_seed())
    @test sprint(show, seeded_feed) == "CrystallisationFeed(concentration = function of time, crystals = " *
          "DistributionInitialCrystals(Uniform{Float64}(a=1.0e-5, b=2.0e-5), mass_concentration = 0.25 kg/m³, " *
          "weighting = NumberWeighted(), d43 = 16.53 µm))"
    @test sprint(show, fixture_msmpr()) == "MSMPROperation(volume = 2.0 m³, inflow = 0.2 m³/s)"
    @test show_plain(fixture_msmpr()) == """
        MSMPROperation
          volume: 2.0 m³
          inflow: 0.2 m³/s
          feed:   CrystallisationFeed(concentration = 2.0 kg/m³)"""
    @test sprint(show, fixture_fed_batch(inflow = t -> 0.5)) ==
          "FedBatchOperation(initial_volume = 2.0 m³, inflow = function of time)"
end

@testset "Display: solvers" begin
    @test sprint(show, FiniteVol()) == "FiniteVol(200 cells)"
    @test sprint(show, WENO(meshsize = 64)) == "WENO(64 cells)"
    @test sprint(show, MoM()) == "MoM(4 moments)"
    @test sprint(show, QMOM(nquadrature = 2)) == "QMOM(2 nodes)"
    @test sprint(show, DQMOM()) == "DQMOM(3 nodes)"
    @test show_plain(FiniteVol(meshsize = 100, lmax = 80.0e-6)) == """
        FiniteVol
          mesh:       100 cells over 0.0 … 80.0 µm
          tolerances: reltol = 0.0001, abstol = 1.0e-10
          stepper:    solver default"""
    @test show_plain(QMOM(nquadrature = 2, timestepping_algorithm = Tsit5())) == """
        QMOM
          quadrature: 2 nodes, moments µ0 … µ3
          tolerances: reltol = 1.0e-7, abstol = 1.0e-8
          stepper:    Tsit5"""
    @test show_plain(MoM(nmoments = 5)) == """
        MoM
          moments:    µ0 … µ5
          tolerances: reltol = 1.0e-10, abstol = 1.0e-8
          stepper:    solver default"""
end

@testset "Display: kinetic and crystallisation models" begin
    dissolving_model = CrystallisationModel(fixture_system;
        growth = EmpiricalGrowth(growth_coefficient = 1.667e-11, growth_order = 3.0))
    @test sprint(show, dissolving_model.growth) == "EmpiricalGrowth(growth_coefficient = 1.667e-11, growth_order = 3.0)"
    @test sprint(show, dissolving_model.nucleation) == "NoNucleation()"
    @test show_plain(dissolving_model) == """
        CrystallisationModel
          density:     1000.0 kg/m³; shape factor 1.0
          solubility:  ConstantSolubility
          nucleation:  NoNucleation()
          growth:      EmpiricalGrowth(growth_coefficient = 1.667e-11, growth_order = 3.0)
          dissolution: NoDissolution()
          aggregation: NoAggregation()
          breakage:    NoBreakage()"""
    # The compact form lists only the active slots.
    @test sprint(show, dissolving_model) ==
          "CrystallisationModel(growth = EmpiricalGrowth(growth_coefficient = 1.667e-11, growth_order = 3.0))"
    # Nested parameter groups keep their nesting.
    nested_kinetic = CombinedNucleation(CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6.0e-4),
        EmpiricalNucleation(log10_nucleation_prefactor = 8.0, nucleation_order = 2.0))
    @test sprint(show, nested_kinetic) == "CombinedNucleation(primary = (ln_nucleation_prefactor = 38.0, " *
          "surface_energy = 0.0006), secondary = (log10_nucleation_prefactor = 8.0, nucleation_order = 2.0))"
end

# Population-by-time arrays are placeholders; only shapes and the headline trajectories are displayed.
function show_hand_solution(population; concentration = [18.0, 11.42], d43 = [12.0e-6, 41.7e-6], success = true,
        ode_stats = nothing)
    return CrystallisationSolution([0.0, 14400.0], concentration, population, d43, d43, d43, zero.(d43),
        (; concentration), (;), nothing, ode_stats, success)
end
const show_stats = let stats = SciMLBase.DEStats(1293)
    stats.naccept = 412
    stats.nreject = 31
    stats
end
const show_mesh_population = MeshPopulation(zeros(200, 2), zeros(200, 2), zeros(2), zeros(2), zeros(2))

@testset "Display: solutions" begin
    mesh_solution = show_hand_solution(show_mesh_population; ode_stats = show_stats)
    @test show_plain(mesh_solution) == """
        CrystallisationSolution{MeshPopulation} — 200-cell mesh, success
          time:          0.0 … 14400.0 s (2 saves)
          concentration: 18.0 → 11.42 kg/m³
          d43:           12.0 → 41.7 µm
          ODE:           1293 RHS evaluations, 412 accepted / 31 rejected steps"""
    @test sprint(show, mesh_solution) == "CrystallisationSolution(MeshPopulation, 2 saves, 0.0–14400.0 s, success)"
    failed_solution = show_hand_solution(MomentPopulation(zeros(5, 2)); success = false)
    @test startswith(show_plain(failed_solution), "CrystallisationSolution{MomentPopulation} — moments µ0 … µ4, failed\n")
    @test occursin("ODE:           not recorded", show_plain(failed_solution))
    @test endswith(sprint(show, failed_solution), ", failed)")
    untracked_solution = show_hand_solution(QuadraturePopulation(zeros(6, 2), zeros(3, 2), zeros(3, 2), nothing);
        d43 = [NaN, NaN])
    @test occursin("— 3-node quadrature, success", show_plain(untracked_solution))
    @test occursin("d43:           not tracked by this solver", show_plain(untracked_solution))
    # Dual-valued trajectories display their primal values and do not throw.
    dual_solution = show_hand_solution(show_mesh_population;
        concentration = ForwardDiff.Dual.([18.0, 11.42], 1.0), d43 = ForwardDiff.Dual.([12.0e-6, 41.7e-6], 1.0))
    @test occursin("concentration: 18.0 → 11.42 kg/m³", show_plain(dual_solution))
    @test occursin("d43:           12.0 → 41.7 µm", show_plain(dual_solution))

    # A real transient solve: the first concentration is the initial condition.
    simulated_solution = simulate(fixture_translation_model(), fixture_seeded_conditions(); solver = MoM(),
        saveat = [0.0, 100.0, 200.0])
    simulated_text = show_plain(simulated_solution)
    @test startswith(simulated_text, "CrystallisationSolution{MomentPopulation} — moments µ0 … µ4, success\n")
    @test occursin("time:          0.0 … 200.0 s (3 saves)", simulated_text)
    @test occursin("concentration: 20.0 → ", simulated_text)
    @test occursin(r"ODE:           \d+ RHS evaluations, \d+ accepted / \d+ rejected steps", simulated_text)

    steady_solution = CrystallisationSteadyStateSolution(nothing, (; concentration = 0.5, d43 = 8.0e-6), nothing, nothing,
        (; residence_time = 10.0), nothing, (; scaled_residual_norm = 1.0e-9), nothing, SciMLBase.ReturnCode.Success, true)
    @test show_plain(steady_solution) == """
        CrystallisationSteadyStateSolution — converged
          concentration:   0.5 kg/m³
          d43:             8.0 µm
          residence time:  10.0 s
          scaled residual: 1.0e-9
          retcode:         Success"""
    @test sprint(show, steady_solution) == "CrystallisationSteadyStateSolution(converged)"
end

@testset "Display: fit, inference and prediction results" begin
    show_specification = OptimisationSpec(bounds = (growth = (growth_coefficient = (1e-12, 1e-10),),),
        algorithm = OptimizationOptimJL.LBFGS())
    hand_fit = FitResult(nothing, (growth = (growth_coefficient = 1.667e-11,), nucleation = (ln_nucleation_prefactor = 38.21,)),
        (concentration = 0.02, d43 = 0.0121), nothing, (; specification = show_specification))
    @test show_plain(hand_fit) == """
        FitResult — OptimisationSpec(LBFGS), loss = 0.0321
          growth.growth_coefficient            1.667e-11
          nucleation.ln_nucleation_prefactor   38.21"""
    @test sprint(show, hand_fit) == "FitResult(OptimisationSpec(LBFGS), loss = 0.0321, 2 parameters)"
    @test occursin("OptimisationSpec(DE)", show_plain(FitResult(nothing, (growth = (speed = 0.01,),), (total = 1.0,), nothing,
        (; specification = OptimisationSpec(bounds = (;), algorithm = DE())))))

    # A fit through Optimization.jl reports convergence from its return code.
    lbfgs_experiment = fixture_experiment((time = [0.0, 300.0, 600.0],
            concentration = fixture_translation_concentration.([0.0, 300.0, 600.0]),
            d43 = fixture_translation_d43.([0.0, 300.0, 600.0])),
        RunConditions(temperature = 293.15, crystals = fixture_uniform_seed()); exp_id = 61)
    lbfgs_fit = fit(prepare_fit(fixture_translation_model(speed = 0.012), [lbfgs_experiment],
        OptimisationSpec(bounds = (growth = (speed = (0.001, 0.03),),), algorithm = OptimizationOptimJL.LBFGS(),
            searchoptions = (maxiters = 50,))))
    lbfgs_lines = split(show_plain(lbfgs_fit), '\n')
    @test startswith(lbfgs_lines[1], "FitResult — OptimisationSpec(LBFGS), loss = ")
    @test endswith(lbfgs_lines[1], ", converged")
    @test lbfgs_lines[2] == "  growth.speed   0.01"

    posterior_samples = ParameterSamples(growth = (speed = [0.01, 0.02, 0.03],); source = :nuts)
    @test sprint(show, posterior_samples) == "ParameterSamples(1 parameters, 3 draws, source = :nuts)"
    hand_inference = InferenceResult(nothing, posterior_samples, nothing,
        (; specification = BayesianSpec(priors = (growth = (speed = Uniform(0.001, 0.03),),))))
    @test show_plain(hand_inference) == """
        InferenceResult — BayesianSpec(NUTS), 3 draws
          growth.speed   0.02 ± 0.01"""
    @test sprint(show, hand_inference) == "InferenceResult(BayesianSpec(NUTS), 1 parameters, 3 draws)"

    prediction_times = [0.0, 100.0]
    hand_ensemble = CriSTool.PredictionEnsemble(prediction_times, (concentration = ones(3, 2), d43 = ones(3, 2)),
        [true, true, false], NamedTuple[])
    hand_predictions = PredictionResult(posterior_samples, [31], [hand_ensemble])
    @test show_plain(hand_predictions) == """
        PredictionResult — 1 runs, 3 draws
          run 31: 2 saves, observables (concentration, d43), 2/3 draws succeeded"""
    @test sprint(show, hand_predictions) == "PredictionResult(1 runs, 3 draws)"
end
