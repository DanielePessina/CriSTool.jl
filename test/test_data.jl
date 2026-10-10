# Measurement data: loaders, condition rules, variance balancing, bootstrap, and
# the frozen gold-fixture regression on the real experimental CSV (loaded once).
using JSON

const data_fixture_path = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.csv")
const data_fixture_runs = load_measurements(data_fixture_path)

@testset "Observable and experiment containers" begin
    scalar_variance_series = Observable(time = [0.0, 1.0], mean = [2.0, 3.0], variance = 0.25)
    @test scalar_variance_series.variance == [0.25, 0.25]
    @test Observable(time = [0.0, 1.0], mean = [1.0, 2.0]).variance === nothing
    @test_throws ArgumentError Observable(time = [0.0, 0.0], mean = [1.0, 2.0])
    @test_throws ArgumentError Observable(time = [0.0, 1.0], mean = [1.0, 2.0], variance = [0.1])
    container_experiment = CrystallisationExperiment(observables = (
            concentration = Observable(time = [0.0, 1800.0], mean = [14.0, 12.0]),),
        conditions = RunConditions(temperature = 290.15), exp_id = 7)
    @test initial_concentration(container_experiment) == 14.0
    @test container_experiment.conditions.temperature == ConstantTemperature(290.15)
    @test container_experiment.conditions.solvent.concentration == FromObservation(:concentration)
end

@testset "CSV loader: hand-checked values from the real fixture" begin
    @test [run.exp_id for run in data_fixture_runs] == [3, 4, 5, 6, 7, 8, 9]
    @test [run.conditions.temperature.value for run in data_fixture_runs] ==
          [290.15, 290.15, 294.15, 294.15, 294.15, 294.15, 294.15]
    @test all(run -> run.conditions.crystals === nothing, data_fixture_runs)
    first_concentration = data_fixture_runs[1].observables.concentration
    @test first_concentration.time == [0.0, 1800.0, 3600.0, 5400.0, 7200.0, 9000.0, 10800.0, 13500.0, 16200.0]
    # Row 1 of the sheet: mean of replicates 14.474, 14.757, 14.793.
    @test first_concentration.mean[1] ≈ (14.474 + 14.757 + 14.793) / 3
    @test first_concentration.variance[1] ≈ 0.17471214420678713
    @test first_concentration.mean[3] ≈ 7.7071
    # Particle size exists only at the final time.
    @test data_fixture_runs[1].observables.d43.time == [16200.0]
    @test data_fixture_runs[1].observables.d43.mean == [9.2480539e-6]
    @test data_fixture_runs[7].observables.d43.mean ≈ [11.868243e-6]
    @test data_fixture_runs[7].observables.d43.variance[1] ≈ 0.2669785799800902e-12

    renamed_runs = load_measurements(data_fixture_path;
        observables = (concentration = ObservableColumns(mean = :Concentration, variance = :Concentration_var),
            particle_size = ObservableColumns(time = :Time, mean = :PS, variance = :PS_var)),
        metadata_cols = (temperature = :Temperature, system = :System), filters = (System = "Unseeded",))
    @test propertynames(renamed_runs[1].observables) == (:concentration, :particle_size)
    @test renamed_runs[1].observables.particle_size.mean == data_fixture_runs[1].observables.d43.mean
    @test renamed_runs[1].metadata.system == "Unseeded"
    @test isempty(load_measurements(data_fixture_path; filters = (System = "UNKNOWN",)))
    @test [run.exp_id for run in load_measurements(data_fixture_path; temperature_range = (292.0, nothing))] ==
          [5, 6, 7, 8, 9]
end

@testset "Table loader: per-observable time grids and JSON records" begin
    repeated_size_table = DataFrame(Exp_ID = [1, 1, 1, 1, 2, 2, 2, 2],
        Time = [0.0, 1800.0, 3600.0, 5400.0, 0.0, 1800.0, 3600.0, 5400.0],
        Concentration = [20.0, 18.0, 15.0, 12.0, 19.0, 17.0, 14.0, 11.0],
        PS = [missing, 4e-6, missing, 7e-6, missing, 3.5e-6, 5.5e-6, missing],
        PS_var = [missing, 0.16e-12, missing, 0.49e-12, missing, 0.09e-12, 0.25e-12, missing],
        ParticleTime = [missing, 1800.0, missing, 5400.0, missing, 1800.0, 3600.0, missing],
        Temperature = fill(293.15, 8))
    repeated_size_runs = experiments_from_table(repeated_size_table;
        observables = (concentration = ObservableColumns(mean = :Concentration),
            d43 = ObservableColumns(time = :ParticleTime, mean = :PS, variance = :PS_var)),
        metadata_cols = (temperature = :Temperature,))
    @test repeated_size_runs[1].observables.d43.time == [1800.0, 5400.0]
    @test repeated_size_runs[1].observables.d43.variance == [0.16e-12, 0.49e-12]
    @test repeated_size_runs[2].observables.d43.mean == [3.5e-6, 5.5e-6]
    @test repeated_size_runs[2].observables.concentration.mean == [19.0, 17.0, 14.0, 11.0]

    # Records arrive out of time order and with null size entries.
    json_records = [
        (Exp_ID = 11, Time = 0.0, Concentration = 20.0, Temperature = 294.15, PS = 3e-6),
        (Exp_ID = 11, Time = 3600.0, Concentration = 15.0, Temperature = 294.15, PS = 6e-6),
        (Exp_ID = 11, Time = 1800.0, Concentration = 18.0, Temperature = 294.15, PS = nothing)]
    mktempdir() do json_directory
        json_path = joinpath(json_directory, "measurements.json")
        open(json_io -> JSON.print(json_io, json_records), json_path, "w")
        json_runs = load_measurements(json_path; observables = (concentration = ObservableColumns(mean = :Concentration),
            d43 = ObservableColumns(mean = :PS)), metadata_cols = (temperature = :Temperature,))
        @test only(json_runs).observables.concentration.mean == [20.0, 18.0, 15.0]
        @test only(json_runs).observables.d43.time == [0.0, 3600.0]
        @test only(json_runs).conditions.temperature.value == 294.15
    end
end

@testset "Table conditions keyword rules and seed columns" begin
    seed_table = DataFrame(Exp_ID = [1, 1], Time = [0.0, 1800.0], Concentration = [18.0, 16.0],
        Temperature = [293.15, 293.15], SeedMass = [0.25, 0.25], SeedD43 = [12e-6, 12e-6],
        SeedSpread = [2e-6, 2e-6])
    seed_columns = (mass_concentration = :SeedMass, d43 = :SeedD43, standard_deviation = :SeedSpread)
    seeded_run = only(experiments_from_table(seed_table; metadata_cols = (temperature = :Temperature,),
        initial_crystals_cols = seed_columns))
    @test seeded_run.conditions.crystals ==
          GaussianInitialCrystals(mass_concentration = 0.25, d43 = 12e-6, standard_deviation = 2e-6)
    # The mapped seed carries its mass and d43 into the moment state (rho*kv = 1000).
    seeded_problem = CrystallisationProblem(fixture_translation_model(), merge(seeded_run.conditions; concentration = 18.0))
    @test 1000.0 * seeded_problem.initial_population[4] ≈ 0.25 rtol = 1e-12
    @test seeded_problem.initial_population[5] / seeded_problem.initial_population[4] ≈ 12e-6 rtol = 1e-12

    # Temperature comes from exactly one place.
    @test_throws ArgumentError experiments_from_table(seed_table; metadata_cols = (temperature = :Temperature,),
        conditions = (temperature = 300.0,))
    @test_throws ArgumentError experiments_from_table(seed_table)
    shared_conditions_run = only(experiments_from_table(seed_table;
        conditions = (temperature = 300.0, crystals = fixture_uniform_seed())))
    @test shared_conditions_run.conditions.temperature.value == 300.0
    # Seeds come from exactly one place, and seed columns are all-or-nothing and constant.
    @test_throws ArgumentError experiments_from_table(seed_table; metadata_cols = (temperature = :Temperature,),
        initial_crystals_cols = seed_columns, conditions = (crystals = fixture_uniform_seed(),))
    @test_throws ArgumentError experiments_from_table(seed_table; metadata_cols = (temperature = :Temperature,),
        initial_crystals_cols = (mass_concentration = :SeedMass, d43 = :SeedD43))
    varying_seed_table = copy(seed_table)
    varying_seed_table.SeedMass = [0.25, 0.3]
    @test_throws ArgumentError experiments_from_table(varying_seed_table; metadata_cols = (temperature = :Temperature,),
        initial_crystals_cols = seed_columns)
    partial_seed_table = copy(seed_table)
    partial_seed_table.SeedD43 = [missing, missing]
    @test_throws ArgumentError experiments_from_table(partial_seed_table; metadata_cols = (temperature = :Temperature,),
        initial_crystals_cols = seed_columns)
end

@testset "Variance balancers raise variances to the relative floor" begin
    balance_run = CrystallisationExperiment(exp_id = 99, conditions = RunConditions(temperature = 290.15),
        observables = (concentration = Observable(time = [0.0, 1.0], mean = [10.0, 20.0], variance = [2.0, 0.5]),
            d43 = Observable(time = [1.0], mean = [10e-6], variance = [0.5e-12]),
            pH = Observable(time = [0.0, 1.0], mean = [7.0, 7.5])))
    # 10% floors: (1.0)^2 and (2.0)^2 for concentration; (1e-6)^2 for d43; 0.7^2, 0.75^2 for pH.
    # The legacy wrappers only accept an abstractly typed vector.
    @test only(repeatmeasurementbalancer(CrystallisationExperiment[balance_run], 10)).observables.concentration.variance == [2.0, 4.0]
    @test only(psd_measurementbalancer(CrystallisationExperiment[balance_run], 10)).observables.d43.variance ≈ [1e-12]
    balanced_ph_run = only(balance_variances([balance_run]; obs = :pH, min_rel_std_pc = 10))
    @test balanced_ph_run.observables.pH.variance ≈ [0.49, 0.5625]
    @test balanced_ph_run.observables.concentration === balance_run.observables.concentration
    @test balanced_ph_run.conditions === balance_run.conditions
    @test_throws ArgumentError balance_variances([balance_run]; obs = :d50q)
end

@testset "Bootstrap resampling keeps anchors, provenance and conditions" begin
    bootstrap_replicates = bootstrap_measurements(data_fixture_runs, 2; seed = 17)
    @test length(bootstrap_replicates) == 2
    @test bootstrap_measurements(data_fixture_runs, 1; seed = 17)[1][3].observables.concentration.mean ==
          bootstrap_replicates[1][3].observables.concentration.mean
    @test bootstrap_replicates[1] != bootstrap_replicates[2]
    for (original_run, resampled_run) in zip(data_fixture_runs, bootstrap_replicates[1])
        original_series, resampled_series = original_run.observables.concentration, resampled_run.observables.concentration
        @test resampled_run.exp_id == original_run.exp_id
        @test resampled_run.conditions === original_run.conditions
        @test resampled_run.metadata == original_run.metadata
        @test (resampled_series.time[1], resampled_series.mean[1]) == (original_series.time[1], original_series.mean[1])
        @test issorted(resampled_series.time)
        # Every resampled point is a genuine (time, mean) observation of the same run.
        original_points = Set(zip(original_series.time, original_series.mean))
        @test all(in(original_points), zip(resampled_series.time, resampled_series.mean))
    end
    anchor_only = bootstrap_measurements(data_fixture_runs, 1; observable_names = (:concentration,),
        n_samples = 0, seed = 1)[1]
    @test [run.observables.concentration.time for run in anchor_only] == fill([0.0], 7)
    @test anchor_only[1].observables.d43 === data_fixture_runs[1].observables.d43
    @test_throws ArgumentError bootstrap_measurements(data_fixture_runs, 1; observable_names = (:pH,))
    repeat_replicates = bootstrap_repeatmeasurements(data_fixture_runs, 2, true; seed = 1)
    @test length(repeat_replicates) == 2 && all(replicate -> length(replicate) == 7, repeat_replicates)
end

@testset "Gold fixture: frozen loss and trajectories on the real CSV" begin
    # Frozen after the SI migration; change only with a deliberate model change.
    gold_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    gold_bounds = (nucleation = (ln_nucleation_prefactor = (30.0, 45.0),),)
    gold_likelihood = prepare_fit(gold_model, data_fixture_runs, OptimisationSpec(bounds = gold_bounds, algorithm = nothing))
    gold_error = prepare_fit(gold_model, data_fixture_runs, OptimisationSpec(bounds = gold_bounds, algorithm = nothing, loss = MeanAbsoluteError()))
    @test loss(gold_likelihood, [38.0]) ≈ 3822.2905404364087 rtol = 1e-10
    @test loss(gold_error, [38.0]) ≈ 3.120095106302579 rtol = 1e-10

    gold_final_values = Dict(3 => (2.9810974132787518, 2.884586907371976e-6),
        4 => (3.821325524615847, 5.255372975716149e-6), 5 => (4.346709032677143, 3.4023333789723677e-6),
        6 => (4.271251947647894, 3.894665285170176e-6), 7 => (4.647118420680006, 4.511084798100737e-6),
        8 => (5.279557687291466, 6.4122082843203735e-6), 9 => (5.842153867404512, 7.3368062334872335e-6))
    gold_predictions = predict(gold_model, data_fixture_runs,
        ParameterSamples(nucleation = (ln_nucleation_prefactor = [38.0],),); solver = MoM(),
        observables = (:concentration, :d43))
    for (gold_run, gold_ensemble) in zip(data_fixture_runs, gold_predictions.ensembles)
        final_concentration, final_d43 = gold_final_values[gold_run.exp_id]
        gold_concentration = vec(observable_values(gold_ensemble, :concentration))
        @test gold_ensemble.time == gold_run.observables.concentration.time
        @test gold_concentration[end] ≈ final_concentration rtol = 1e-10
        @test observable_values(gold_ensemble, :d43)[1, end] ≈ final_d43 rtol = 1e-10
        # Unseeded batch: starts at the measured C0 and only consumes solute.
        @test gold_concentration[1] == initial_concentration(gold_run)
        @test all(diff(gold_concentration) .<= 1e-12)
    end
end
