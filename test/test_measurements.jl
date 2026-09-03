using JSON

@testset "Measurements container" begin
    # Construction and accessors
    conc = Observable(; time = [0.0, 1800.0, 3600.0], mean = [14.0, 12.0, 9.0],
                            variance = [0.1, 0.2, 0.3])
    d43 = Observable(; time = [3600.0], mean = [10.5e-6], variance = [2e-12])
    expt = CrystallisationExperiment(; observables = (; concentration = conc, d43 = d43),
                                     temperature = 290.15, exp_id = 7)

    @test expt.observables.concentration === conc
    @test expt.observables.d43 === d43
    @test expt.observables.d43.time == [3600.0]
    @test expt.observables.d43.mean == [10.5e-6]
    @test initial_concentration(expt) == 14.0
    @test expt.temperature == 290.15
    @test expt.exp_id == 7

    # Observable invariants
    @test length(conc.time) == length(conc.mean) == length(conc.variance)
    @test issorted(conc.time)

    # variance = nothing for single replicates
    single = Observable(; time = [0.0, 1.0], mean = [1.0, 2.0])
    @test single.variance === nothing
end

@testset "load_measurements on real fixture" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.csv")
    ms = load_measurements(fixture)

    @test length(ms) == 7
    @test [m.exp_id for m in ms] == [3, 4, 5, 6, 7, 8, 9]
    @test all(m.initial_crystals === nothing for m in ms)
    @test [m.temperature for m in ms] == [290.15, 290.15, 294.15, 294.15, 294.15, 294.15, 294.15]

    # Hand-checked from the sheet (Exp 3): 9 timepoints, initial conc = mean of
    # the three first-column replicates
    conc3 = ms[1].observables.concentration
    @test conc3.time == [0.0, 1800.0, 3600.0, 5400.0, 7200.0, 9000.0, 10800.0, 13500.0, 16200.0]
    @test conc3.mean[1] ≈ 14.674666666666667
    @test conc3.variance[1] ≈ 0.17471214420678713
    @test conc3.mean[3] ≈ 7.7071

    # Particle size is a one-point time series because the fixture has a
    # particle-size value only at the final time.
    @test ms[1].observables.d43.mean[1] ≈ 9.2480539e-6
    @test ms[1].observables.d43.variance[1] ≈ 5.345406308581576e-12
    @test ms[1].observables.d43.time == [conc3.time[end]]

    # Exp 9 (7th experiment) has a different final PS value.
    @test ms[7].observables.d43.mean[1] ≈ 11.868243e-6
    @test ms[7].observables.d43.variance[1] ≈ 0.2669785799800902e-12

    # Generic filtering still returns no data for an unknown system.
    @test isempty(load_measurements(fixture; filters = (; System = "UNKNOWN")))
end

@testset "Table-driven measurement loader" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.csv")
    measurements = load_measurements(
        fixture;
        observables = (; concentration = ObservableColumns(
                           mean = :Concentration, variance = :Concentration_var),
                       particle_size = ObservableColumns(
                           time = :Time, mean = :PS, variance = :PS_var)),
        metadata_cols = (; temperature = :Temperature, system = :System),
        filters = (; System = "Unseeded"))

    @test length(measurements) == 7
    @test propertynames(measurements[1].observables) == (:concentration, :particle_size)
    @test measurements[1].observables.concentration.time ==
          [0.0, 1800.0, 3600.0, 5400.0, 7200.0, 9000.0, 10800.0, 13500.0, 16200.0]
    @test measurements[1].observables.particle_size.time == [16200.0]
    @test measurements[1].observables.particle_size.mean[1] ≈ 9.2480539e-6
    @test measurements[1].temperature == 290.15
    @test measurements[1].initial_crystals === nothing
    @test measurements[1].metadata.system == "Unseeded"
end

@testset "Repeated particle-size observations" begin
    table = DataFrame(
        Exp_ID = [1, 1, 1, 1, 2, 2, 2, 2],
        Time = [0.0, 1800.0, 3600.0, 5400.0, 0.0, 1800.0, 3600.0, 5400.0],
        Concentration = [20.0, 18.0, 15.0, 12.0, 19.0, 17.0, 14.0, 11.0],
        Concentration_var = fill(0.25, 8),
        PS = [missing, 4e-6, missing, 7e-6, missing, 3.5e-6, 5.5e-6, missing],
        PS_var = [missing, 0.16e-12, missing, 0.49e-12, missing, 0.09e-12, 0.25e-12, missing],
        ParticleTime = [missing, 1800.0, missing, 5400.0,
                        missing, 1800.0, 3600.0, missing],
        Temperature = fill(293.15, 8),
    )

    experiments = experiments_from_table(
        table;
        observables = (; concentration = ObservableColumns(
                           mean = :Concentration, variance = :Concentration_var),
                       d43 = ObservableColumns(time = :ParticleTime,
                                              mean = :PS, variance = :PS_var)),
        metadata_cols = (; temperature = :Temperature),
    )

    @test length(experiments) == 2
    @test experiments[1].observables.d43.time == [1800.0, 5400.0]
    @test experiments[1].observables.d43.mean == [4e-6, 7e-6]
    @test experiments[1].observables.d43.variance == [0.16e-12, 0.49e-12]
    @test experiments[2].observables.d43.time == [1800.0, 3600.0]
    @test experiments[2].observables.d43.mean == [3.5e-6, 5.5e-6]

    # A scalar variance is normalized to one value per observation.
    normalized = Observable(; time = [0.0, 1.0], mean = [2.0, 3.0], variance = 0.25)
    @test normalized.variance == [0.25, 0.25]
    @test_throws ArgumentError Observable(; time = [0.0, 0.0], mean = [1.0, 2.0])
    @test_throws ArgumentError Observable(; time = [0.0, 1.0], mean = [1.0, 2.0],
                                          variance = [0.1])
end

@testset "JSON measurement adapter" begin
    records = [
        (; Exp_ID = 11, Time = 0.0, Concentration = 20.0,
           Concentration_var = 0.25, Temperature = 294.15,
           ParticleTime = 0.0, PS = 3e-6, PS_var = 0.09e-12),
        (; Exp_ID = 11, Time = 3600.0, Concentration = 15.0,
           Concentration_var = 0.25, Temperature = 294.15,
           ParticleTime = 3600.0, PS = 6e-6, PS_var = 0.36e-12),
        (; Exp_ID = 11, Time = 1800.0, Concentration = 18.0,
           Concentration_var = 0.25, Temperature = 294.15,
           ParticleTime = nothing, PS = nothing, PS_var = nothing),
    ]
    mktempdir() do directory
        filepath = joinpath(directory, "measurements.json")
        open(filepath, "w") do io
            JSON.print(io, records)
        end
        experiments = load_measurements(
            filepath;
            format = :json,
            observables = (; concentration = ObservableColumns(
                               mean = :Concentration, variance = :Concentration_var),
                           d43 = ObservableColumns(time = :ParticleTime,
                                                  mean = :PS, variance = :PS_var)),
            metadata_cols = (; temperature = :Temperature),
            temperature_transform = identity)
        @test length(experiments) == 1
        @test experiments[1].observables.d43.time == [0.0, 3600.0]
        @test experiments[1].observables.d43.mean == [3e-6, 6e-6]
        @test experiments[1].temperature == 294.15
    end
end

@testset "Balancers" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.csv")
    ms = load_measurements(fixture)

    # Concentration variance floor: 10% of the mean, squared
    balanced = repeatmeasurementbalancer(ms, 10)
    for (m, mb) in zip(ms, balanced)
        conc, concb = m.observables.concentration, mb.observables.concentration
        floor = (0.1 .* conc.mean) .^ 2
        @test all(concb.variance .>= floor .- 1e-12)
        # Points already above the floor are untouched
        above = conc.variance .>= floor
        @test all(concb.variance[above] .== conc.variance[above])
        # Points below the floor are raised exactly to it
        below = .!above
        @test all(concb.variance[below] .≈ floor[below])
    end

    # PSD variance floor: 10% std -> (0.1 * value)^2
    psd_balanced = psd_measurementbalancer(ms, 10)
    for m in psd_balanced
        @test all(m.observables.d43.variance .>=
                  (0.1 .* m.observables.d43.mean).^2 .- 1e-12)
    end

    custom = CrystallisationExperiment(;
        observables = (; pH = Observable(; time = [0.0, 1.0], mean = [7.0, 7.5])),
        temperature = 290.15, exp_id = 99)
    custom_balanced = balance_variances([custom]; obs = :pH, min_rel_std_pc = 10)
    @test custom_balanced[1].observables.pH.variance ≈ [0.49, 0.5625]
end

@testset "Bootstrap resampling" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.csv")
    ms = load_measurements(fixture)

    b1 = bootstrap_repeatmeasurements(ms, true; seed = 42)
    @test b1 isa Vector{CrystallisationExperiment}
    @test !isempty(b1)

    # Seeded determinism
    b2 = bootstrap_repeatmeasurements(ms, true; seed = 42)
    @test length(b1) == length(b2)
    for (e1, e2) in zip(b1, b2)
        c1, c2 = e1.observables.concentration, e2.observables.concentration
        @test c1.time == c2.time
        @test c1.mean == c2.mean
        @test e1.observables.d43.mean == e2.observables.d43.mean
    end

    # Resampled times are a subset of the original grid; initial point is kept
    orig_times = Set(vcat([m.observables.concentration.time for m in ms]...))
    for e in b1
        conc = e.observables.concentration
        orig = ms[findfirst(m -> m.exp_id == e.exp_id, ms)].observables.concentration
        @test conc.time[1] == orig.time[1]
        @test all(t -> t in orig_times, conc.time)
        @test all(diff(conc.time) .>= 0)
    end

    # Multiple bootstraps with different seeds produce (in general) different draws
    b3 = bootstrap_repeatmeasurements(ms, 2, true; seed = 1)
    @test length(b3) == 2
    @test all(sample -> length(sample) == length(ms), b3)
    @test all(sample -> all(experiment ->
                            !isempty(experiment.observables.concentration.mean),
                            sample), b3)

    generic1 = bootstrap_measurements(ms, 1; seed = 17)
    generic2 = bootstrap_measurements(ms, 1; seed = 17)
    @test length(generic1) == 1
    @test length(generic1[1]) == length(ms)
    @test generic1[1][1].observables.concentration.time ==
          generic2[1][1].observables.concentration.time
    @test generic1[1][1].observables.d43.mean == generic2[1][1].observables.d43.mean
    @test generic1[1][1].metadata == ms[1].metadata
end
