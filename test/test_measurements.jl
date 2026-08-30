@testset "Measurements container" begin
    # Construction and accessors
    conc = Observable(; time = [0.0, 30.0, 60.0], mean = [14.0, 12.0, 9.0],
                            variance = [0.1, 0.2, 0.3])
    d43 = Observable(; time = 60.0, mean = 10.5, variance = 2.0)
    expt = CrystallisationExperiment(; observables = (; concentration = conc, d43 = d43),
                                     temperature = 290.15, exp_id = 7)

    @test expt.observables.concentration === conc
    @test expt.observables.d43 === d43
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

@testset "load_experiments on real fixture" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    ms = load_experiments(fixture, "Unseeded_PE")

    @test length(ms) == 7
    @test [m.exp_id for m in ms] == [3, 4, 5, 6, 7, 8, 9]
    @test all(m.initial_crystals === nothing for m in ms)
    @test [m.temperature for m in ms] == [290.15, 290.15, 294.15, 294.15, 294.15, 294.15, 294.15]

    # Hand-checked from the sheet (Exp 3): 9 timepoints, initial conc = mean of
    # the three first-column replicates
    conc3 = ms[1].observables.concentration
    @test conc3.time == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0]
    @test conc3.mean[1] ≈ 14.674666666666667
    @test conc3.variance[1] ≈ 0.17471214420678713
    @test conc3.mean[3] ≈ 7.7071

    # Particle size: last timepoint PS, same value in d43 and d50q slots
    @test ms[1].observables.d43.mean ≈ 9.2480539
    @test ms[1].observables.d43.variance ≈ 5.345406308581576
    @test ms[1].observables.d43.mean == ms[1].observables.d50q.mean
    @test ms[1].observables.d43.time == conc3.time[end]

    # PS = -1 sentinel -> dummy 10.0 / 100.0 (Exp 9 has real PS).
    @test ms[7].observables.d43.mean ≈ 11.868243
    @test ms[7].observables.d43.variance ≈ 0.2669785799800902

    # Generic filtering still returns no data for an unknown system.
    @test isempty(load_experiments(fixture, "Unseeded_PE"; filters = (; System = "UNKNOWN")))
end

@testset "Table-driven measurement loader" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    measurements = load_measurements(
        fixture,
        "Unseeded_PE";
        observables = (; concentration = (:Concentration, :Concentration_var),
                       particle_size = (:PS, :PS_var)),
        scalar_observables = (:particle_size,),
        metadata_cols = (; temperature = :Temperature, system = :System),
        temperature_transform = value -> value + 273.15,
        filters = (; System = "Unseeded"))

    @test length(measurements) == 7
    @test propertynames(measurements[1].observables) == (:concentration, :particle_size)
    @test measurements[1].observables.concentration.time ==
          [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0]
    @test measurements[1].observables.particle_size.time == 270.0
    @test measurements[1].observables.particle_size.mean ≈ 9.2480539
    @test measurements[1].temperature == 290.15
    @test measurements[1].initial_crystals === nothing
    @test measurements[1].metadata.system == "Unseeded"
end

@testset "Balancers" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    ms = load_experiments(fixture, "Unseeded_PE")

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
        @test m.observables.d43.variance >= (0.1 * m.observables.d43.mean)^2 - 1e-12
        @test m.observables.d50q.variance >= (0.1 * m.observables.d50q.mean)^2 - 1e-12
    end

    custom = CrystallisationExperiment(;
        observables = (; pH = Observable(; time = [0.0, 1.0], mean = [7.0, 7.5])),
        temperature = 290.15, exp_id = 99)
    custom_balanced = balance_variances([custom]; obs = :pH, min_rel_std_pc = 10)
    @test custom_balanced[1].observables.pH.variance ≈ [0.49, 0.5625]
end

@testset "Bootstrap resampling" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    ms = load_experiments(fixture, "Unseeded_PE")

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
    @test all(isempty, b3) == false

    generic1 = bootstrap_measurements(ms, 1; seed = 17)
    generic2 = bootstrap_measurements(ms, 1; seed = 17)
    @test length(generic1) == 1
    @test length(generic1[1]) == length(ms)
    @test generic1[1][1].observables.concentration.time ==
          generic2[1][1].observables.concentration.time
    @test generic1[1][1].observables.d43.mean == generic2[1][1].observables.d43.mean
    @test generic1[1][1].metadata == ms[1].metadata
end

@testset "Legacy loaders" begin
    # The legacy "c i"/"q i"/"d i" sheet format has no committed fixture; verify
    # the API surface exists and is typed (thin importer kept for old files).
    @test hasmethod(load_experiments_legacy, Tuple{AbstractString, Int64})
    @test hasmethod(load_experiments_legacy, Tuple{AbstractString, Vector{Int64}})
    @test hasmethod(load_experiments_legacy_single, Tuple{AbstractString, Int64})
end
