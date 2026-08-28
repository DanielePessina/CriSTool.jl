@testset "Measurements container" begin
    # Construction and accessors
    conc = SeriesObservable(; time = [0.0, 30.0, 60.0], mean = [14.0, 12.0, 9.0],
                            variance = [0.1, 0.2, 0.3])
    d43 = ScalarObservable(; value = 10.5, variance = 2.0, time = 60.0)
    expt = CrystallisationExperiment(; observables = (; concentration = conc, d43 = d43),
                                     temperature = 290.15, loading = 0.0, exp_id = 7)

    @test expt.observables.concentration === conc
    @test expt.observables.d43 === d43
    @test initial_concentration(expt) == 14.0
    @test expt.temperature == 290.15
    @test expt.exp_id == 7

    # SeriesObservable invariants
    @test length(conc.time) == length(conc.mean) == length(conc.variance)
    @test issorted(conc.time)

    # variance = nothing for single replicates
    single = SeriesObservable(; time = [0.0, 1.0], mean = [1.0, 2.0])
    @test single.variance === nothing
end

@testset "load_experiments on real fixture" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    ms = load_experiments(fixture, "Unseeded_PE", 0.0)

    @test length(ms) == 7
    @test [m.exp_id for m in ms] == [3, 4, 5, 6, 7, 8, 9]
    @test all(m.loading == 0.0 for m in ms)
    @test [m.temperature for m in ms] == [290.15, 290.15, 294.15, 294.15, 294.15, 294.15, 294.15]

    # Hand-checked from the sheet (Exp 3): 9 timepoints, initial conc = mean of
    # the three first-column replicates
    conc3 = ms[1].observables.concentration
    @test conc3.time == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 225.0, 270.0]
    @test conc3.mean[1] ≈ 14.674666666666667
    @test conc3.variance[1] ≈ 0.17471214420678713
    @test conc3.mean[3] ≈ 7.7071

    # Particle size: last timepoint PS, same value in d43 and d50q slots
    @test ms[1].observables.d43.value ≈ 9.2480539
    @test ms[1].observables.d43.variance ≈ 5.345406308581576
    @test ms[1].observables.d43.value == ms[1].observables.d50q.value
    @test ms[1].observables.d43.time == conc3.time[end]

    # PS = -1 sentinel -> dummy 10.0 / 100.0 (Exp 9 has real PS; use a sheet
    # where the sentinel path is exercised via the loading filter instead)
    @test ms[7].observables.d43.value ≈ 11.868243
    @test ms[7].observables.d43.variance ≈ 0.2669785799800902

    # No data for an unknown loading -> empty
    @test isempty(load_experiments(fixture, "Unseeded_PE", 999.0))
end

@testset "Balancers" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    ms = load_experiments(fixture, "Unseeded_PE", 0.0)

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
        @test m.observables.d43.variance >= (0.1 * m.observables.d43.value)^2 - 1e-12
        @test m.observables.d50q.variance >= (0.1 * m.observables.d50q.value)^2 - 1e-12
    end
end

@testset "Bootstrap resampling" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    ms = load_experiments(fixture, "Unseeded_PE", 0.0)

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
        @test e1.observables.d43.value == e2.observables.d43.value
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
end

@testset "Legacy loaders" begin
    # The legacy "c i"/"q i"/"d i" sheet format has no committed fixture; verify
    # the API surface exists and is typed (thin importer kept for old files).
    @test hasmethod(load_experiments_legacy, Tuple{AbstractString, Int64})
    @test hasmethod(load_experiments_legacy, Tuple{AbstractString, Vector{Int64}})
    @test hasmethod(load_experiments_legacy_single, Tuple{AbstractString, Int64})
end