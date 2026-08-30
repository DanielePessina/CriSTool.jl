# Gold fixture regression test.
#
# Ground truth: research/gold_fixture.jl run on 2026-08-28 with the v1.0
# measurements container and loss API, frozen to
# research/gold_fixture_output.json. These numbers must not change unless the
# model semantics deliberately change (then regenerate the gold and update
# here).

@testset "Gold fixture: loader + MoM at canonical params" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
    ms = load_experiments(fixture, "Unseeded_PE")
    @test length(ms) == 7

    problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_CNT(),
        kinetics_growthfunction = growth_empirical(),
        kinetics_aggregationfunction = noaggregation(),
        kinetics_breakagefunction = nobreakage(),
        solver = MoM())

    params = [38.0, 0.6, 1.0, 3.0]

    # Frozen loss values (logMLE and MAE, weightings (1,1))
    L_mle = loss(logMLE(), problem, params, ms)
    L_mae = loss(mae(), problem, params, ms)
    @test L_mle ≈ 3918.9027957924145 rtol = 1e-10
    @test L_mae ≈ 8.031591885353649 rtol = 1e-10

    # Frozen spot trajectories (final concentration, final d43 per experiment)
    expected_final = Dict(
        3 => (2.981097413279001, 2.8843131909147552),
        4 => (3.821325524623562, 5.254621067639209),
        5 => (4.346709032691379, 3.4020710664410574),
        6 => (4.271251947637579, 3.8943450335595258),
        7 => (4.6471184206776615, 4.5106730564994715),
        8 => (5.279557687295433, 6.411478717213067),
        9 => (5.8421538674079985, 7.335860001471332),
    )
    for m in ms
        sol = CriSTool._solve_experiment(problem, params, m)
        c_final, d43_final = expected_final[m.exp_id]
        @test sol.concentration[end] ≈ c_final rtol = 1e-10
        @test sol.d43[end] ≈ d43_final rtol = 1e-10
        @test sol.time == m.observables.concentration.time
    end

    # Physical identities: concentration is monotonically non-increasing for
    # these runs, d43 is non-decreasing, initial conc matches the measurement
    for m in ms
        sol = CriSTool._solve_experiment(problem, params, m)
        @test all(diff(sol.concentration) .<= 1e-12)
        @test all(diff(sol.d43) .>= -1e-12)
        @test sol.concentration[1] ≈ initial_concentration(m)
    end
end
