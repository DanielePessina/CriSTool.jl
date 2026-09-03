# Gold fixture regression test.
#
# Ground truth: the canonical SI measurements container and loss API, frozen
# after the numeric SI migration. These numbers must not change unless model
# semantics deliberately change (then regenerate the gold and update here).

@testset "Gold fixture: loader + MoM at canonical params" begin
    fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.csv")
    ms = load_measurements(fixture)
    @test length(ms) == 7

    problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_CNT(),
        kinetics_growthfunction = growth_empirical(),
        kinetics_aggregationfunction = noaggregation(),
        kinetics_breakagefunction = nobreakage(),
        solver = MoM())

    params = [38.0, 0.0006, 1e-9 / 60, 3.0]

    # Frozen loss values (logMLE and MAE, weightings (1,1))
    L_mle = loss(logMLE(), problem, params, ms)
    L_mae = loss(mae(), problem, params, ms)
    @test L_mle ≈ 3822.2905404364087 rtol = 1e-10
    @test L_mae ≈ 3.120095106302579 rtol = 1e-10

    # Frozen spot trajectories (final concentration, final d43 per experiment)
    expected_final = Dict(
        3 => (2.9810974132787518, 2.884586907371976e-6),
        4 => (3.821325524615847, 5.255372975716149e-6),
        5 => (4.346709032677143, 3.4023333789723677e-6),
        6 => (4.271251947647894, 3.894665285170176e-6),
        7 => (4.647118420680006, 4.511084798100737e-6),
        8 => (5.279557687291466, 6.4122082843203735e-6),
        9 => (5.842153867404512, 7.3368062334872335e-6),
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
