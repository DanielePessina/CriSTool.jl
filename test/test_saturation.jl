@testset "Saturation models" begin
    prof = CriSTool.ConstantTemperature(290.15)

    # Hand-computed polynomial values: 0.3705 + 7.171e-2 x - 1.924e-3 x^2 + 17.97e-5 x^3
    sm = lysozyme_solubility()
    @test sm isa PolynomialSolubility
    for T in (288.15, 290.15, 294.15, 298.15)
        x = T - 273.15
        expected = 0.3705 + 7.171e-2 * x - 1.924e-3 * x^2 + 17.97e-5 * x^3
        @test saturation_concentration(sm, CriSTool.ConstantTemperature(T), 0.0) ≈ expected
    end

    # Constant model ignores temperature and time
    @test saturation_concentration(ConstantSolubility(2.47), prof, 500.0) == 2.47

    # Callable model receives (T_K, t)
    callable = CallableSolubility((T, t) -> 1.0 + 1e-3 * t)
    @test saturation_concentration(callable, prof, 1800.0) ≈ 2.8

    # supersaturation dispatch on the problem
    problem = CrystallisationProblem(; saturation_model = ConstantSolubility(2.47),
                                     solver = MoM())
    @test supersaturation(problem, [0.0, 0.0, 0.0, 0.0, 0.0, 4.94], 0.0) ≈ 2.0
end

@testset "Generic MoM moments" begin
    run_std(p; nmoments = 4) = runsimulation(p; nucl = nucl_CNT(), gr = growth_empirical(),
                                             agg = noaggregation(), br = nobreakage(),
                                             initial_concentration = 14.67,
                                             save_idx = [0.0, 1800.0, 3600.0, 7200.0, 10800.0, 16200.0],
                                             solver = MoM(nmoments = nmoments))[2]
    p = [38.0, 0.0006, 1e-9 / 60, 3.0]

    sol4 = run_std(p)
    sol3 = run_std(p; nmoments = 3)
    sol2 = run_std(p; nmoments = 2)
    sol5 = run_std(p; nmoments = 5)

    # Physical identity: the (µ0, µ1, µ2, C) subsystem is closed for any
    # nmoments >= 2 -> the concentration trajectory is identical
    @test sol4.concentration ≈ sol3.concentration rtol = 1e-8
    @test sol4.concentration ≈ sol2.concentration rtol = 1e-8
    @test sol4.d43 ≈ sol5.d43 rtol = 1e-8  # d43 uses µ4/µ3 — not affected by µ5

    # Default reproduces the legacy 6-state model (d43 = µ4/µ3 etc.)
    @test sol4.concentration[end] ≈ 4.988605763490885 rtol = 1e-10
    @test sol4.d43[end] ≈ 5.2665996075480655e-6 rtol = 1e-10

    # Unavailable metrics are NaN, available ones are finite
    @test all(isnan, sol2.d43)
    @test all(isnan, sol2.d32)
    @test all(isfinite, sol2.concentration)
    @test all(isfinite, sol3.d32)
    @test all(isnan, sol3.d43)
end

@testset "Solution observables interface" begin
    p = [38.0, 0.0006, 1e-9 / 60, 3.0]
    _, sol = runsimulation(p; nucl = nucl_CNT(), gr = growth_empirical(),
                           agg = noaggregation(), br = nobreakage(),
                           initial_concentration = 14.67,
                           save_idx = [0.0, 3600.0, 7200.0, 14400.0],
                           solver = MoM())
    @test CriSTool.time(sol) == sol.time
    sv = state_vars(sol)
    @test sv.concentration === sol.concentration
    sm = size_metrics(sol)
    @test sm.d43 === sol.d43
    @test sm.d32 === sol.d32
    @test sm.moment2 === sol.moment2
    @test size_metrics(sol).d43[end] == CriSTool.get_characteristic_size(sol)

    _, fvsol = runsimulation(p; nucl = nucl_CNT(), gr = growth_empirical(),
                             agg = noaggregation(), br = nobreakage(),
                             initial_concentration = 14.67,
                             save_idx = collect(0.0:3600.0:14400.0),
                             solver = FiniteVol(meshsize = 50, lmax = 50e-6))
    @test state_vars(fvsol).numberdensity === fvsol.numberdensity
    @test size_metrics(fvsol).d50q === fvsol.d50q
    @test size_metrics(fvsol).d50q[end] == CriSTool.get_characteristic_size(fvsol)
end
