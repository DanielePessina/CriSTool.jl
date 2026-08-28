@testset "Forwardsensitivity (P0 regression)" begin
    prob_mom = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_CNT(),
        kinetics_growthfunction = growth_empirical(),
        solver = MoM(),
        parameterset_nucleation = [38.0, 0.6],
        parameterset_growth = [1.0, 3.0],
        initial_concentration = 14.67,
        temp_profile = CriSTool.ConstantTemperature(290.15))
    saveat = [0.0, 60.0, 120.0, 180.0]

    sol_mom = CriSTool.forwardsensitivity(prob_mom, saveat)
    sol_vals, dp = CriSTool.extract_local_sensitivities(sol_mom)
    @test size(sol_vals) == (6, 4)
    @test length(dp) == 4
    @test all(all(isfinite, dpi) for dpi in dp)

    # Sensitivity of the final concentration w.r.t. pre-exponential factor Aj
    # (dp[1] = ∂u/∂Aj; state index 6 = concentration): more nucleation ->
    # more crystal surface -> faster solute depletion -> ∂C/∂Aj < 0.
    # (∂C/∂g is NOT used here: its sign flips when S-1 crosses 1.)
    @test size(dp[1]) == (6, 4)
    @test dp[1][6, end] < 0

    prob_fv = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_CNT(),
        kinetics_growthfunction = growth_empirical(),
        solver = FiniteVol(meshsize = 100),
        parameterset_nucleation = [38.0, 0.6],
        parameterset_growth = [1.0, 3.0],
        initial_concentration = 14.67,
        temp_profile = CriSTool.ConstantTemperature(290.15))

    sol_fv = CriSTool.forwardsensitivity(prob_fv, saveat)
    sol_vals_fv, dp_fv = CriSTool.extract_local_sensitivities(sol_fv)
    @test size(sol_vals_fv) == (101, 4)
    @test all(all(isfinite, dpi) for dpi in dp_fv)
    # Concentration is the last state; more nucleation -> faster depletion
    @test size(dp_fv[1]) == (101, 4)
    @test dp_fv[1][101, end] < 0
end