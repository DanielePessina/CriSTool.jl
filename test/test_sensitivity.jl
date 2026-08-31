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

    # Verify the forward sensitivity against an independent central
    # difference of the public simulation interface.
    function final_concentration_at_Aj(Aj)
        _, solution = runsimulation([Aj, 0.6, 1.0, 3.0];
                                    nucl = nucl_CNT(), gr = growth_empirical(),
                                    solver = MoM(), initial_concentration = 14.67,
                                    temp_profile = CriSTool.ConstantTemperature(290.15),
                                    save_idx = saveat)
        return solution.concentration[end]
    end
    finite_difference_step = 1e-4
    finite_difference_sensitivity =
        (final_concentration_at_Aj(38.0 + finite_difference_step) -
         final_concentration_at_Aj(38.0 - finite_difference_step)) /
        (2 * finite_difference_step)
    @test dp[1][6, end] ≈ finite_difference_sensitivity rtol = 1e-4 atol = 1e-8

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
