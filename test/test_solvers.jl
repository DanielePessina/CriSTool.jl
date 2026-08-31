# Solver consistency tests - verify different solvers produce similar results
# Using parameter values based on README:
# nucl_CNT: [Aj, γ] ≈ [38, 0.7], growth_empirical: [Ag, g] ≈ [1, 3]

@testset "Solver Consistency" begin

    @testset "MoM vs FiniteVol Agreement" begin
        # Use the same kinetic parameters for both solvers
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()
        params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0
        save_times = collect(0:60.0:480.0)

        # Run with MoM
        _,
        sol_mom = runsimulation(params,
                                nucl_func,
                                grow_func,
                                initial_conc;
                                save_idx = save_times,
                                solver = MoM())

        # Run with FiniteVol
        _,
        sol_fv = runsimulation(params,
                               nucl_func,
                               grow_func,
                               noaggregation(),
                               nobreakage(),
                               initial_conc;
                               save_idx = save_times,
                               solver = FiniteVol(meshsize = 200, lmax = 50e-6))

        # Both should succeed
        @test sol_mom.success
        @test sol_fv.success

        # Final concentrations should be similar (within tolerance)
        # Note: MoM and FV use different discretizations so exact match is not expected
        conc_mom_final = sol_mom.concentration[end]
        conc_fv_final = sol_fv.concentration[end]

        relative_diff = abs(conc_mom_final - conc_fv_final) /
                        max(conc_mom_final, conc_fv_final)
        @test relative_diff < 0.3  # 30% tolerance for different numerical methods
    end

    @testset "FiniteVol Quantile Ordering" begin
        params = [38.0, 0.7, 1.0, 3.0]

        _,
        solution = runsimulation(params,
                                 nucl_CNT(),
                                 growth_empirical(),
                                 noaggregation(),
                                 nobreakage(),
                                 18.0;
                                 save_idx = collect(0:60.0:480.0),
                                 solver = FiniteVol(meshsize = 100, lmax = 50e-6))

        @test solution.success

        # Empty distributions are represented by equal zero quantiles, so the
        # ordering contract applies at every saved time.
        @test all(solution.d10q .<= solution.d50q)
        @test all(solution.d50q .<= solution.d90q)
    end

    @testset "WENO Solver Basic Test" begin
        params = [38.0, 0.7, 1.0, 3.0]

        _,
        solution = runsimulation(params,
                                 nucl_CNT(),
                                 growth_empirical(),
                                 noaggregation(),
                                 nobreakage(),
                                 18.0;
                                 save_idx = collect(0:120.0:480.0),
                                 solver = WENO(meshsize = 100, lmax = 50e-6))

        @test solution.success
        @test length(solution.time) > 0
    end

    @testset "Mesh Size Independence" begin
        # Results should be qualitatively similar with different mesh sizes
        params = [38.0, 0.7, 1.0, 3.0]
        save_times = collect(0:120.0:480.0)

        # Coarse mesh
        _,
        sol_coarse = runsimulation(params,
                                   nucl_CNT(),
                                   growth_empirical(),
                                   noaggregation(),
                                   nobreakage(),
                                   18.0;
                                   save_idx = save_times,
                                   solver = FiniteVol(meshsize = 50, lmax = 50e-6))

        # Fine mesh
        _,
        sol_fine = runsimulation(params,
                                 nucl_CNT(),
                                 growth_empirical(),
                                 noaggregation(),
                                 nobreakage(),
                                 18.0;
                                 save_idx = save_times,
                                 solver = FiniteVol(meshsize = 200, lmax = 50e-6))

        @test sol_coarse.success
        @test sol_fine.success

        # Final concentrations should be in the same ballpark
        ratio = sol_coarse.concentration[end] / sol_fine.concentration[end]
        @test 0.5 < ratio < 2.0  # Within factor of 2
    end

    @testset "ODE Solver Returns Success Code" begin
        params = [38.0, 0.7, 1.0, 3.0]

        # Test all solver types return proper success codes
        for solver in [MoM(), FiniteVol(meshsize = 50), WENO(meshsize = 50)]
            if solver isa MoM
                _,
                sol = runsimulation(params, nucl_CNT(), growth_empirical(), 18.0;
                                    solver = solver, save_idx = 0:120.0:240.0)
            else
                _,
                sol = runsimulation(params, nucl_CNT(), growth_empirical(),
                                    noaggregation(), nobreakage(), 18.0;
                                    solver = solver, save_idx = collect(0:120.0:240.0))
            end
            @test sol.success
        end
    end
end
