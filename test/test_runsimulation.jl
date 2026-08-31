# Tests for the main runsimulation function
# Using parameter values based on README:
# nucl_CNT: [Aj, γ] ≈ [38, 0.7], growth_empirical: [Ag, g] ≈ [1, 3]

@testset "runsimulation" begin

    @testset "Basic Simulation - MoM Solver" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()

        # Total params = 2 (nucleation) + 2 (growth) = 4
        params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        problem,
        solution = runsimulation(params,
                                 nucl_func,
                                 grow_func,
                                 initial_conc;
                                 save_idx = 0:60.0:480.0,
                                 solver = MoM())

        # Basic checks
        @test solution.success
        @test length(solution.time) > 0
        @test length(solution.concentration) == length(solution.time)

        # Physical checks: the initial condition is retained and crystallization
        # cannot increase the dissolved concentration.
        @test solution.concentration[1] ≈ initial_conc
        @test all(diff(solution.concentration) .<= 1e-10)
    end

    @testset "Basic Simulation - FiniteVol Solver" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()
        agg_func = noaggregation()
        br_func = nobreakage()

        # Total params = 2 + 2 + 0 + 0 = 4
        params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        problem,
        solution = runsimulation(params,
                                 nucl_func,
                                 grow_func,
                                 agg_func,
                                 br_func,
                                 initial_conc;
                                 save_idx = collect(0:60.0:480.0),
                                 solver = FiniteVol(meshsize = 100, lmax = 50e-6))

        @test solution.success
        @test length(solution.time) > 0

        # Check that d10 <= d50 <= d90 at every saved time.  Empty
        # distributions are represented by equal zero quantiles and are still
        # part of the observable contract.
        @test all(solution.d10q .<= solution.d50q)
        @test all(solution.d50q .<= solution.d90q)
    end

    @testset "Simulation with configured aggregation and breakage functions" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()
        agg_func = aggr_scalar()
        br_func = breakage_empirical()

        # Use a short undersaturated run with an empty population.  This keeps
        # the integration test cheap while exercising the active operator
        # dispatch and the exact zero-population invariant.
        params = [38.0, 0.7, 1.0, 3.0, -6.0, 1e-3, 1.0]
        initial_conc = 1.0
        save_times = [0.0, 1.0, 2.0]

        problem,
        solution = runsimulation(params,
                                 nucl_func,
                                 grow_func,
                                 agg_func,
                                 br_func,
                                 initial_conc;
                                 save_idx = save_times,
                                 solver = FiniteVol(meshsize = 10, lmax = 10e-6))

        @test solution.success
        @test solution.concentration == fill(initial_conc, length(save_times))
        @test solution.d43 == zeros(length(save_times))
    end

    @testset "Keyword Interface" begin
        params = [38.0, 0.7, 1.0, 3.0]

        problem,
        solution = runsimulation(params;
                                 nucl = nucl_CNT(),
                                 gr = growth_empirical(),
                                 agg = noaggregation(),
                                 br = nobreakage(),
                                 solver = MoM(),
                                 initial_concentration = 18.0,
                                 save_idx = 0:120.0:480.0)

        @test solution.success
        @test problem isa CriSTool.CrystallisationProblem
    end

    @testset "Solution Structure - MoM" begin
        params = [38.0, 0.7, 1.0, 3.0]

        _,
        solution = runsimulation(params,
                                 nucl_CNT(),
                                 growth_empirical(),
                                 18.0;
                                 solver = MoM(),
                                 save_idx = 0:60.0:240.0)

        # Check MoM solution has expected fields
        @test hasproperty(solution, :time)
        @test hasproperty(solution, :concentration)
        @test hasproperty(solution, :d10)
        @test hasproperty(solution, :d32)
        @test hasproperty(solution, :d43)
        @test hasproperty(solution, :success)
    end

    @testset "Solution Structure - FiniteVol" begin
        params = [38.0, 0.7, 1.0, 3.0]

        _,
        solution = runsimulation(params,
                                 nucl_CNT(),
                                 growth_empirical(),
                                 noaggregation(),
                                 nobreakage(),
                                 18.0;
                                 solver = FiniteVol(meshsize = 50, lmax = 50e-6),
                                 save_idx = collect(0:60.0:240.0))

        # Check FV solution has expected fields
        @test hasproperty(solution, :time)
        @test hasproperty(solution, :concentration)
        @test hasproperty(solution, :numberdensity)
        @test hasproperty(solution, :voldensity)
        @test hasproperty(solution, :d10q)
        @test hasproperty(solution, :d50q)
        @test hasproperty(solution, :d90q)
        @test hasproperty(solution, :success)
    end

    @testset "Different Initial Concentrations" begin
        params = [38.0, 0.7, 1.0, 3.0]

        # Lower supersaturation
        _,
        sol_low = runsimulation(params,
                                nucl_CNT(),
                                growth_empirical(),
                                12.0;  # Lower initial concentration
                                solver = MoM(),
                                save_idx = 0:120.0:480.0)
        @test sol_low.success

        # High supersaturation
        _,
        sol_high = runsimulation(params,
                                 nucl_CNT(),
                                 growth_empirical(),
                                 25.0;  # Higher initial concentration
                                 solver = MoM(),
                                 save_idx = 0:120.0:480.0)
        @test sol_high.success
    end
end
