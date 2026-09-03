# Input validation tests based on test_suggestions.md

@testset "Input Validation" begin

    @testset "Parameter Vector Sizing" begin
        nucl_func = nucl_CNT()      # nparams = 2
        grow_func = growth_empirical()  # nparams = 2
        agg_func = noaggregation()  # nparams = 0
        br_func = nobreakage()      # nparams = 0

        # Total required: 2 + 2 + 0 + 0 = 4 parameters
        # Using reasonable parameter values based on README
        correct_params = [38.0, 0.0007, 1e-9 / 60, 3.0]  # [Aj, γ, Ag, g]
        short_params = [38.0, 0.0007, 1.0]  # Too short (3 instead of 4)
        long_params = [38.0, 0.0007, 1e-9 / 60, 3.0, 0.0]  # Too long (5 instead of 4)

        # Correct params should work
        @test begin
            _,
            sol = runsimulation(correct_params, nucl_func, grow_func,
                                agg_func, br_func, 18.0;
                                solver = FiniteVol(meshsize = 50),
                                save_idx = collect(0:7200.0:14400.0))
            sol.success
        end

        # Short params should throw ArgumentError
        @test_throws ArgumentError runsimulation(short_params, nucl_func, grow_func,
                                                 agg_func, br_func, 18.0;
                                                 solver = FiniteVol(meshsize = 50),
                                                 save_idx = collect(0:7200.0:14400.0))

        # Long params should also throw ArgumentError (guards against extra params)
        @test_throws ArgumentError runsimulation(long_params, nucl_func, grow_func,
                                                 agg_func, br_func, 18.0;
                                                 solver = FiniteVol(meshsize = 50),
                                                 save_idx = collect(0:7200.0:14400.0))
    end

    @testset "Parameter Vector Sizing - Different Kinetics" begin
        # Test with empirical nucleation (2 params for nucl_empirical)
        nucl_emp = nucl_empirical()  # nparams = 2
        grow_func = growth_empirical()  # nparams = 2

        # Total required: 2 + 2 = 4 parameters
        correct_params = [10.0, 2.0, 1.0, 3.0]

        @test begin
            _,
            sol = runsimulation(correct_params, nucl_emp, grow_func, 18.0;
                                solver = MoM(),
                                save_idx = 0:7200.0:14400.0)
            sol.success
        end
    end

    @testset "Parameter Validation - Keyword Interface" begin
        # Test that keyword interface also validates parameter length
        correct_params = [38.0, 0.0007, 1e-9 / 60, 3.0]
        long_params = [38.0, 0.0007, 1e-9 / 60, 3.0, 0.0]
        short_params = [38.0, 0.0007, 1.0]

        # Correct should work
        @test begin
            _,
            sol = runsimulation(correct_params;
                                nucl = nucl_CNT(),
                                gr = growth_empirical(),
                                agg = noaggregation(),
                                br = nobreakage(),
                                solver = MoM(),
                                initial_concentration = 18.0,
                                save_idx = 0:7200.0:14400.0)
            sol.success
        end

        # Long params should throw
        @test_throws ArgumentError runsimulation(long_params;
                                                 nucl = nucl_CNT(),
                                                 gr = growth_empirical(),
                                                 agg = noaggregation(),
                                                 br = nobreakage(),
                                                 solver = MoM(),
                                                 initial_concentration = 18.0,
                                                 save_idx = 0:7200.0:14400.0)

        # Short params should throw
        @test_throws ArgumentError runsimulation(short_params;
                                                 nucl = nucl_CNT(),
                                                 gr = growth_empirical(),
                                                 agg = noaggregation(),
                                                 br = nobreakage(),
                                                 solver = MoM(),
                                                 initial_concentration = 18.0,
                                                 save_idx = 0:7200.0:14400.0)
    end

    @testset "Type Stability - runsimulation Return Types" begin
        params = [38.0, 0.0007, 1e-9 / 60, 3.0]

        # MoM solver
        problem_mom,
        solution_mom = runsimulation(params, nucl_CNT(), growth_empirical(), 18.0;
                                     solver = MoM(), save_idx = 0:7200.0:14400.0)

        @test problem_mom isa CriSTool.CrystallisationProblem
        @test solution_mom isa CrystallisationMoMSolution

        # FiniteVol solver
        problem_fv,
        solution_fv = runsimulation(params, nucl_CNT(), growth_empirical(),
                                    noaggregation(), nobreakage(), 18.0;
                                    solver = FiniteVol(meshsize = 50),
                                    save_idx = collect(0:7200.0:14400.0))

        @test problem_fv isa CriSTool.CrystallisationProblem
        @test solution_fv isa CrystallisationFVSolution
    end

    @testset "Solution Field Types" begin
        params = [38.0, 0.0007, 1e-9 / 60, 3.0]

        _,
        sol = runsimulation(params, nucl_CNT(), growth_empirical(), 18.0;
                            solver = MoM(), save_idx = 0:3600.0:14400.0)

        # All output arrays should be concrete types, not Any
        @test eltype(sol.time) <: Real
        @test eltype(sol.concentration) <: Real
        @test sol.success
    end

    @testset "CrystallisationProblem Type Stability" begin
        # Test that CrystallisationProblem construction is type-stable
        problem = CriSTool.CrystallisationProblem(kinetics_nucleationfunction = nucl_CNT(),
                                                  kinetics_growthfunction = growth_empirical(),
                                                  parameterset_nucleation = [38.0, 0.0007],
                                                  parameterset_growth = [1e-9 / 60, 3.0],
                                                  kinetics_aggregationfunction = noaggregation(),
                                                  kinetics_breakagefunction = nobreakage(),
                                                  initial_concentration = 18.0,
                                                  solver = FiniteVol(meshsize = 50))

        @test problem isa CriSTool.CrystallisationProblem
        @test problem.initial_concentration == 18.0
        @test problem.parameterset_nucleation == [38.0, 0.0007]
    end

    @testset "Edge Cases - Very Short Simulation" begin
        params = [38.0, 0.0007, 1e-9 / 60, 3.0]

        # Very short time span
        _,
        sol = runsimulation(params, nucl_CNT(), growth_empirical(), 18.0;
                            solver = MoM(), save_idx = 0:60.0:300.0)

        @test sol.success
        @test length(sol.time) >= 2
    end

    @testset "Edge Cases - Single Time Point" begin
        params = [38.0, 0.0007, 1e-9 / 60, 3.0]

        # Single time point (just initial condition)
        _,
        sol = runsimulation(params, nucl_CNT(), growth_empirical(), 18.0;
                            solver = MoM(), save_idx = [0.0])

        @test sol.success
        @test sol.concentration[1] ≈ 18.0  # Should be initial concentration
    end
end
