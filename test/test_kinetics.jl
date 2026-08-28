# Test kinetic rate functions (nucleation and growth)
# These functions are called millions of times in the ODE solver, so type stability is critical

@testset "Kinetics" begin

    # Setup a minimal CrystallisationProblem for testing kinetic functions
    # Using parameter values based on README: nucl_CNT: [Aj, γ] ≈ [38, 0.7], growth_empirical: [Ag, g] ≈ [1, 3]
    function make_test_problem(; solver = FiniteVol(meshsize = 50, lmax = 50e-6))
        CriSTool.CrystallisationProblem(kinetics_nucleationfunction = nucl_CNT(),
                                        kinetics_growthfunction = growth_empirical(),
                                        parameterset_nucleation = [38.0, 0.7],
                                        parameterset_growth = [1.0, 3.0],
                                        kinetics_aggregationfunction = noaggregation(),
                                        kinetics_breakagefunction = nobreakage(),
                                        initial_concentration = 18.0,
                                        solver = solver)
    end

    # FV solver state: [numberdensity(mesh); C]. Concentrate the state so that
    # the liquid concentration gives the desired supersaturation.
    function state_at_S(problem, S; meshsize = 50)
        C = S * saturation_concentration(problem, 0.0)
        return [zeros(meshsize); C]
    end

    @testset "Nucleation Rate - nucl_CNT" begin
        problem = make_test_problem()
        params = [38.0, 0.7]  # [Aj, γ] - reasonable values from README
        t = 0.0

        # Test: Supersaturated (S > 1) should give positive nucleation
        rate_super = CriSTool.nucleationrate(nucl_CNT(), params, problem,
                                             state_at_S(problem, 1.5), t)
        @test rate_super >= 0.0

        # Test: Undersaturated (S <= 1) should give zero nucleation
        rate_under = CriSTool.nucleationrate(nucl_CNT(), params, problem,
                                             state_at_S(problem, 0.9), t)
        @test rate_under == 0.0

        # Test: At saturation threshold (S ~ 1.001)
        rate_threshold = CriSTool.nucleationrate(nucl_CNT(), params, problem,
                                                 state_at_S(problem, 1.0001), t)
        @test rate_threshold == 0.0  # Below 1.001 threshold
    end

    @testset "Nucleation Rate - nucl_empirical" begin
        problem = make_test_problem()
        params = [10.0, 2.0]  # [Aj, j] - nucl_empirical has 2 params
        t = 0.0

        # Supersaturated should give positive rate
        @test CriSTool.nucleationrate(nucl_empirical(), params, problem,
                                      state_at_S(problem, 1.5), t) >= 0.0

        # Undersaturated should give zero
        @test CriSTool.nucleationrate(nucl_empirical(), params, problem,
                                      state_at_S(problem, 0.8), t) == 0.0
    end

    @testset "Growth Rate - growth_empirical" begin
        problem = make_test_problem()
        params = [1.0, 3.0]  # [Ag, g] - reasonable values from README
        t = 0.0

        # Supersaturated should give positive growth
        @test CriSTool.growthrate(growth_empirical(), params, problem,
                                  state_at_S(problem, 1.5), t) >= 0.0

        # Undersaturated should give zero
        @test CriSTool.growthrate(growth_empirical(), params, problem,
                                  state_at_S(problem, 0.8), t) == 0.0
    end

    @testset "Growth Rate - growth_BCF" begin
        problem = make_test_problem()
        params = [1.0, 1000.0]  # BCF parameters
        t = 0.0

        # Supersaturated should give positive growth
        @test CriSTool.growthrate(growth_BCF(), params, problem,
                                  state_at_S(problem, 1.5), t) >= 0.0

        # Undersaturated should give zero
        @test CriSTool.growthrate(growth_BCF(), params, problem,
                                  state_at_S(problem, 0.8), t) == 0.0
    end

    @testset "Nucleation Rate Type Stability" begin
        problem = make_test_problem()
        params = [38.0, 0.7]

        # Test that nucleationrate returns a concrete type (not a Union)
        result = @inferred CriSTool.nucleationrate(nucl_CNT(), params, problem,
                                                   state_at_S(problem, 1.5), 0.0)
        @test result isa Real
    end

    @testset "Growth Rate Type Stability" begin
        problem = make_test_problem()
        params = [1.0, 3.0]

        # Test that growthrate returns a concrete type
        result = @inferred CriSTool.growthrate(growth_empirical(), params, problem,
                                               state_at_S(problem, 1.5), 0.0)
        @test result isa Real
    end

    @testset "Aggregation and Breakage - No-op functions" begin
        # noaggregation and nobreakage should return zero
        problem = make_test_problem()
        numberdensity = ones(50)
        params = Float64[]

        state = [numberdensity; 18.0]
        agg_rate = CriSTool.aggregationrate(noaggregation(), params, problem, state, 0.0)
        @test agg_rate == 0.0

        br_rate = CriSTool.breakagerate(nobreakage(), params, problem, state, 0.0)
        @test br_rate == 0.0
    end

end