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

    @testset "Nucleation Rate - nucl_CNT" begin
        problem = make_test_problem()
        params = [38.0, 0.7]  # [Aj, γ] - reasonable values from README
        temp = 293.15
        numberdensity = zeros(50)
        loading = 0.0

        # Test: Supersaturated (S > 1) should give positive nucleation
        S_super = 1.5
        rate_super = CriSTool.nucleationrate(nucl_CNT(), params, S_super, problem, temp,
                                             loading, numberdensity)
        @test rate_super >= 0.0

        # Test: Undersaturated (S <= 1) should give zero nucleation
        S_under = 0.9
        rate_under = CriSTool.nucleationrate(nucl_CNT(), params, S_under, problem, temp,
                                             loading, numberdensity)
        @test rate_under == 0.0

        # Test: At saturation threshold (S ~ 1.001)
        S_threshold = 1.0001
        rate_threshold = CriSTool.nucleationrate(nucl_CNT(), params, S_threshold, problem,
                                                 temp, loading, numberdensity)
        @test rate_threshold == 0.0  # Below 1.001 threshold
    end

    @testset "Nucleation Rate - nucl_empirical" begin
        problem = make_test_problem()
        params = [10.0, 2.0]  # [Aj, j] - nucl_empirical has 2 params
        temp = 293.15
        numberdensity = zeros(50)
        loading = 0.0

        # Supersaturated should give positive rate
        S = 1.5
        rate = CriSTool.nucleationrate(nucl_empirical(), params, S, problem, temp, loading,
                                       numberdensity)
        @test rate >= 0.0

        # Undersaturated should give zero
        rate_under = CriSTool.nucleationrate(nucl_empirical(), params, 0.8, problem, temp,
                                             loading, numberdensity)
        @test rate_under == 0.0
    end

    @testset "Growth Rate - growth_empirical" begin
        problem = make_test_problem()
        params = [1.0, 3.0]  # [Ag, g] - reasonable values from README
        temp = 293.15
        numberdensity = zeros(50)
        loading = 0.0

        # Supersaturated should give positive growth
        S = 1.5
        rate = CriSTool.growthrate(growth_empirical(), params, S, problem, temp, loading,
                                   numberdensity)
        @test rate >= 0.0

        # Undersaturated should give zero
        rate_under = CriSTool.growthrate(growth_empirical(), params, 0.8, problem, temp,
                                         loading, numberdensity)
        @test rate_under == 0.0
    end

    @testset "Growth Rate - growth_BCF" begin
        problem = make_test_problem()
        params = [1.0, 1000.0]  # BCF parameters
        temp = 293.15
        numberdensity = zeros(50)
        loading = 0.0

        # Supersaturated should give positive growth
        S = 1.5
        rate = CriSTool.growthrate(growth_BCF(), params, S, problem, temp, loading,
                                   numberdensity)
        @test rate >= 0.0

        # Undersaturated should give zero
        rate_under = CriSTool.growthrate(growth_BCF(), params, 0.8, problem, temp, loading,
                                         numberdensity)
        @test rate_under == 0.0
    end

    @testset "Nucleation Rate Type Stability" begin
        problem = make_test_problem()
        params = [38.0, 0.7]
        S = 1.5
        temp = 293.15
        numberdensity = zeros(50)
        loading = 0.0

        # Test that nucleationrate returns a concrete type (not a Union)
        result = @inferred CriSTool.nucleationrate(nucl_CNT(), params, S, problem, temp,
                                                   loading, numberdensity)
        @test result isa Real
    end

    @testset "Growth Rate Type Stability" begin
        problem = make_test_problem()
        params = [1.0, 3.0]
        S = 1.5
        temp = 293.15
        numberdensity = zeros(50)
        loading = 0.0

        # Test that growthrate returns a concrete type
        result = @inferred CriSTool.growthrate(growth_empirical(), params, S, problem, temp,
                                               loading, numberdensity)
        @test result isa Real
    end

    @testset "Aggregation and Breakage - No-op functions" begin
        # noaggregation and nobreakage should return zero
        cell_centre = LinRange(0, 50e-6, 50)
        numberdensity = ones(50)
        params = Float64[]

        agg_rate = CriSTool.aggregationrate(noaggregation(), params, cell_centre,
                                            numberdensity)
        @test agg_rate == 0.0

        br_rate = CriSTool.breakagerate(nobreakage(), params, cell_centre, numberdensity)
        @test br_rate == 0.0
    end
end
