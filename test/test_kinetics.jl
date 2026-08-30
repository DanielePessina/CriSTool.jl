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

    # FV solver state: [numberdensity(mesh); C]. Set the state so that
    # the solute concentration gives the desired supersaturation.
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

    @testset "Migrated kinetic families use problem state" begin
        problem = make_test_problem()
        state = state_at_S(problem, 1.5)
        temperature = CriSTool.temperature(problem.temp_profile, 0.0)
        saturation = saturation_concentration(problem, 0.0)
        supersaturation_value = state[end] / saturation

        expected_cnt = (60 * exp(38.0)) * supersaturation_value *
                       exp(-16π * (0.7e-3)^3 * problem.molecular_volume^2 /
                           (3 * (problem.kb * temperature)^3 * log(supersaturation_value)^2))
        @test CriSTool.nucleationrate(CriSTool.nucl_CNT_fixed([38.0, 0.7]),
                                      Float64[], problem, state, 0.0) ≈ expected_cnt

        expected_empirical = (60 * 10.0^10.0) * (supersaturation_value - 1.0)^2.0
        @test CriSTool.nucleationrate(CriSTool.nucl_empirical_fixed([10.0, 2.0]),
                                      Float64[], problem, state, 0.0) ≈ expected_empirical

        expected_growth = 1.0e-9 * (supersaturation_value - 1.0)^3.0
        @test CriSTool.growthrate(CriSTool.growth_empirical_fixed([1.0, 3.0]),
                                  Float64[], problem, state, 0.0) ≈ expected_growth
    end

    @testset "Composite and loading-dependent kinetics dispatch" begin
        problem = make_test_problem()
        state = state_at_S(problem, 1.5)

        primary = CriSTool.nucl_empirical_energy()
        secondary = CriSTool.nucl_secondary()
        composite = CriSTool.nucl_prim_plus_second()
        composite_params = [8.0, 2.0, 6.0, 1.0, 2.0, 1.5]
        expected_composite =
            CriSTool.nucleationrate(primary, composite_params[1:3], problem, state, 0.0) +
            CriSTool.nucleationrate(secondary, composite_params[4:6], problem, state, 0.0)
        @test CriSTool.nucleationrate(composite, composite_params, problem, state, 0.0) ≈
              expected_composite

        multi_growth = CriSTool.growth_energy_multiloading([0.0, 1.0])
        loaded_problem = CrystallisationProblem(; kinetics_nucleationfunction = nucl_CNT(),
                                                 kinetics_growthfunction = multi_growth,
                                                 parameterset_nucleation = [38.0, 0.7],
                                                 parameterset_growth = [1.0, 2.0, 3.0, 4.0],
                                                 loading = 1.0,
                                                 solver = MoM())
        loaded_state = [zeros(5); 1.5 * saturation_concentration(loaded_problem, 0.0)]
        expected_loaded = exp10(3.0) *
                          exp(-multi_growth.Ea / (8.314 *
                                                  CriSTool.temperature(loaded_problem.temp_profile, 0.0))) *
                          (1.5 - 1.0)^4.0
        @test CriSTool.growthrate(multi_growth, [1.0, 2.0, 3.0, 4.0],
                                  loaded_problem, loaded_state, 0.0) ≈ expected_loaded
    end

    @testset "Dissolution uses the configured saturation model" begin
        solver = FiniteVol(meshsize = 4, lmax = 4e-6)
        problem = CrystallisationProblem(;
            kinetics_nucleationfunction = nucl_CNT(),
            kinetics_growthfunction = CriSTool.growth_dissolution(),
            saturation_model = ConstantSolubility(10.0),
            initial_concentration = 5.0,
            solver = solver)
        state = [zeros(solver.meshsize); 5.0]
        parameters = [2.0, 4.0, 1.5]
        temperature = CriSTool.temperature(problem.temp_profile, 0.0)
        expected_scalar = -(2.0e-9) * exp(-(4.0e3) / (8.314 * temperature)) *
                          (1.0 - 0.5)^1.5
        @test CriSTool.growthrate(CriSTool.growth_dissolution(), parameters,
                                  problem, state, 0.0) ≈ expected_scalar

        length_problem = CrystallisationProblem(;
            kinetics_nucleationfunction = nucl_CNT(),
            kinetics_growthfunction = CriSTool.growth_dissolution_length(),
            saturation_model = ConstantSolubility(10.0),
            initial_concentration = 5.0,
            solver = solver)
        expected_length = -(2.0e-9) .* exp(-(4.0e3) ./ (problem.R .* temperature)) .*
                          (1.0 - 0.5)^1.5 .*
                          (1.0 .+ solver.cell_centre ./ 1.0e-6) .^ 2.0
        @test CriSTool.growthrate(CriSTool.growth_dissolution_length(),
                                  [2.0, 4.0, 1.5, 1.0, 2.0],
                                  length_problem, state, 0.0) ≈ expected_length
    end

    @testset "Combined growth and dissolution dispatch" begin
        problem = make_test_problem()
        super_state = state_at_S(problem, 1.5)
        combined = CriSTool.growth_energy_dissolution()
        combined_parameters = [1.0, 3.0, 2.0, 4.0, 1.5]
        @test CriSTool.growthrate(combined, combined_parameters, problem,
                                  super_state, 0.0) ≈
              CriSTool.growthrate(CriSTool.growth_energy(), combined_parameters[1:2],
                                  problem, super_state, 0.0)

        under_problem = CrystallisationProblem(;
            kinetics_nucleationfunction = nucl_CNT(),
            kinetics_growthfunction = combined,
            saturation_model = ConstantSolubility(10.0),
            initial_concentration = 5.0,
            solver = MoM())
        under_state = [zeros(5); 5.0]
        @test CriSTool.growthrate(combined, combined_parameters, under_problem,
                                  under_state, 0.0) ≈
              CriSTool.growthrate(CriSTool.growth_dissolution(), combined_parameters[3:5],
                                  under_problem, under_state, 0.0)
    end

    @testset "Breakage rates dispatch on the configured mesh" begin
        solver = FiniteVol(meshsize = 4, lmax = 4e-6)
        problem = CrystallisationProblem(; kinetics_nucleationfunction = nucl_CNT(),
                                         kinetics_growthfunction = growth_empirical(),
                                         solver = solver)
        state = [ones(solver.meshsize); 2.0]
        empirical = CriSTool.breakagerate(CriSTool.breakage_empirical(),
                                           [0.5, 1.0], problem, state, 0.0)
        @test length(empirical) == solver.meshsize
        empirical_parent_length = solver.cell_centre[end]
        empirical_parent_frequency = 0.5 * empirical_parent_length^3
        empirical_same_cell_birth = empirical_parent_frequency *
                                     2 * (empirical_parent_length^3 - solver.cell_face[end - 1]^3) /
                                     empirical_parent_length^3
        @test empirical[end] ≈ empirical_same_cell_birth - empirical_parent_frequency

        uniform = CriSTool.breakagerate(CriSTool.breakage_uniform(),
                                         [log(0.5), 1.0], problem, state, 0.0)
        @test length(uniform) == solver.meshsize
        uniform_parent_length = solver.cell_centre[end]
        uniform_parent_frequency = 0.5 * (1e6 * uniform_parent_length)^3
        uniform_same_cell_birth = uniform_parent_frequency *
                                  2 * (uniform_parent_length^3 - solver.cell_face[end - 1]^3) /
                                  uniform_parent_length^3
        @test uniform[end] ≈ uniform_same_cell_birth - uniform_parent_frequency
    end

end
