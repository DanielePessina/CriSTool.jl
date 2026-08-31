using ComponentArrays

@testset "Signed dissolution" begin
    dissolution_saturation = ConstantSolubility(10.0)
    fixed_no_nucleation = CriSTool.nucl_empirical_fixed([0.0, 1.0])

    function seeded_moment_problem()
        return CrystallisationProblem(
            kinetics_nucleationfunction = fixed_no_nucleation,
            kinetics_growthfunction = growth_dissolution(),
            parameterset_nucleation = Float64[],
            parameterset_growth = [1000.0, 0.0, 1.0],
            saturation_model = dissolution_saturation,
            initial_concentration = 5.0,
            initial_state = [1e12, 1e6, 1.0, 1e-6, 1e-12, 5.0],
            solid_volume_threshold = 5e-4,
            solver = MoM())
    end

    @testset "Rate law and deadband" begin
        problem = seeded_moment_problem()
        undersaturated_state = problem.initial_state
        expected = -(2.0e-9) * (1.0 - 0.5)^1.5
        @test CriSTool.growthrate(growth_dissolution(), [2.0, 0.0, 1.5], problem,
                                  undersaturated_state, 0.0) ≈ expected

        deadband_state = copy(undersaturated_state)
        deadband_state[end] = 9.995
        @test CriSTool.growthrate(growth_dissolution(), [2.0, 0.0, 1.5], problem,
                                  deadband_state, 0.0) == 0.0
    end

    @testset "Vectorized rate adapters agree with scalar laws" begin
        problem = CrystallisationProblem(
            kinetics_nucleationfunction = fixed_no_nucleation,
            kinetics_growthfunction = growth_empirical(),
            kinetics_dissolutionfunction = growth_dissolution(),
            parameterset_nucleation = Float64[],
            parameterset_growth = [1.0, 2.0],
            parameterset_dissolution = [2.0, 0.0, 1.5],
            saturation_model = dissolution_saturation,
            initial_concentration = 5.0,
            solver = FiniteVol(meshsize = 3, lmax = 3e-6))
        mesh = problem.solver.cell_centre
        super_state = [zeros(3); 15.0]
        under_state = [zeros(3); 5.0]

        growth_destination = zeros(3)
        growth_value = CriSTool.growthrate!(growth_destination,
                                            growth_empirical(), [1.0, 2.0],
                                            problem, super_state, 0.0, mesh)
        expected_growth = 1.0e-9 * 0.5^2.0
        @test growth_value === growth_destination
        @test growth_destination == fill(expected_growth, 3)
        @test CriSTool.growthrate_at_length(growth_empirical(), [1.0, 2.0],
                                            problem, super_state, 0.0, mesh[2]) ≈
              expected_growth

        dissolution_destination = zeros(3)
        dissolution_value = CriSTool.dissolutionrate!(
            dissolution_destination, growth_dissolution(), [2.0, 0.0, 1.5],
            problem, under_state, 0.0, mesh)
        expected_dissolution = -(2.0e-9) * 0.5^1.5
        @test dissolution_value === dissolution_destination
        @test dissolution_destination == fill(expected_dissolution, 3)
        @test CriSTool.dissolutionrate_at_length(
                  growth_dissolution(), [2.0, 0.0, 1.5], problem,
                  under_state, 0.0, mesh[2]) ≈ expected_dissolution

        net_destination = zeros(3)
        CriSTool.net_growth_rate!(net_destination, growth_empirical(),
                                  [1.0, 2.0], growth_dissolution(),
                                  [2.0, 0.0, 1.5], problem, under_state, 0.0,
                                  mesh)
        @test net_destination == fill(expected_dissolution, 3)
    end

    @testset "Independent growth and dissolution slots" begin
        problem = CrystallisationProblem(
            kinetics_nucleationfunction = fixed_no_nucleation,
            kinetics_growthfunction = growth_empirical(),
            kinetics_dissolutionfunction = growth_dissolution(),
            parameterset_nucleation = Float64[],
            parameterset_growth = [1.0, 2.0],
            parameterset_dissolution = [2.0, 0.0, 1.5],
            saturation_model = dissolution_saturation,
            initial_concentration = 5.0,
            solver = MoM())
        parameters = [1.0, 2.0, 2.0, 0.0, 1.5]
        _, solution = runsimulation(parameters;
                                    nucl = fixed_no_nucleation,
                                    gr = growth_empirical(),
                                    diss = growth_dissolution(),
                                    agg = noaggregation(),
                                    br = nobreakage(),
                                    solver = MoM(),
                                    initial_concentration = 5.0,
                                    saturation_model = dissolution_saturation,
                                    save_idx = [0.0, 0.1])
        @test solution.success
        structured_parameters = ComponentArray(parameters, paramaxis(problem))
        @test structured_parameters.diss.Ad == 2.0
        @test solution.concentration[end] == solution.concentration[1]
    end

    @testset "MoM extinction clears population and conserves residual mass" begin
        problem = seeded_moment_problem()
        solution = CriSTool._simulatecrystallisation(problem, collect(0.0:0.25:1.0))
        expected_total = 5.0 + problem.ρ * problem.kv * 1e-6
        @test solution.success
        @test solution.concentration[end] ≈ expected_total rtol = 1e-8
        @test all(iszero, solution.final_state[1:5])
        @test all(==(0.0), solution.d10[end:end])
        @test all(==(0.0), solution.d32[end:end])
        @test all(==(0.0), solution.d43[end:end])
    end

    @testset "Empty undersaturated populations are valid" begin
        for solver in (MoM(), FiniteVol(meshsize = 20, lmax = 4e-6),
                       WENO(meshsize = 20, lmax = 4e-6))
            parameters = [1000.0, 0.0, 1.0]
            _, solution = runsimulation(parameters;
                                        nucl = fixed_no_nucleation,
                                        gr = growth_dissolution(),
                                        agg = noaggregation(),
                                        br = nobreakage(),
                                        solver,
                                        initial_concentration = 5.0,
                                        saturation_model = dissolution_saturation,
                                        save_idx = [0.0, 0.1])
            @test solution.success
            @test all(==(5.0), solution.concentration)
            @test solution.d32[end] == 0.0
            @test solution.d43[end] == 0.0
        end
    end

    @testset "Length-dependent dissolution is FV/WENO-only" begin
        for solver in (FiniteVol(meshsize = 20, lmax = 4e-6),
                       WENO(meshsize = 20, lmax = 4e-6))
            initial_state = vcat(fill(1e12, solver.meshsize), 5.0)
            _, solution = runsimulation([1000.0, 0.0, 1.0, 1.0, 2.0];
                                        nucl = fixed_no_nucleation,
                                        gr = growth_dissolution_length(),
                                        agg = noaggregation(),
                                        br = nobreakage(),
                                        solver,
                                        initial_concentration = 5.0,
                                        initial_state,
                                        saturation_model = dissolution_saturation,
                                        save_idx = [0.0, 0.1])
            @test solution.success
            @test solution.concentration[end] > solution.concentration[1]
            @test minimum(solution.numberdensity) >= 0.0
        end

        moment_problem = CrystallisationProblem(
            kinetics_nucleationfunction = fixed_no_nucleation,
            kinetics_growthfunction = growth_dissolution_length(),
            parameterset_nucleation = Float64[],
            parameterset_growth = [1000.0, 0.0, 1.0, 1.0, 2.0],
            saturation_model = dissolution_saturation,
            initial_concentration = 5.0,
            solver = MoM())
        @test_throws ArgumentError crystallisation_odeproblem(moment_problem, [0.0, 0.1])
    end
end
