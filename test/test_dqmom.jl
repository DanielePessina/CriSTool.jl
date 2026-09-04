@testset "DQMOM" begin
    nquadrature = 3
    coordinate_scale = 1e-6
    weight_scale = 1e12
    dqmom_solver = DQMOM(nquadrature = nquadrature,
                         coordinate_scale = coordinate_scale,
                         weight_scale = weight_scale)

    no_nucleation = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                                         nucleation_order = 1.0)
    length_growth = growth_empirical_length()
    length_growth_parameters = [3e-9, 1.2, 0.25, 0.75]
    seeded_population = vcat([1.2e12, 2.0e12, 0.8e12],
                             [0.4e-6, 1.0e-6, 1.8e-6],
                             [20.0])

    @test DQMOM() isa AbstractMomentSolver
    @test moment_order(dqmom_solver) == 2 * nquadrature - 1

    function dqmom_test_problem(; growthfunction = length_growth,
                                growth_parameters = length_growth_parameters,
                                dissolutionfunction = nodissolution(),
                                dissolution_parameters = Float64[],
                                initial_state = seeded_population,
                                solver = dqmom_solver)
        return CrystallisationProblem(
            kinetics_nucleationfunction = no_nucleation,
            kinetics_growthfunction = growthfunction,
            kinetics_dissolutionfunction = dissolutionfunction,
            kinetics_aggregationfunction = noaggregation(),
            kinetics_breakagefunction = nobreakage(),
            parameterset_nucleation = Float64[],
            parameterset_growth = growth_parameters,
            parameterset_dissolution = dissolution_parameters,
            parameterset_aggregation = Float64[],
            parameterset_breakage = Float64[],
            saturation_model = ConstantSolubility(10.0),
            initial_concentration = 20.0,
            initial_state = initial_state,
            solver = solver)
    end

    @testset "Projection matrix" begin
        nodes = [0.5, 1.5]
        matrix = CriSTool._dqmom_projection_matrix(nodes)
        expected_matrix = [1.0 1.0 0.0 0.0;
                           0.0 0.0 1.0 1.0;
                           -0.25 -2.25 1.0 3.0;
                           -0.25 -6.75 0.75 6.75]
        @test matrix ≈ expected_matrix

        known_rates = [0.2, -0.1, 0.3, 0.4]
        expected_source = [
            sum(known_rates[1:2]),
            sum(known_rates[3:4]),
            sum((1 - 2) * nodes[index]^2 * known_rates[index]
                for index in 1:2) +
            sum(2 * nodes[index] * known_rates[2 + index] for index in 1:2),
            sum((1 - 3) * nodes[index]^3 * known_rates[index]
                for index in 1:2) +
            sum(3 * nodes[index]^2 * known_rates[2 + index] for index in 1:2)]
        @test matrix * known_rates ≈ expected_source
    end

    @testset "Binary source invariants" begin
        physical_nodes = [0.5e-6, 1.5e-6]
        normalized_weights = [0.25, 0.75]
        aggregation_source = CriSTool._aggregation_moment_source(
            aggr_scalar(), [0.0], physical_nodes, normalized_weights, 3;
            shape_factor = 0.81,
            pair_weight_scale = weight_scale)
        # With K = 1 and sum(q) = 1, the normalized total collision rate is
        # W/2.  Aggregation removes one particle per collision and preserves
        # the additive crystal volume exactly.
        @test aggregation_source[1] ≈ -weight_scale / 2
        @test aggregation_source[4] ≈ 0.0 atol = 1e-20

        breakage_source = CriSTool._breakage_moment_source(
            breakage_empirical(), [1.0, 0.0], physical_nodes,
            normalized_weights, 3)
        # Unit-frequency binary breakage creates one net daughter per parent
        # event and conserves parent volume under the uniform-in-volume law.
        @test breakage_source[1] ≈ 1.0
        @test breakage_source[4] == 0.0
    end

    @testset "Length-dependent empirical growth equation" begin
        problem = dqmom_test_problem()
        physical_node = 2.0e-6
        rate = growthrate_at_length(length_growth,
                                    length_growth_parameters,
                                    problem,
                                    seeded_population,
                                    0.0,
                                    physical_node)
        expected_rate = length_growth_parameters[1] *
                        (2.0 - 1.0)^length_growth_parameters[2] *
                        (1.0 + length_growth_parameters[3] *
                         (physical_node / length_growth.Lref)) ^
                        length_growth_parameters[4]
        @test rate ≈ expected_rate
        @test growthrate_at_length(length_growth,
                                   length_growth_parameters,
                                   problem,
                                   seeded_population,
                                   0.0,
                                   length_growth.Lref) ≈
              length_growth_parameters[1] * 1.0^length_growth_parameters[2] *
              (1.0 + length_growth_parameters[3])^length_growth_parameters[4]
    end

    @testset "Seeded initialization preserves independent characteristics" begin
        initial_crystals = LogNormalInitialCrystals(mass_concentration = 0.25,
                                                    d43 = 12e-6,
                                                    geometric_std = 1.25)
        problem = CrystallisationProblem(solver = dqmom_solver,
                                         initial_concentration = 20.0)
        direct_state = initial_state_from_characteristics(problem, initial_crystals)
        weights = @view direct_state[1:nquadrature]
        nodes = @view direct_state[(nquadrature + 1):(2 * nquadrature)]
        @test all(>(0.0), weights)
        @test all(>(0.0), nodes)
        @test problem.crystal_density * problem.volume_shape_factor *
              sum(weights .* nodes .^ 3) ≈ initial_crystals.mass_concentration rtol = 1e-10
        @test sum(weights .* nodes .^ 4) / sum(weights .* nodes .^ 3) ≈
              initial_crystals.d43 rtol = 1e-10
    end

    @testset "Seeded DQMOM growth and solvent coupling" begin
        problem = dqmom_test_problem()
        ode_problem, _ = crystallisation_odeproblem(problem, [0.0, 1.0])
        derivative = similar(ode_problem.u0)
        ode_problem.f(derivative, ode_problem.u0, ode_problem.p, 0.0)

        public_weights = seeded_population[1:nquadrature]
        public_nodes = seeded_population[(nquadrature + 1):(2 * nquadrature)]
        supersaturation_ratio = seeded_population[end] / 10.0
        expected_growth_rates = [
            length_growth_parameters[1] * (supersaturation_ratio - 1.0)^length_growth_parameters[2] *
            (1.0 + length_growth_parameters[3] *
             (public_nodes[index] / length_growth.Lref))^length_growth_parameters[4]
            for index in 1:nquadrature]
        expected_moment_rates = [
            moment_index == 0 ? 0.0 :
            moment_index * sum(public_weights[index] *
                               public_nodes[index]^(moment_index - 1) *
                               expected_growth_rates[index]
                               for index in 1:nquadrature)
            for moment_index in 0:(2 * nquadrature - 1)]

        scaled_nodes = [ode_problem.u0[nquadrature + index] /
                        ode_problem.u0[index] for index in 1:nquadrature]
        inferred_moment_rates = [
            weight_scale * coordinate_scale^moment_index *
            sum((1 - moment_index) * scaled_nodes[index]^moment_index *
                derivative[index] +
                (moment_index == 0 ? 0.0 :
                 moment_index * scaled_nodes[index]^(moment_index - 1) *
                 derivative[nquadrature + index])
                for index in 1:nquadrature)
            for moment_index in 0:(2 * nquadrature - 1)]
        @test inferred_moment_rates ≈ expected_moment_rates rtol = 1e-10 atol = 1e-20
        expected_concentration_rate = -problem.crystal_density *
                                      problem.volume_shape_factor *
                                      expected_moment_rates[4]
        @test derivative[end] ≈ expected_concentration_rate rtol = 1e-10 atol = 1e-12

        _, solution = runsimulation(vcat(length_growth_parameters);
                                    nucl = no_nucleation,
                                    gr = length_growth,
                                    agg = noaggregation(),
                                    br = nobreakage(),
                                    solver = dqmom_solver,
                                    initial_concentration = 20.0,
                                    initial_state = seeded_population,
                                    saturation_model = ConstantSolubility(10.0),
                                    save_idx = [0.0, 1.0])
        @test solution isa CrystallisationDQMOMSolution
        @test solution.moments[:, 1] ≈ [
            sum(public_weights[index] * public_nodes[index]^moment_index
                for index in 1:nquadrature)
            for moment_index in 0:(2 * nquadrature - 1)]
        @test solution.d43[1] ≈ sum(public_weights .* public_nodes .^ 4) /
                                sum(public_weights .* public_nodes .^ 3)
        @test solution.final_state isa Vector{Float64}
        @test size_metrics(solution).d43 === solution.d43
        @test observable_values(solution, :concentration) === solution.concentration
        @test quadrature(solution, 1).nodes ≈ public_nodes
        @test quadrature(solution, 1).weights ≈ public_weights
    end

    @testset "Constant-growth characteristic oracle" begin
        scalar_growth = growth_empirical()
        scalar_growth_parameters = [3e-9, 1.2]
        initial_weights = [1.2e12, 2.0e12, 0.8e12]
        initial_nodes = [0.4e-6, 1.0e-6, 1.8e-6]
        initial_concentration = 20.0
        growth_problem = CrystallisationProblem(
            kinetics_nucleationfunction = no_nucleation,
            kinetics_growthfunction = scalar_growth,
            parameterset_nucleation = Float64[],
            parameterset_growth = scalar_growth_parameters,
            saturation_model = ConstantSolubility(10.0),
            initial_concentration = initial_concentration,
            initial_state = vcat(initial_weights, initial_nodes,
                                 [initial_concentration]),
            solver = dqmom_solver)
        _, growth_solution = runsimulation(scalar_growth_parameters;
                                           nucl = no_nucleation,
                                           gr = scalar_growth,
                                           agg = noaggregation(),
                                           br = nobreakage(),
                                           solver = dqmom_solver,
                                           initial_concentration = initial_concentration,
                                           initial_state = vcat(initial_weights,
                                                                initial_nodes,
                                                                [initial_concentration]),
                                           saturation_model = ConstantSolubility(10.0),
                                           save_idx = [0.0, 10.0])
        growth_rate = scalar_growth_parameters[1]
        expected_nodes = initial_nodes .+ growth_rate * 10.0
        expected_moments = [sum(initial_weights[index] *
                                expected_nodes[index]^moment_index
                                for index in 1:nquadrature)
                            for moment_index in 0:(2 * nquadrature - 1)]
        expected_concentration = initial_concentration -
                                 growth_problem.crystal_density *
                                 growth_problem.volume_shape_factor *
                                 (expected_moments[4] -
                                  sum(initial_weights[index] * initial_nodes[index]^3
                                      for index in 1:nquadrature))
        @test growth_solution.weights[:, end] ≈ initial_weights rtol = 1e-8
        @test growth_solution.nodes[:, end] ≈ expected_nodes rtol = 1e-5 atol = 1e-12
        @test growth_solution.moments[:, end] ≈ expected_moments rtol = 1e-8
        @test growth_solution.concentration[end] ≈ expected_concentration rtol = 1e-8
    end

    @testset "Generic binary-source integration" begin
        binary_initial_state = copy(seeded_population)
        _, aggregation_solution = runsimulation(
            [0.0, 1.0, -12.0];
            nucl = no_nucleation,
            gr = growth_empirical(),
            agg = aggr_scalar(),
            br = nobreakage(),
            solver = dqmom_solver,
            initial_concentration = 20.0,
            initial_state = binary_initial_state,
            saturation_model = ConstantSolubility(10.0),
            save_idx = [0.0, 1e-3])
        @test aggregation_solution.success
        @test aggregation_solution.moments[4, end] ≈
              aggregation_solution.moments[4, 1] rtol = 1e-10
        @test aggregation_solution.concentration[end] ≈
              aggregation_solution.concentration[1] rtol = 1e-10

        _, breakage_solution = runsimulation(
            [0.0, 1.0, 1.0, 0.0];
            nucl = no_nucleation,
            gr = growth_empirical(),
            agg = noaggregation(),
            br = breakage_empirical(),
            solver = dqmom_solver,
            initial_concentration = 20.0,
            initial_state = binary_initial_state,
            saturation_model = ConstantSolubility(10.0),
            save_idx = [0.0, 1e-3])
        @test breakage_solution.success
        @test breakage_solution.moments[4, end] ≈
              breakage_solution.moments[4, 1] rtol = 1e-10
        @test breakage_solution.concentration[end] ≈
              breakage_solution.concentration[1] rtol = 1e-10
    end

    @testset "Seeded-only validation and unsupported dissolution" begin
        empty_problem = CrystallisationProblem(solver = dqmom_solver,
                                               initial_concentration = 20.0)
        @test_throws ArgumentError crystallisation_odeproblem(empty_problem, [0.0, 1.0])

        zero_weight_state = copy(seeded_population)
        zero_weight_state[1] = 0.0
        zero_weight_problem = dqmom_test_problem(initial_state = zero_weight_state)
        @test_throws ArgumentError crystallisation_odeproblem(zero_weight_problem,
                                                              [0.0, 1.0])

        repeated_node_state = copy(seeded_population)
        repeated_node_state[4] = repeated_node_state[5]
        repeated_node_problem = dqmom_test_problem(initial_state = repeated_node_state)
        @test_throws ArgumentError crystallisation_odeproblem(repeated_node_problem,
                                                              [0.0, 1.0])

        length_dissolution_problem = dqmom_test_problem(
            growthfunction = growth_empirical(),
            growth_parameters = [3e-9, 1.2],
            dissolutionfunction = growth_dissolution_length(),
            dissolution_parameters = [1e-9, 0.0, 1.5, 0.1, 1.0])
        @test_throws ArgumentError crystallisation_odeproblem(length_dissolution_problem,
                                                              [0.0, 1.0])

        scalar_dissolution_state = vcat([1.2e12, 2.0e12, 0.8e12],
                                        [0.5e-6, 1.0e-6, 1.5e-6],
                                        [5.0])
        @test_throws DomainError runsimulation(
            [0.0, 1.0, 1e-8, 0.0, 1.5];
            nucl = no_nucleation,
            gr = growth_empirical(),
            diss = growth_dissolution(),
            agg = noaggregation(),
            br = nobreakage(),
            solver = dqmom_solver,
            initial_concentration = 5.0,
            initial_state = scalar_dissolution_state,
            saturation_model = ConstantSolubility(10.0),
            save_idx = [0.0, 200.0])
    end

    @testset "DQMOM ForwardDiff agrees with finite differences" begin
        finite_difference_backend = DI.AutoFiniteDifferences(
            FiniteDifferences.central_fdm(5, 1))
        forward_difference_backend = DI.AutoForwardDiff()

        function dqmom_objective(normalized_parameters)
            si_growth_parameters = [normalized_parameters[1] * 1e-9,
                                    normalized_parameters[2],
                                    normalized_parameters[3],
                                    normalized_parameters[4]]
            _, solution = runsimulation(si_growth_parameters;
                                        nucl = no_nucleation,
                                        gr = length_growth,
                                        agg = noaggregation(),
                                        br = nobreakage(),
                                        solver = dqmom_solver,
                                        initial_concentration = 20.0,
                                        initial_state = seeded_population,
                                        saturation_model = ConstantSolubility(10.0),
                                        save_idx = [0.0, 1.0])
            return solution.concentration[end] + solution.d43[end]
        end

        test_parameters = [3.0, 1.2, 0.25, 0.75]
        finite_difference_gradient = DI.gradient(dqmom_objective,
                                                 finite_difference_backend,
                                                 test_parameters)
        forward_difference_gradient = DI.gradient(dqmom_objective,
                                                   forward_difference_backend,
                                                   test_parameters)
        @test forward_difference_gradient ≈ finite_difference_gradient rtol = 0.2 atol = 1e-8
    end
end
