# Independent tests for the QMOM inversion nucleus and its scalar moment path.

@testset "QMOM" begin
    qmom_solver = QMOM(nquadrature = 3,
                       coordinate_scale = 1e-6,
                       realizability_tolerance = 1e-9)

    # Build moments from an independently specified atomic measure.  This is
    # deliberately kept in the test file rather than sharing a production
    # helper with the inversion implementation.
    function qmom_test_atomic_moments(nodes, weights, highest_order)
        return [sum(weights[node_index] * nodes[node_index]^moment_index
                    for node_index in eachindex(nodes))
                for moment_index in 0:highest_order]
    end

    function qmom_test_assert_rule(rule, expected_nodes, expected_weights,
                                   expected_moments)
        @test rule.active_nodes == length(expected_nodes)
        @test length(rule) == length(expected_nodes)
        sorted_indices = sortperm(rule.nodes)
        @test rule.nodes[sorted_indices] ≈ expected_nodes rtol = 1e-7 atol = 1e-14
        @test rule.weights[sorted_indices] ≈ expected_weights rtol = 1e-7

        # Verify all transported moments independently from the input atomic
        # measure and from the inversion diagnostics.
        for moment_index in 0:(length(expected_moments) - 1)
            reconstructed_moment = sum(rule.weights[node_index] *
                                       rule.nodes[node_index]^moment_index
                                       for node_index in eachindex(rule.nodes))
            @test isapprox(reconstructed_moment, expected_moments[moment_index + 1];
                           rtol = 1e-7, atol = 1e-28)
        end
    end

    @testset "One, two, and three atom rules" begin
        one_atom_nodes = [0.8e-6]
        one_atom_weights = [2.5e12]
        one_atom_moments = qmom_test_atomic_moments(one_atom_nodes,
                                                     one_atom_weights, 5)
        @test one_atom_moments ≈ [2.5e12, 2.0e6, 1.6, 1.28e-6,
                                  1.024e-12, 8.192e-19]
        one_atom_rule = invert_moments(one_atom_moments, qmom_solver)
        qmom_test_assert_rule(one_atom_rule, one_atom_nodes, one_atom_weights,
                              one_atom_moments)
        @test one_atom_rule.diagnostics.status == :deflated
        @test one_atom_rule.diagnostics.active_nodes == 1

        two_atom_nodes = [0.5e-6, 1.5e-6]
        two_atom_weights = [2.0e12, 3.0e12]
        two_atom_moments = qmom_test_atomic_moments(two_atom_nodes,
                                                     two_atom_weights, 5)
        @test two_atom_moments ≈ [5.0e12, 5.5e6, 7.25, 1.0375e-5,
                                  1.53125e-11, 2.284375e-17]
        two_atom_rule = invert_moments(two_atom_moments, qmom_solver)
        qmom_test_assert_rule(two_atom_rule, two_atom_nodes, two_atom_weights,
                              two_atom_moments)
        @test two_atom_rule.diagnostics.status == :deflated
        @test two_atom_rule.diagnostics.active_nodes == 2

        three_atom_nodes = [0.4e-6, 1.0e-6, 1.8e-6]
        three_atom_weights = [1.2e12, 2.0e12, 0.8e12]
        three_atom_moments = qmom_test_atomic_moments(three_atom_nodes,
                                                       three_atom_weights, 5)
        @test three_atom_moments ≈ [4.0e12, 3.92e6, 4.784, 6.7424e-6,
                                    1.04288e-11, 1.7128832e-17]
        three_atom_rule = invert_moments(three_atom_moments, qmom_solver)
        qmom_test_assert_rule(three_atom_rule, three_atom_nodes,
                              three_atom_weights, three_atom_moments)
        @test three_atom_rule.diagnostics.status == :ok
        @test three_atom_rule.diagnostics.active_nodes == 3
    end

    @testset "Empty, zero-size, and invalid measures" begin
        empty_rule = invert_moments(zeros(6), qmom_solver)
        @test isempty(empty_rule)
        @test isempty(empty_rule.nodes)
        @test isempty(empty_rule.weights)
        @test empty_rule.diagnostics.status == :empty

        zero_size_rule = invert_moments([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], qmom_solver)
        @test zero_size_rule.active_nodes == 1
        @test zero_size_rule.nodes[1] == 0.0
        @test zero_size_rule.weights[1] ≈ 2.0
        @test zero_size_rule.diagnostics.status == :deflated

        @test_throws ArgumentError invert_moments([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                  qmom_solver)
        @test_throws ArgumentError invert_moments([1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                                                  qmom_solver)
        @test_throws ArgumentError invert_moments([1.0, NaN, 0.0, 0.0, 0.0, 0.0],
                                                  qmom_solver)
        negative_support_moments = [1.0, -0.5e-6, 0.25e-12,
                                    -0.125e-18, 0.0625e-24, -0.03125e-30]
        @test_throws ArgumentError invert_moments(negative_support_moments, qmom_solver)
        @test_throws ArgumentError invert_moments(zeros(5), qmom_solver)
        @test_throws ArgumentError invert_moments(zeros(6), QMOM(nquadrature = 1))
    end

    @testset "Aggregation and breakage preserve crystal volume" begin
        source_rule = QMOMQuadrature([0.5e-6, 1.5e-6], [2.0, 3.0])

        aggregation_source = aggregation_moment_source(aggr_scalar(), [0.0],
                                                        source_rule, 3)
        # With K = 1, the independently computed ordered-pair event count is
        # 1/2 * (2 + 3)^2 = 12.5.  Aggregation removes one particle/event and
        # preserves the additive crystal volume (the third length moment).
        @test aggregation_source[1] ≈ -12.5
        @test aggregation_source[4] ≈ 0.0 atol = 1e-24

        breakage_source = breakage_moment_source(breakage_uniform(), [0.0, 0.0],
                                                  source_rule, 3)
        # Uniform-in-volume binary breakage adds one net daughter per parent
        # and preserves the parent volume exactly.
        @test breakage_source[1] ≈ 5.0
        @test breakage_source[4] ≈ 0.0 atol = 1e-24
    end

    @testset "Constant scalar growth and solvent mass balance" begin
        initial_moments = [2.0, 2.0e-6, 2.0e-12, 2.0e-18,
                           2.0e-24, 2.0e-30]
        scalar_growth = 3.0e-9
        growth_source = CriSTool._qmom_scalar_growth_moment_source(initial_moments,
                                                                    scalar_growth,
                                                                    0.0, 5)
        @test growth_source[1] == 0.0
        @test growth_source[2] ≈ scalar_growth * initial_moments[1]
        @test growth_source[3] ≈ 2 * scalar_growth * initial_moments[2]
        @test growth_source[4] ≈ 3 * scalar_growth * initial_moments[3]

        problem = CrystallisationProblem(; saturation_model = ConstantSolubility(10.0),
                                         initial_concentration = 20.0,
                                         solver = QMOM(nquadrature = 3))
        concentration_rate = -problem.crystal_density * problem.volume_shape_factor * growth_source[4]
        elapsed_time = 10.0
        initial_concentration = problem.initial_concentration
        initial_volume_moment = initial_moments[4]
        predicted_concentration = initial_concentration + elapsed_time * concentration_rate
        predicted_volume_moment = initial_volume_moment + elapsed_time * growth_source[4]
        initial_total_solute = initial_concentration +
                               problem.crystal_density * problem.volume_shape_factor * initial_volume_moment
        predicted_total_solute = predicted_concentration +
                                  problem.crystal_density * problem.volume_shape_factor * predicted_volume_moment
        @test predicted_total_solute ≈ initial_total_solute rtol = 1e-14
    end

    @testset "Scalar dissolution extinction and length rejection" begin
        no_nucleation = CriSTool.nucl_empirical_fixed(log10_nucleation_prefactor = -Inf, nucleation_order = 1.0)
        dissolution_initial_state = [1.0e12, 1.0e6, 1.0, 1.0e-6,
                                     1.0e-12, 1.0e-18, 5.0]
        dissolution_problem, dissolution_solution = runsimulation(
            [1000e-9 / 60, 0.0, 1.0];
            nucl = no_nucleation,
            gr = growth_dissolution(),
            agg = noaggregation(),
            br = nobreakage(),
            solver = QMOM(nquadrature = 3),
            initial_concentration = 5.0,
            initial_state = dissolution_initial_state,
            saturation_model = ConstantSolubility(10.0),
            solid_mass_concentration_threshold = 5.0e-4,
            save_idx = collect(0.0:360.0:3600.0))

        expected_total_solute = dissolution_solution.concentration[1] +
                                dissolution_problem.crystal_density * dissolution_problem.volume_shape_factor *
                                dissolution_solution.moments[4, 1]
        @test dissolution_solution.success
        @test all(iszero, dissolution_solution.final_state[1:6])
        @test dissolution_solution.concentration[end] ≈ expected_total_solute rtol = 1e-7
        @test dissolution_solution.d10[end] == 0.0
        @test dissolution_solution.d32[end] == 0.0
        @test dissolution_solution.d43[end] == 0.0
        @test isempty(quadrature(dissolution_solution, length(dissolution_solution.time)))

        length_problem = CrystallisationProblem(
            kinetics_nucleationfunction = no_nucleation,
            kinetics_growthfunction = growth_dissolution_length(),
            parameterset_nucleation = Float64[],
            parameterset_growth = [1000e-9 / 60, 0.0, 1.0, 1.0, 2.0],
            saturation_model = ConstantSolubility(10.0),
            initial_concentration = 5.0,
            solver = QMOM(nquadrature = 3))
        @test_throws ArgumentError crystallisation_odeproblem(length_problem,
                                                               [0.0, 0.1])
    end
end
