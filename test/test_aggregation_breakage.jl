@testset "Aggregation and breakage kernel verification" begin

    function kernel_test_problem(meshsize = 3, lmax = 3.0)
        solver = FiniteVol(meshsize = meshsize, lmax = lmax)
        problem = CrystallisationProblem(; solver = solver,
                                         initial_solvent_state = (; concentration = 1.0))
        return problem, solver
    end

    @testset "Binary aggregation event accounting" begin
        problem, solver = kernel_test_problem()
        state = [1.0, 0.0, 0.0, 1.0]
        expected_single_parent_rate = [-27 / 52, 1 / 52, 0.0]

        for (aggregationfunction, kernel_scale) in
            ((aggr_scalar(), 1.0),
             (aggr_linear(), 1.0),
             (aggr_linearvol(), 0.2025),
             (aggr_avg(), 0.5))
            aggregation_rate = CriSTool.aggregationrate(aggregationfunction,
                                                         [0.0],
                                                         problem,
                                                         state,
                                                         0.0)
            @test aggregation_rate ≈ kernel_scale .* expected_single_parent_rate
            @test sum(aggregation_rate) * solver.cell_dL ≈ -0.5 * kernel_scale
            @test sum(solver.cell_centre .^ 3 .* aggregation_rate) * solver.cell_dL ≈ 0.0 atol = 1e-12
        end
    end

    @testset "Aggregation number and crystal-volume moments" begin
        problem, solver = kernel_test_problem()
        state = [1.0, 2.0, 0.0, 1.0]
        expected_event_rates =
            ((aggr_scalar(), 4.5),
             (aggr_linear(), 10.5),
             (aggr_linearvol(), 16.70625),
             (aggr_avg(), 5.25))

        for (aggregationfunction, expected_event_rate) in expected_event_rates
            aggregation_rate = CriSTool.aggregationrate(aggregationfunction,
                                                         [0.0],
                                                         problem,
                                                         state,
                                                         0.0)
            @test sum(aggregation_rate) * solver.cell_dL ≈ -expected_event_rate
            @test sum(solver.cell_centre .^ 3 .* aggregation_rate) * solver.cell_dL ≈ 0.0 atol = 1e-12
        end
    end

    @testset "Aggregation upper-boundary outflow" begin
        problem, solver = kernel_test_problem()
        state = [0.0, 0.0, 1.0, 1.0]
        rate = CriSTool.aggregationrate(aggr_scalar(),
                                        [0.0],
                                        problem,
                                        state,
                                        0.0)
        # Two largest represented parents would produce L^3 = 31.25,
        # beyond the upper mesh face L^3 = 27. The event leaves the tracked
        # domain instead of being fabricated at the final pivot.
        @test rate ≈ [0.0, 0.0, -1.0]
    end

    @testset "Uniform-in-volume binary breakage" begin
        problem, solver = kernel_test_problem()
        state = [0.0, 1.0, 0.0, 1.0]
        expected_rate = [16 / 27, 11 / 27, 0.0]

        empirical_rate = CriSTool.breakagerate(breakage_empirical(),
                                               [1.0, 0.0],
                                               problem,
                                               state,
                                               0.0)
        uniform_rate = CriSTool.breakagerate(breakage_uniform(),
                                             [0.0, 0.0],
                                             problem,
                                             state,
                                             0.0)

        # A parent at L=1.5 produces 2 ΔV/Vparent daughters in each
        # overlapping length cell. The parent death leaves one net particle.
        @test empirical_rate ≈ expected_rate
        @test uniform_rate ≈ expected_rate
        @test sum(empirical_rate) * solver.cell_dL ≈ 1.0
        @test sum(uniform_rate) * solver.cell_dL ≈ 1.0
    end

    @testset "Breakage daughter count for every resolved parent" begin
        problem, solver = kernel_test_problem()
        for parent_index in eachindex(solver.cell_centre)
            state = zeros(solver.meshsize + 1)
            state[parent_index] = 1.0
            state[end] = 1.0
            rate = CriSTool.breakagerate(breakage_empirical(),
                                         [1.0, 0.0],
                                         problem,
                                         state,
                                         0.0)
            @test sum(rate) * solver.cell_dL ≈ 1.0
        end
    end

    @testset "Breakage crystal-volume convergence" begin
        volume_moment_errors = Float64[]
        for meshsize in (20, 40, 80)
            problem, solver = kernel_test_problem(meshsize, 1.0)
            state = [ones(meshsize); 1.0]
            rate = CriSTool.breakagerate(breakage_empirical(),
                                         [1.0, 0.0],
                                         problem,
                                         state,
                                         0.0)
            # For constant selection rate and a uniform-in-volume binary
            # daughter law, the continuous M0 source is one parent per unit
            # length and the M3 source is exactly zero.
            @test sum(rate) * solver.cell_dL ≈ 1.0
            push!(volume_moment_errors,
                  abs(sum(solver.cell_centre .^ 3 .* rate) * solver.cell_dL))
        end
        @test volume_moment_errors[2] < volume_moment_errors[1] / 3
        @test volume_moment_errors[3] < volume_moment_errors[2] / 3
    end
end
