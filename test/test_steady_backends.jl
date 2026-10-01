@testset "Steady scalar-growth MSMPR across solver representations" begin
    backend_reference = steady_state_test_problem(nucleation_rate = 3e5,
        growth_rate = 2e-7, feed_concentration = 1.0,
        initial_concentration_value = 1.0, initial_state = [zeros(5); 1.0])
    for steady_solver in (QMOM(), FiniteVol(meshsize = 100, lmax = 50e-6),
                           WENO(meshsize = 100, lmax = 50e-6))
        configured_backend = CriSTool._copy_crystallisation_problem(backend_reference;
            solver = steady_solver, initial_state = nothing)
        backend_equilibrium = solve_steadystate(configured_backend;
            relaxation_horizon = 1000.0, autonomous = true,
            residual_reltol = 1e-8, residual_abstol = 1e-10)
        @test backend_equilibrium.success
        # Independent ideal MSMPR reference: d43 = 4 G tau = 8 µm.
        # Mesh solutions approximate this reference on their finite cells.
        @test only(observable_values(backend_equilibrium, :d43)) ≈ 8e-6 rtol =
            (steady_solver isa QMOM ? 2e-6 : 0.02)
        @test only(backend_equilibrium.concentration) ≈ 0.9999998402032 atol = 5e-9
        @test backend_equilibrium.scaled_residual_norm <= 1.01e-8
        @test backend_equilibrium.diagnostics.total_api_balance.relative_error < 1e-5
        if steady_solver isa QMOM
            @test backend_equilibrium.physical_result.moments[:, 1] ≈
                [3e6, 6.0, 2.4e-5, 1.44e-10, 1.152e-15, 1.152e-20] rtol = 2e-6
            @test size(state_vars(backend_equilibrium).moments) == (6, 1)
        elseif steady_solver isa WENO
            population_range = CriSTool._population_state_range(configured_backend)
            steady_population = backend_equilibrium.diagnostics.ode_final_state[population_range]
            population_diagnostics = backend_equilibrium.diagnostics.population_density_diagnostics
            population_scale = maximum(abs, steady_population)
            @test population_diagnostics.minimum == minimum(steady_population)
            @test population_diagnostics.scale == population_scale
            @test minimum(backend_equilibrium.physical_result.numberdensity) ==
                  population_diagnostics.minimum
            @test population_diagnostics.negative_count ==
                  count(value -> value < 0.0, steady_population)
            @test 0.0 < population_diagnostics.roundoff_tolerance < 1.0
            @test backend_equilibrium.diagnostics.physical_state_status ==
                  (population_diagnostics.negative_count == 0 ? :ok :
                   :roundoff_negative_number_density)

            roundoff_state = copy(backend_equilibrium.diagnostics.ode_final_state)
            roundoff_state[last(population_range)] = -3.334541490692464e-16
            roundoff_valid, roundoff_status = CriSTool._steady_physical_constraints(
                configured_backend, roundoff_state, nothing)
            @test roundoff_valid
            @test roundoff_status == :roundoff_negative_number_density

            material_negative_state = copy(roundoff_state)
            material_negative_state[first(population_range)] = -1.0
            material_valid, material_status = CriSTool._steady_physical_constraints(
                configured_backend, material_negative_state, nothing)
            @test !material_valid
            @test material_status == :negative_number_density

            zero_population_state = copy(backend_equilibrium.diagnostics.ode_final_state)
            zero_population_state[population_range] .= 0.0
            zero_valid, zero_status = CriSTool._steady_physical_constraints(
                configured_backend, zero_population_state, nothing)
            @test zero_valid
            @test zero_status == :ok

            all_negative_state = copy(zero_population_state)
            all_negative_state[population_range] .= -1.0
            all_negative_valid, all_negative_status = CriSTool._steady_physical_constraints(
                configured_backend, all_negative_state, nothing)
            @test !all_negative_valid
            @test all_negative_status == :negative_number_density
        end
    end
end
