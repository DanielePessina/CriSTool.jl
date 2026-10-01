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
        end
    end
end
