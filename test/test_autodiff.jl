# Tests for automatic differentiation compatibility
# Using DifferentiationInterface.jl for backend-agnostic AD testing

import DifferentiationInterface as DI
import ForwardDiff
import FiniteDifferences

_si_test_parameters(parameters) =
    [parameters[1], parameters[2] * 1e-3,
     clamp(parameters[3], 0.0, 5.0) * (1e-9 / 60), parameters[4]]

@testset "Automatic Differentiation" begin

    # Define AD backends
    forwarddiff_backend = DI.AutoForwardDiff()
    finitediff_backend = DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1))

    @testset "ForwardDiff Backend - MoM Solver" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        # Define objective function: final concentration
        function objective_mom(params)
            _,
            solution = runsimulation(_si_test_parameters(params),
                                     nucl_func,
                                     grow_func,
                                     initial_conc;
                                     save_idx = 0:3600.0:14400.0,
                                     solver = MoM())
            return solution.concentration[end]
        end

        # Test ForwardDiff gradient
        grad = DI.gradient(objective_mom, forwarddiff_backend, base_params)
        finite_difference_grad = DI.gradient(objective_mom, finitediff_backend, base_params)

        @test length(grad) == 4
        @test grad ≈ finite_difference_grad rtol = 0.05 atol = 1e-8
    end

    @testset "ForwardDiff Backend - FiniteVol Solver" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()
        agg_func = noaggregation()
        br_func = nobreakage()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function objective_fv(params)
            _,
            solution = runsimulation(_si_test_parameters(params),
                                     nucl_func,
                                     grow_func,
                                     agg_func,
                                     br_func,
                                     initial_conc;
                                     save_idx = collect(0:3600.0:14400.0),
                                     solver = FiniteVol(meshsize = 50, lmax = 50e-6))
            return solution.concentration[end]
        end

        grad = DI.gradient(objective_fv, forwarddiff_backend, base_params)
        finite_difference_grad = DI.gradient(objective_fv, finitediff_backend, base_params)

        @test length(grad) == 4
        @test grad ≈ finite_difference_grad rtol = 0.15 atol = 1e-8
    end

    @testset "Gradient Numerical Verification - MoM" begin
        # Verify ForwardDiff gradients match FiniteDifferences
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function objective_verify(params)
            _,
            solution = runsimulation(_si_test_parameters(params),
                                     nucl_func,
                                     grow_func,
                                     initial_conc;
                                     save_idx = 0:7200.0:14400.0,
                                     solver = MoM())
            return solution.concentration[end]
        end

        ad_grad = DI.gradient(objective_verify, forwarddiff_backend, base_params)
        fd_grad = DI.gradient(objective_verify, finitediff_backend, base_params)

        # Compare gradients with relative tolerance
        for i in 1:4
            if abs(ad_grad[i]) > 1e-8
                rel_error = abs(ad_grad[i] - fd_grad[i]) / abs(ad_grad[i])
                @test rel_error < 0.05  # 5% relative tolerance
            else
                @test abs(fd_grad[i]) < 1e-6
            end
        end
    end

    @testset "Gradient Numerical Verification - FiniteVol" begin
        # Verify ForwardDiff gradients match FiniteDifferences for FV solver
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()
        agg_func = noaggregation()
        br_func = nobreakage()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function objective_verify_fv(params)
            _,
            solution = runsimulation(_si_test_parameters(params),
                                     nucl_func,
                                     grow_func,
                                     agg_func,
                                     br_func,
                                     initial_conc;
                                     save_idx = collect(0:7200.0:14400.0),
                                     solver = FiniteVol(meshsize = 50, lmax = 50e-6))
            return solution.concentration[end]
        end

        ad_grad = DI.gradient(objective_verify_fv, forwarddiff_backend, base_params)
        fd_grad = DI.gradient(objective_verify_fv, finitediff_backend, base_params)

        # Compare gradients with relative tolerance
        for i in 1:4
            if abs(ad_grad[i]) > 1e-8
                rel_error = abs(ad_grad[i] - fd_grad[i]) / abs(ad_grad[i])
                @test rel_error < 0.15  # 15% tolerance for FV (more complex solver)
            else
                @test abs(fd_grad[i]) < 1e-6
            end
        end
    end

    @testset "Jacobian Computation - Multiple Outputs" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function multi_objective(params)
            _,
            solution = runsimulation(_si_test_parameters(params),
                                     nucl_func,
                                     grow_func,
                                     initial_conc;
                                     save_idx = 0:3600.0:14400.0,
                                     solver = MoM())
            return solution.concentration[1:3]
        end

        jac = DI.jacobian(multi_objective, forwarddiff_backend, base_params)
        finite_difference_jac = DI.jacobian(multi_objective, finitediff_backend, base_params)

        @test size(jac) == (3, 4)
        @test jac ≈ finite_difference_jac rtol = 0.05 atol = 1e-8
    end

    @testset "d50 Objective Gradient - FiniteVol" begin
        # Test gradient of particle size objective
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()
        agg_func = noaggregation()
        br_func = nobreakage()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function objective_d50(params)
            _,
            solution = runsimulation(_si_test_parameters(params),
                                     nucl_func,
                                     grow_func,
                                     agg_func,
                                     br_func,
                                     initial_conc;
                                     save_idx = collect(0:3600.0:14400.0),
                                     solver = FiniteVol(meshsize = 50, lmax = 50e-6))
            return solution.d50q[end]
        end

        grad = DI.gradient(objective_d50, forwarddiff_backend, base_params)
        finite_difference_grad = DI.gradient(objective_d50, finitediff_backend, base_params)

        @test length(grad) == 4
        @test grad ≈ finite_difference_grad rtol = 0.15 atol = 1e-8
    end

end
