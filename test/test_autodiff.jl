# Tests for automatic differentiation compatibility
# Using DifferentiationInterface.jl for backend-agnostic AD testing

import DifferentiationInterface as DI
import ForwardDiff
import FiniteDifferences

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
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     initial_conc;
                                     save_idx = 0:60.0:240.0,
                                     solver = MoM())
            return solution.concentration[end]
        end

        # Test ForwardDiff gradient
        grad = DI.gradient(objective_mom, forwarddiff_backend, base_params)

        @test length(grad) == 4
        @test all(isfinite.(grad))
        @test !all(grad .≈ 0)  # Parameters should affect output
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
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     agg_func,
                                     br_func,
                                     initial_conc;
                                     save_idx = collect(0:60.0:240.0),
                                     solver = FiniteVol(meshsize = 50, lmax = 50e-6))
            return solution.concentration[end]
        end

        grad = DI.gradient(objective_fv, forwarddiff_backend, base_params)

        @test length(grad) == 4
        @test all(isfinite.(grad))
        @test !all(grad .≈ 0)
    end

    @testset "FiniteDifferences Backend - MoM Solver" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function objective_mom_fd(params)
            _,
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     initial_conc;
                                     save_idx = 0:60.0:240.0,
                                     solver = MoM())
            return solution.concentration[end]
        end

        # Test FiniteDifferences gradient
        grad = DI.gradient(objective_mom_fd, finitediff_backend, base_params)

        @test length(grad) == 4
        @test all(isfinite.(grad))
        @test !all(grad .≈ 0)
    end

    @testset "FiniteDifferences Backend - FiniteVol Solver" begin
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()
        agg_func = noaggregation()
        br_func = nobreakage()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function objective_fv_fd(params)
            _,
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     agg_func,
                                     br_func,
                                     initial_conc;
                                     save_idx = collect(0:60.0:240.0),
                                     solver = FiniteVol(meshsize = 50, lmax = 50e-6))
            return solution.concentration[end]
        end

        grad = DI.gradient(objective_fv_fd, finitediff_backend, base_params)

        @test length(grad) == 4
        @test all(isfinite.(grad))
    end

    @testset "Gradient Numerical Verification - MoM" begin
        # Verify ForwardDiff gradients match FiniteDifferences
        nucl_func = nucl_CNT()
        grow_func = growth_empirical()

        base_params = [38.0, 0.7, 1.0, 3.0]
        initial_conc = 18.0

        function objective_verify(params)
            _,
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     initial_conc;
                                     save_idx = 0:120.0:240.0,
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
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     agg_func,
                                     br_func,
                                     initial_conc;
                                     save_idx = collect(0:120.0:240.0),
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
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     initial_conc;
                                     save_idx = 0:60.0:240.0,
                                     solver = MoM())
            return solution.concentration[1:3]
        end

        jac = DI.jacobian(multi_objective, forwarddiff_backend, base_params)

        @test size(jac) == (3, 4)
        @test all(isfinite.(jac))
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
            solution = runsimulation(params,
                                     nucl_func,
                                     grow_func,
                                     agg_func,
                                     br_func,
                                     initial_conc;
                                     save_idx = collect(0:60.0:240.0),
                                     solver = FiniteVol(meshsize = 50, lmax = 50e-6))
            return solution.d50q[end]
        end

        grad = DI.gradient(objective_d50, forwarddiff_backend, base_params)

        @test length(grad) == 4
        @test all(isfinite.(grad))
    end

end
