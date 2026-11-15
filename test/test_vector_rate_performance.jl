@eval begin
    mutable struct VectorRateFallbackLengthDissolution <:
                   CriSTool.AbstractFPLengthDissolutionFunction
        evaluations::Base.RefValue{Int}
    end

    function CriSTool.dissolutionrate_at_length(
        dissolutionfunction::VectorRateFallbackLengthDissolution,
        parameters,
        problem::CriSTool.CrystallisationProblem,
        state,
        time,
        crystal_length::Real)
        dissolutionfunction.evaluations[] += 1
        return -parameters[1] * crystal_length
    end
end

@testset "Vectorized length-rate kernels" begin
    mesh = [0.5e-6, 1.0e-6, 2.0e-6]
    problem = CrystallisationProblem(
        kinetics_growthfunction = growth_empirical_length(Lref = 1e-6),
        saturation_model = ConstantSolubility(10.0),
        initial_concentration = 10.0,
        solver = FiniteVol(meshsize = length(mesh), lmax = 3e-6))
    supersaturated_state = [zeros(length(mesh)); 15.0]
    undersaturated_state = [zeros(length(mesh)); 5.0]

    @testset "Empirical length growth matches hand-computed values" begin
        growthfunction = growth_empirical_length(Lref = 1e-6)
        growth_parameters = [3e-8, 2.0, 0.5, 2.0]
        growth_destination = zeros(length(mesh))

        growth_result = CriSTool.growthrate!(growth_destination, growthfunction,
                                             growth_parameters, problem,
                                             supersaturated_state, 0.0, mesh)
        expected_growth_rates = [1.171875e-8, 1.6875e-8, 3.0e-8]

        @test growth_result === growth_destination
        @test growth_destination ≈ expected_growth_rates rtol = 1e-14
        @test CriSTool.growthrate_at_length(growthfunction, growth_parameters,
                                            problem, supersaturated_state, 0.0,
                                            mesh[2]) ≈ 1.6875e-8 rtol = 1e-14

        near_equilibrium_state = [zeros(length(mesh)); 10.009]
        CriSTool.growthrate!(growth_destination, growthfunction, growth_parameters,
                             problem, near_equilibrium_state, 0.0, mesh)
        @test all(iszero, growth_destination)

        invalid_size_parameters = [3e-8, 2.0, -0.5, 2.0]
        @test_throws DomainError CriSTool.growthrate!(
            growth_destination, growthfunction, invalid_size_parameters, problem,
            supersaturated_state, 0.0, [2.5e-6, 2.6e-6, 2.7e-6])
        @test CriSTool.growthrate!(growth_destination, growthfunction,
                                   invalid_size_parameters, problem,
                                   near_equilibrium_state, 0.0, mesh) ===
              growth_destination
        @test all(iszero, growth_destination)
    end

    @testset "Length dissolution is hoisted in net rates" begin
        growthfunction = growth_empirical_length(Lref = 1e-6)
        growth_parameters = [3e-8, 2.0, 0.5, 2.0]
        dissolutionfunction = growth_dissolution_length(Lref = 1e-6)
        dissolution_parameters = [6e-8, 0.0, 1.0, 0.5, 2.0]
        dissolution_destination = zeros(length(mesh))

        CriSTool.dissolutionrate!(dissolution_destination, dissolutionfunction,
                                  dissolution_parameters, problem,
                                  undersaturated_state, 0.0, mesh)
        expected_dissolution_rates = [-4.6875e-8, -6.75e-8, -1.2e-7]
        @test dissolution_destination ≈ expected_dissolution_rates rtol = 1e-14

        net_destination = zeros(length(mesh))
        net_result = CriSTool.net_growth_rate!(
            net_destination, growthfunction, growth_parameters,
            dissolutionfunction, dissolution_parameters, problem,
            undersaturated_state, 0.0, mesh)
        @test net_result === net_destination
        @test net_destination ≈ expected_dissolution_rates rtol = 1e-14

        invalid_dissolution_parameters = [6e-8, 0.0, 1.0, -0.5, 2.0]
        @test_throws DomainError CriSTool.net_growth_rate!(
            net_destination, growthfunction, growth_parameters,
            dissolutionfunction, invalid_dissolution_parameters, problem,
            undersaturated_state, 0.0, [2.5e-6, 2.6e-6, 2.7e-6])
    end

    @testset "Custom length dissolution keeps scalar fallback" begin
        custom_dissolution = VectorRateFallbackLengthDissolution(Ref(0))
        custom_parameters = [0.03]
        net_destination = zeros(length(mesh))

        CriSTool.net_growth_rate!(
            net_destination, growth_empirical(), [3e-8, 2.0],
            custom_dissolution, custom_parameters, problem,
            undersaturated_state, 0.0, mesh)

        expected_custom_rates = [-1.5e-8, -3.0e-8, -6.0e-8]
        @test net_destination ≈ expected_custom_rates rtol = 1e-14
        @test custom_dissolution.evaluations[] == length(mesh)
    end

    @testset "Length-rate gradients agree across DI backends" begin
        forwarddiff_backend = DI.AutoForwardDiff()
        finitedifference_backend = DI.AutoFiniteDifferences(
            FiniteDifferences.central_fdm(5, 1))
        weights = [1.0, 2.0, 4.0]

        function empirical_length_growth_objective(growth_parameters)
            growth_destination = similar(mesh, eltype(growth_parameters))
            CriSTool.growthrate!(growth_destination,
                                 growth_empirical_length(Lref = 1e-6),
                                 growth_parameters, problem,
                                 supersaturated_state, 0.0, mesh)
            return sum(weights .* growth_destination) * 1e8
        end

        growth_parameters = [3e-8, 1.4, 0.45, 1.3]
        growth_ad_gradient = DI.gradient(empirical_length_growth_objective,
                                         forwarddiff_backend, growth_parameters)
        growth_fd_gradient = DI.gradient(empirical_length_growth_objective,
                                         finitedifference_backend, growth_parameters)
        @test growth_ad_gradient ≈ growth_fd_gradient rtol = 2e-4 atol = 2e-5

        function length_dissolution_objective(dissolution_parameters)
            net_destination = similar(mesh, eltype(dissolution_parameters))
            CriSTool.net_growth_rate!(
                net_destination, growth_empirical(), [3e-8, 2.0],
                growth_dissolution_length(Lref = 1e-6),
                dissolution_parameters, problem, undersaturated_state, 0.0, mesh)
            return sum(weights .* net_destination) * 1e8
        end

        dissolution_parameters = [6e-8, 1200.0, 1.2, 0.4, 1.3]
        dissolution_ad_gradient = DI.gradient(length_dissolution_objective,
                                              forwarddiff_backend,
                                              dissolution_parameters)
        dissolution_fd_gradient = DI.gradient(length_dissolution_objective,
                                              finitedifference_backend,
                                              dissolution_parameters)
        @test dissolution_ad_gradient ≈ dissolution_fd_gradient rtol = 2e-4 atol = 2e-5
    end
end
