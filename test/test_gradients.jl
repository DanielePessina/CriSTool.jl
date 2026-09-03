# Gradient harness: cross-backend gradient agreement via DifferentiationInterface.
# FiniteDifferences is the numerical reference and ForwardDiff is the supported
# automatic-differentiation backend for the current release. Reverse Enzyme
# differentiation through OrdinaryDiffEq is deferred because of upstream Julia
# 1.12 ecosystem blockers; see AUDIT_v1.0.md.

import DifferentiationInterface as DI
import ForwardDiff
import FiniteDifferences

const TEST_SI_GROWTH_SCALE = 1e-9 / 60
_si_growth_parameters(parameters) = [parameters[1] * TEST_SI_GROWTH_SCALE,
                                     parameters[2]]
_si_dissolution_parameters(parameters) = [parameters[1] * TEST_SI_GROWTH_SCALE,
                                          parameters[2], parameters[3]]
_si_full_growth_dissolution_parameters(parameters) =
    [_si_growth_parameters(parameters[1:2]);
     _si_dissolution_parameters(parameters[3:5])]

@testset "Gradient harness (DI cross-backend)" begin
    fd_backend = DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1))
    fwd_backend = DI.AutoForwardDiff()

    base_params = [38.0, 0.6, 1.0, 3.0]

    function assert_gradient_agreement(name, g1, g2; rtol)
        for i in eachindex(g1)
            scale = max(abs(g1[i]), abs(g2[i]))
            if scale > 1e-8
                @test abs(g1[i] - g2[i]) <= rtol * scale
            else
                @test abs(g1[i] - g2[i]) <= 1e-6
            end
        end
    end

@testset "Kinetics rates" begin
        nucl = nucl_CNT()
        gr = growth_empirical()
        system = CrystallisationProblem(; kinetics_nucleationfunction = nucl,
                                        kinetics_growthfunction = gr,
                                        solver = MoM())
        S, T = 3.2, 290.15
        nd = [1e10, 1e11, 1e12, 1e13, 1e14]
        state = [nd; S * saturation_concentration(system, 0.0)]

        f_nucl(p) = CriSTool.nucleationrate(nucl, [p[1], p[2] * 1e-3], system, state, 0.0)
        g_fd = DI.gradient(f_nucl, fd_backend, [38.0, 0.6])
        g_fwd = DI.gradient(f_nucl, fwd_backend, [38.0, 0.6])
        assert_gradient_agreement("nucleationrate (nucl_CNT)", g_fd, g_fwd; rtol = 1e-4)

        f_gr(p) = CriSTool.growthrate(gr, p, system, state, 0.0)
        f_gr_scaled(parameters) = f_gr(_si_growth_parameters(parameters))
        g_fd_gr = DI.gradient(f_gr_scaled, fd_backend, [1.0, 3.0])
        g_fwd_gr = DI.gradient(f_gr_scaled, fwd_backend, [1.0, 3.0])
        assert_gradient_agreement("growthrate (growth_empirical)", g_fd_gr, g_fwd_gr;
                                  rtol = 1e-4)

    end

    @testset "MoM solution observables" begin
        nucl = nucl_CNT()
        gr = growth_empirical()
        save_idx = [0.0, 1800.0, 3600.0, 7200.0, 10800.0, 16200.0]

        function run_mom(params)
            _, sol = runsimulation(_si_test_parameters(params); nucl = nucl, gr = gr,
                                   agg = noaggregation(), br = nobreakage(),
                                   initial_concentration = 14.67,
                                   save_idx = save_idx,
                                   solver = MoM(),
                                   temp_profile = CriSTool.ConstantTemperature(290.15))
            return sol
        end

        f_conc(params) = run_mom(params).concentration[end]
        g_fd = DI.gradient(f_conc, fd_backend, base_params)
        g_fwd = DI.gradient(f_conc, fwd_backend, base_params)
        assert_gradient_agreement("MoM final concentration", g_fd, g_fwd; rtol = 0.05)
        f_d43(params) = run_mom(params).d43[end]
        g_fd43 = DI.gradient(f_d43, fd_backend, base_params)
        g_fwd43 = DI.gradient(f_d43, fwd_backend, base_params)
        assert_gradient_agreement("MoM final d43", g_fd43, g_fwd43; rtol = 0.05)
    end

    @testset "Full logMLE loss (gold fixture)" begin
        fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.csv")
        ms = load_measurements(fixture)
        problem = CrystallisationProblem(;
            kinetics_nucleationfunction = nucl_CNT(),
            kinetics_growthfunction = growth_empirical(),
            kinetics_aggregationfunction = noaggregation(),
            kinetics_breakagefunction = nobreakage(),
            solver = MoM())
        f_loss(params) = loss(logMLE(), problem, params, ms[1:2])
        g_fd = DI.gradient(f_loss, fd_backend, base_params)
        g_fwd = DI.gradient(f_loss, fwd_backend, base_params)
        assert_gradient_agreement("logMLE loss (2 experiments)", g_fd, g_fwd; rtol = 0.1)
    end

    @testset "FiniteVol final concentration" begin
        nucl = nucl_CNT()
        gr = growth_empirical()
        save_idx = collect(0.0:3600.0:14400.0)

        f_fv(params) = begin
            _, sol = runsimulation(params; nucl = nucl, gr = gr,
                                   agg = noaggregation(), br = nobreakage(),
                                   initial_concentration = 14.67,
                                   save_idx = save_idx,
                                   solver = FiniteVol(meshsize = 50, lmax = 50e-6),
                                   temp_profile = CriSTool.ConstantTemperature(290.15))
            return sol.concentration[end]
        end

        g_fd = DI.gradient(f_fv, fd_backend, base_params)
        g_fwd = DI.gradient(f_fv, fwd_backend, base_params)
        assert_gradient_agreement("FV final concentration", g_fd, g_fwd; rtol = 0.15)
    end
end

@testset "Independent dissolution and QMOM gradient verification" begin
    fd_backend = DI.AutoFiniteDifferences(FiniteDifferences.central_fdm(5, 1))
    fwd_backend = DI.AutoForwardDiff()

    fixed_no_nucleation = CriSTool.nucl_empirical_fixed(log10_nucleation_prefactor = -Inf, nucleation_order = 1.0)
    dissolution_model = growth_dissolution()
    growth_model = growth_empirical()
    shared_problem = CrystallisationProblem(;
        kinetics_nucleationfunction = fixed_no_nucleation,
        kinetics_growthfunction = growth_model,
        kinetics_dissolutionfunction = dissolution_model,
        parameterset_nucleation = Float64[],
        parameterset_growth = [TEST_SI_GROWTH_SCALE, 2.0],
        parameterset_dissolution = [2e-9 / 60, 0.0, 1.5],
        saturation_model = ConstantSolubility(10.0),
        initial_concentration = 5.0,
        solver = MoM())
    seeded_moments = [1.0e12, 1.0e6, 1.0, 1.0e-6, 1.0e-12]
    qmom_seeded_moments = vcat(seeded_moments, 1.0e-18)
    seeded_state = vcat(seeded_moments, 5.0)

    @testset "Independent net rate" begin
        rate_objective(parameters) = net_growth_rate(
            growth_model, _si_growth_parameters(parameters[1:2]),
            dissolution_model, _si_dissolution_parameters(parameters[3:5]),
            shared_problem, seeded_state, 0.0)
        fd_gradient = DI.gradient(rate_objective, fd_backend,
                                  [1.0, 2.0, 2.0, 0.0, 1.5])
        fwd_gradient = DI.gradient(rate_objective, fwd_backend,
                                   [1.0, 2.0, 2.0, 0.0, 1.5])
        @test fwd_gradient ≈ fd_gradient rtol = 1e-4 atol = 1e-10
    end

    @testset "MoM independent dissolution simulation" begin
        simulation_objective(parameters) = begin
            _, solution = runsimulation(_si_full_growth_dissolution_parameters(parameters);
                nucl = fixed_no_nucleation,
                gr = growth_model,
                diss = dissolution_model,
                agg = noaggregation(),
                br = nobreakage(),
                solver = MoM(),
                initial_concentration = 5.0,
                initial_state = seeded_state,
                saturation_model = ConstantSolubility(10.0),
                save_idx = [0.0, 6.0])
            solution.concentration[end]
        end
        fd_gradient = DI.gradient(simulation_objective, fd_backend,
                                  [1.0, 2.0, 2.0, 0.0, 1.5])
        fwd_gradient = DI.gradient(simulation_objective, fwd_backend,
                                   [1.0, 2.0, 2.0, 0.0, 1.5])
        @test fwd_gradient ≈ fd_gradient rtol = 0.15 atol = 1e-9
    end

    @testset "QMOM simulation and binary source gradients" begin
        qmom_objective(parameters) = begin
            _, solution = runsimulation(_si_growth_parameters(parameters);
                nucl = fixed_no_nucleation,
                gr = growth_model,
                diss = nodissolution(),
                agg = noaggregation(),
                br = nobreakage(),
                solver = QMOM(nquadrature = 3),
                initial_concentration = 20.0,
                initial_state = [1.0e12, 1.0e6, 1.0, 1.0e-6, 1.0e-12, 1.0e-18, 20.0],
                saturation_model = ConstantSolubility(10.0),
                save_idx = [0.0, 6.0])
            solution.d43[end]
        end
        qmom_fd = DI.gradient(qmom_objective, fd_backend, [1.0, 2.0])
        qmom_fwd = DI.gradient(qmom_objective, fwd_backend, [1.0, 2.0])
        @test qmom_fwd ≈ qmom_fd rtol = 0.2 atol = 1e-9

        rule = QMOMQuadrature([0.5e-6, 1.5e-6], [2.0, 3.0])
        aggregation_objective(parameters) = aggregation_moment_source(
            aggr_scalar(), parameters, rule, 3)[1]
        breakage_objective(parameters) = breakage_moment_source(
            breakage_empirical(), parameters, rule, 3)[1]
        @test DI.gradient(aggregation_objective, fwd_backend, [0.0]) ≈
              DI.gradient(aggregation_objective, fd_backend, [0.0]) rtol = 1e-4
        @test DI.gradient(breakage_objective, fwd_backend, [0.0, 1.0]) ≈
              DI.gradient(breakage_objective, fd_backend, [0.0, 1.0]) rtol = 1e-4
    end
end
