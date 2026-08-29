# Gradient harness: cross-backend gradient agreement via DifferentiationInterface.
# FiniteDifferences is the numerical reference and ForwardDiff is the supported
# automatic-differentiation backend for the current release. Reverse Enzyme
# differentiation through OrdinaryDiffEq is deferred because of upstream Julia
# 1.12 ecosystem blockers; see AUDIT_v1.0.md.

import DifferentiationInterface as DI
import ForwardDiff
import FiniteDifferences

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
        S, T, loading = 3.2, 290.15, 0.0
        nd = [1e10, 1e11, 1e12, 1e13, 1e14]
        state = [nd; S * saturation_concentration(system, 0.0)]

        f_nucl(p) = CriSTool.nucleationrate(nucl, p, system, state, 0.0)
        g_fd = DI.gradient(f_nucl, fd_backend, [38.0, 0.6])
        g_fwd = DI.gradient(f_nucl, fwd_backend, [38.0, 0.6])
        assert_gradient_agreement("nucleationrate (nucl_CNT)", g_fd, g_fwd; rtol = 1e-4)

        f_gr(p) = CriSTool.growthrate(gr, p, system, state, 0.0)
        g_fd_gr = DI.gradient(f_gr, fd_backend, [1.0, 3.0])
        g_fwd_gr = DI.gradient(f_gr, fwd_backend, [1.0, 3.0])
        assert_gradient_agreement("growthrate (growth_empirical)", g_fd_gr, g_fwd_gr;
                                  rtol = 1e-4)

    end

    @testset "MoM solution observables" begin
        nucl = nucl_CNT()
        gr = growth_empirical()
        save_idx = [0.0, 30.0, 60.0, 120.0, 180.0, 270.0]

        function run_mom(params)
            _, sol = runsimulation(params; nucl = nucl, gr = gr,
                                   agg = noaggregation(), br = nobreakage(),
                                   initial_concentration = 14.67,
                                   save_idx = save_idx,
                                   solver = MoM(),
                                   loading = 0.0,
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
        fixture = joinpath(@__DIR__, "fixtures", "real-experimental-dataset.xlsx")
        ms = load_experiments(fixture, "Unseeded_PE", 0.0)
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
        save_idx = collect(0.0:60.0:240.0)

        f_fv(params) = begin
            _, sol = runsimulation(params; nucl = nucl, gr = gr,
                                   agg = noaggregation(), br = nobreakage(),
                                   initial_concentration = 14.67,
                                   save_idx = save_idx,
                                   solver = FiniteVol(meshsize = 50, lmax = 50e-6),
                                   loading = 0.0,
                                   temp_profile = CriSTool.ConstantTemperature(290.15))
            return sol.concentration[end]
        end

        g_fd = DI.gradient(f_fv, fd_backend, base_params)
        g_fwd = DI.gradient(f_fv, fwd_backend, base_params)
        assert_gradient_agreement("FV final concentration", g_fd, g_fwd; rtol = 0.15)
    end
end
