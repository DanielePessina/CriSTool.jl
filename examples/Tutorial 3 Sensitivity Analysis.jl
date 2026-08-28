"""
Tutorial 3: Sensitivity analysis.

Two complementary global sensitivity workflows on the terminal concentration:
  1. Sobol indices (variance-decomposition) via `gsa(...; Sobol2007 estimator)`.
  2. DGSM (derivative-based) via `gsa(...; DGSM())`.

Both wrap the same `runsimulation` forward map. Bounds are 0.5×–1.5× around
a baseline parameter set.
"""

using CriSTool
using GlobalSensitivity, QuasiMonteCarlo, Distributions
using Random

function main()
    Random.seed!(0)

    # Baseline kinetics + flat parameter vector [Aj, γ, Ag, g].
    nucl, gr   = nucl_CNT(), growth_empirical()
    agg, br    = noaggregation(), nobreakage()
    solver     = MoM()
    base_p     = [38.0, 0.7, 1.0, 3.0]
    save_grid  = 0.0:6.0:360.0    # minutes
    C0         = 18.0

    # Forward map: parameters → terminal concentration. GSA samples pass in
    # flat parameter vectors; we let runsimulation wrap them internally.
    sim_end = function (p)
        _, sol = runsimulation(p; nucl=nucl, gr=gr, agg=agg, br=br,
                                initial_concentration=C0, solver=solver,
                                save_idx=save_grid)
        return sol.concentration[end]
    end

    # 1. Sobol indices on a 256-sample QMC design.
    lb = 0.5 .* base_p
    ub = 1.5 .* base_p
    A, B = QuasiMonteCarlo.generate_design_matrices(256, lb, ub, SobolSample())
    sobol = gsa(sim_end,
                Sobol(order = [0, 1, 2], nboot = 32, conf_level = 0.95),
                A, B; Ei_estimator = :Sobol2007)
    println("Sobol first-order indices (S1):  ", sobol.S1)
    println("Sobol total-order indices (ST):  ", sobol.ST)

    # 2. DGSM on the same bounds.
    priors = [Uniform(lb[i], ub[i]) for i in eachindex(lb)]
    dgsm   = gsa(sim_end, DGSM(), priors; samples = 256)
    println("DGSM indices: ", dgsm)
end

main()
