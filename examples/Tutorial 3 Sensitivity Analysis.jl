#=
Tutorial 3: Sensitivity analysis.

Two complementary global sensitivity workflows on the terminal concentration:
  1. Sobol indices (variance-decomposition) via `gsa(...; Sobol2007 estimator)`.
  2. DGSM (derivative-based) via `gsa(...; DGSM())`.

Both wrap the same `solve(remake(problem; p = parameters), MoM())` forward map. Bounds are 0.5×–1.5× around
a baseline parameter set.
=#

using CriSTool
using GlobalSensitivity, QuasiMonteCarlo, Distributions
using Random

function main()
    Random.seed!(0)

    # Baseline kinetics configured once; the flat SI parameter vector follows
    # the composite paramaxis order [ln prefactor, surface energy, kg, g].
    base_p     = [38.0, 0.0007, 1e-9 / 60, 3.0]
    baseline_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = base_p[1], surface_energy = base_p[2]),
        growth = EmpiricalGrowth(growth_coefficient = base_p[3], growth_order = base_p[4]))
    problem    = CrystallisationProblem(baseline_model,
        RunConditions(temperature = 293.15, concentration = 18.0), (0.0, 21600.0))
    save_grid  = 0.0:360.0:21600.0    # seconds

    # Forward map: parameters → terminal concentration. GSA samples pass in
    # flat parameter vectors, which replace the kinetic values of `problem`.
    sim_end = function (p)
        sol = solve(remake(problem; p), MoM(); saveat = save_grid)
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
