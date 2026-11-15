#=Tutorial 8: fit the same named kinetics to real measurements with MoM and QMOM.=#
using CriSTool, Metaheuristics

function main()
    crystal_model = CrystallisationModel(LysozymeSystem();
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    real_runs = load_measurements(joinpath(pkgdir(CriSTool), "test", "fixtures", "real-experimental-dataset.csv"))
    mom_problem = EstimationProblem(crystal_model, real_runs; bounds = (growth = (growth_order = (1.0, 4.0),),))
    for selected_solver in (MoM(), QMOM(nquadrature = 3))
        # remake rebuilds the same study under another solver.
        fit_result = solve(remake(mom_problem; solver = selected_solver),
            DE(N = 12, options = Options(iterations = 3, seed = 42)))
        println(nameof(typeof(selected_solver)), ": ", fit_result.parameters, " objectives = ", fit_result.objectives)
    end
end
main()
