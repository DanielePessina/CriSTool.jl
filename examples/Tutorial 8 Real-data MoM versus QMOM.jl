#=Tutorial 8: fit the same named kinetics to real measurements with MoM and QMOM.=#
using CriSTool, Metaheuristics

function main()
    crystal_model = CrystallisationModel(LysozymeSystem();
        nucleation = nucl_CNT(ln_nucleation_prefactor = 38.0, surface_energy = 0.0006),
        growth = growth_empirical(growth_coefficient = 1e-9 / 60, growth_order = 3.0))
    real_runs = load_measurements(joinpath(pkgdir(CriSTool), "test", "fixtures", "real-experimental-dataset.csv"))
    selected_fit = OptimisationSpec(bounds = (growth = (growth_order = (1.0, 4.0),),),
        algorithm = DE(N = 12, options = Options(iterations = 3, seed = 42)))
    for selected_solver in (MoM(), QMOM(nquadrature = 3))
        fit_result = fit(prepare_fit(crystal_model, real_runs, selected_fit; solver = selected_solver))
        println(nameof(typeof(selected_solver)), ": ", fit_result.parameters, " objectives = ", fit_result.objectives)
    end
end
main()
