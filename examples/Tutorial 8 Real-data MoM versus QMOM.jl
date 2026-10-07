#=Tutorial 8: fit the same named kinetics to real measurements with MoM and QMOM.=#
using CriSTool, Metaheuristics
include(joinpath(@__DIR__, "model_helpers.jl"))

function main()
    crystal_model = tutorial_model()
    real_runs = load_measurements(joinpath(pkgdir(CriSTool), "test", "fixtures", "real-experimental-dataset.csv"))
    selected_fit = OptimisationSpec(bounds = (growth = (growth_order = (1.0, 4.0),),), loss = logMLE())
    for selected_solver in (MoM(), QMOM(nquadrature = 3))
        prepared_fit = prepare_fit(crystal_model, real_runs, selected_fit; solver = selected_solver)
        fit_result = fit(prepared_fit; algorithm = DE(N = 12,
            options = Options(iterations = 3, seed = 42, parallel_evaluation = false)))
        println(typeof(selected_solver), ": ", fit_result.parameters, " objectives = ", fit_result.objectives)
    end
end
main()
