#=Tutorial 2: named parameter selection, measured initial conditions and a reusable fitted model.=#
using CriSTool, Metaheuristics
include(joinpath(@__DIR__, "model_helpers.jl"))

function main()
    crystal_model = tutorial_model()
    measured_runs = load_measurements(joinpath(@__DIR__, "fake-experimental-dataset.csv"))
    fit_specification = OptimisationSpec(bounds = (
        nucleation = (ln_nucleation_prefactor = (20.0, 50.0), surface_energy = (0.0001, 0.002)),
        growth = (growth_coefficient = (1e-12, 1e-9), growth_order = (1.0, 4.0))), loss = logMLE())
    prepared_fit = prepare_fit(crystal_model, measured_runs, fit_specification; solver = MoM())
    fitted_result = fit(prepared_fit; algorithm = DE(N = 16,
        options = Options(iterations = 4, seed = 42, parallel_evaluation = false)))
    println("Named estimates: ", fitted_result.parameters)
    println("Observable objectives: ", fitted_result.objectives)
    # For a larger study increase particles/iterations and inspect backend convergence.
    fitted_result
end
main()
