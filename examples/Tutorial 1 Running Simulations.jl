#=Tutorial 1: reuse one named model across temperature profiles and solvers.=#
using CriSTool
include(joinpath(@__DIR__, "model_helpers.jl"))

function main()
    crystal_model = tutorial_model()
    save_times = 0.0:600.0:21600.0
    for numerical_solver in (MoM(), QMOM(nquadrature = 3), FiniteVol(meshsize = 100, lmax = 100e-6))
        configured_run = CrystallisationProblem(crystal_model;
            initial_conditions = (concentration = 18.0,),
            temperature = ConstantTemperature(293.15), solver = numerical_solver)
        trajectory = simulate(configured_run; saveat = save_times)
        @assert trajectory.success
        println(typeof(numerical_solver), ": C(final) = ", trajectory.concentration[end],
            " kg/m³; d43(final) = ", trajectory.d43[end], " m")
    end
end
main()
