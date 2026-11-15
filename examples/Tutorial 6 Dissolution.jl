#=
Tutorial 6: signed growth and dissolution.

The same scalar dissolution rate is run with MoM, FiniteVol, and WENO.
The seeded population starts below a constant saturation concentration, so
the crystal growth rate is negative and the solution concentration rises.
For a length-dependent dissolution law, use Tutorial 6's discretised solver
pattern with `ArrheniusLengthDissolution()`; QMOM support is intentionally
limited to scalar signed kinetics.
=#

using CriSTool
using CairoMakie

function main()
    # Dissolution has its own model slot; the solvers add its (negative) rate to
    # the growth rate. Its three SI parameters are the dissolution coefficient,
    # activation energy and order. Growth is zero below saturation, so only
    # dissolution acts here.
    crystal_model = CrystallisationModel(LysozymeSystem(saturation_model = ConstantSolubility(10.0));
        growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 1.0),
        dissolution = ArrheniusDissolution(dissolution_coefficient = 2e-9 / 60, activation_energy = 0.0,
            dissolution_order = 1.5))
    # A narrow 25 µm seed starts below saturation, so crystals dissolve and c rises.
    undersaturated_run = RunConditions(temperature = 293.15, concentration = 5.0,
        crystals = LogNormalInitialCrystals(mass_concentration = 0.17, d43 = 25e-6, geometric_std = 1.05))
    undersaturated_problem = CrystallisationProblem(crystal_model, undersaturated_run, (0.0, 18000.0))
    save_grid = collect(0.0:1800.0:18000.0)

    solvers = [MoM(),
               FiniteVol(meshsize = 80, lmax = 50.0e-6),
               WENO(meshsize = 80, lmax = 50.0e-6)]
    solutions = [solve(undersaturated_problem, solver; saveat = save_grid) for solver in solvers]

    # Characteristic size: the volume median on a mesh, d43 otherwise.
    size_trajectory(solution::CrystallisationSolution{<:MeshPopulation}) = solution.population.d50q
    size_trajectory(solution::CrystallisationSolution) = solution.d43
    for (solver, solution) in zip(solvers, solutions)
        println("$(typeof(solver)): retcode=$(solution.retcode), " *
                "c_final=$(round(solution.concentration[end], sigdigits = 5)), " *
                "size_final=$(round(1e6 * size_trajectory(solution)[end], sigdigits = 5)) μm")
    end

    figure = Figure(size = (900, 650))
    Label(figure[0, :], "Tutorial 6: scalar signed dissolution", fontsize = 16)
    concentration_axis = Axis(figure[1, 1], xlabel = "Time (min)", ylabel = "c")
    size_axis = Axis(figure[2, 1], xlabel = "Time (min)", ylabel = "Characteristic size (μm)")
    colors = (:dodgerblue, :crimson, :seagreen)
    for (solver, solution, color) in zip(solvers, solutions, colors)
        label = string(typeof(solver))
        lines!(concentration_axis, solution.t ./ 60, solution.concentration,
               color = color, label = label)
        lines!(size_axis, solution.t ./ 60, 1e6 .* size_trajectory(solution),
               color = color, label = label)
    end
    axislegend(concentration_axis, position = :rb)
    axislegend(size_axis, position = :rb)
    display(figure)
end

main()
