"""
Tutorial 6: signed growth and dissolution.

The same scalar dissolution rate is run with MoM, FiniteVol, and WENO.
The seeded population starts below a constant saturation concentration, so
the crystal growth rate is negative and the solution concentration rises.
For a length-dependent dissolution law, use Tutorial 6's discretised solver
pattern with `growth_dissolution_length()`; QMOM support is intentionally
limited to scalar signed kinetics.
"""

using CriSTool
using CairoMakie

function seeded_state(solver)
    seed_number = 1.0e10
    seed_length = 25.0e-6
    if solver isa MoM
        # One narrow atomic population, represented by M₀:M₄, followed by c.
        return [seed_number .* seed_length .^ (0:4); 5.0]
    end

    density = zeros(solver.meshsize)
    seed_index = clamp(round(Int, solver.meshsize * 0.5), 1, solver.meshsize)
    density[seed_index] = seed_number / solver.cell_dL
    return [density; 5.0]
end

function main()
    # `nucl_empirical_fixed` has no fitted parameters and returns no
    # nucleation for this example; the three growth parameters are [Ad, Ead, d].
    nucl = CriSTool.nucl_empirical_fixed([0.0, 1.0])
    growth = growth_dissolution()
    params = [2.0, 0.0, 1.5]
    saturation = ConstantSolubility(10.0)
    save_grid = collect(0.0:30.0:300.0)

    solvers = [MoM(),
               FiniteVol(meshsize = 80, lmax = 50.0e-6),
               WENO(meshsize = 80, lmax = 50.0e-6)]
    solutions = map(solvers) do solver
        _, solution = runsimulation(
            params;
            nucl = nucl,
            gr = growth,
            agg = noaggregation(),
            br = nobreakage(),
            solver = solver,
            initial_concentration = 5.0,
            initial_state = seeded_state(solver),
            saturation_model = saturation,
            save_idx = save_grid,
        )
        solution
    end

    for (solver, solution) in zip(solvers, solutions)
        println("$(typeof(solver)): success=$(solution.success), " *
                "c_final=$(round(solution.concentration[end], sigdigits = 5)), " *
                "size_final=$(round(get_characteristic_size(solution), sigdigits = 5)) μm")
    end

    figure = Figure(size = (900, 650))
    Label(figure[0, :], "Tutorial 6: scalar signed dissolution", fontsize = 16)
    concentration_axis = Axis(figure[1, 1], xlabel = "Time (min)", ylabel = "c")
    size_axis = Axis(figure[2, 1], xlabel = "Time (min)", ylabel = "Characteristic size (μm)")
    colors = (:dodgerblue, :crimson, :seagreen)
    size_trajectory(solution) = solution isa CrystallisationMoMSolution ||
                               solution isa CrystallisationQMOMSolution ?
                               solution.d43 : solution.d50q
    for (solver, solution, color) in zip(solvers, solutions, colors)
        label = string(typeof(solver))
        lines!(concentration_axis, solution.time, solution.concentration,
               color = color, label = label)
        lines!(size_axis, solution.time, size_trajectory(solution),
               color = color, label = label)
    end
    axislegend(concentration_axis, position = :rb)
    axislegend(size_axis, position = :rb)
    display(figure)
end

main()
