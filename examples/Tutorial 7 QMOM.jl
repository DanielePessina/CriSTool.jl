"""
Tutorial 7: a three-node Quadrature Method of Moments (QMOM) simulation.

QMOM evolves raw moments and reconstructs a Gaussian quadrature at each saved
time. The quadrature is useful for evaluating size-dependent closures while
the public observables remain the moment-derived d32 and d43 metrics.
"""

using CriSTool
using CairoMakie

function main()
    params = [38.0, 0.6, 1.0, 3.0]
    solver = QMOM(nquadrature = 3)
    save_grid = collect(0.0:30.0:300.0)

    _, solution = runsimulation(
        params;
        nucl = nucl_CNT(),
        gr = growth_empirical(),
        agg = noaggregation(),
        br = nobreakage(),
        solver = solver,
        initial_concentration = 18.0,
        save_idx = save_grid,
    )

    final_rule = quadrature(solution, length(solution.time))
    println("QMOM success: ", solution.success)
    println("Tracked raw moments: ", size(solution.moments, 1))
    println("Final d43: ", round(solution.d43[end], sigdigits = 5), " μm")
    println("Final nodes (μm): ", round.(final_rule.nodes .* 1.0e6, sigdigits = 5))
    println("Final weights: ", round.(final_rule.weights, sigdigits = 5))

    figure = Figure(size = (900, 650))
    Label(figure[0, :], "Tutorial 7: three-node QMOM", fontsize = 16)
    concentration_axis = Axis(figure[1, 1], xlabel = "Time (min)", ylabel = "c")
    size_axis = Axis(figure[2, 1], xlabel = "Time (min)", ylabel = "Size (μm)")
    lines!(concentration_axis, solution.time, solution.concentration,
           color = :dodgerblue, label = "concentration")
    lines!(size_axis, solution.time, solution.d43,
           color = :crimson, label = "d43")
    for node_index in axes(solution.quadrature_nodes, 1)
        lines!(size_axis, solution.time,
               solution.quadrature_nodes[node_index, :] .* 1.0e6,
               linestyle = :dash, label = "node $node_index")
    end
    axislegend(size_axis, position = :rb)
    display(figure)
end

main()
