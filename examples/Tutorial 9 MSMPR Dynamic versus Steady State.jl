"""
Tutorial 9: MSMPR dynamic relaxation versus an explicit steady-state solve.

A fixed-flow MSMPR with a crystal-bearing feed and no growth or nucleation
relaxes exponentially toward a known steady state. This script compares two
ways to obtain that operating point:

1. *Dynamic* integration of the full transient equations:

       V(t)   = V₀                              (fixed volume)
       C(t)   = C_f + (C₀ − C_f) exp(−t/τ), τ = V/Q
       M₀(t)  = M₀_feed + (M₀(0) − M₀_feed) exp(−t/τ)

   The final dynamic point approaches the steady state but the transient runner
   stops at the requested horizon — it does not prove equilibrium.

2. `solve_steadystate(problem)`, the explicit steady-state route for a constant
   autonomous MSMPR. It relaxes the dynamics until the scaled full-state RHS
   residual is within tolerance and returns a `CrystallisationSteadyStateSolution`
   with one value per observable plus product flows, the scaled residual, and
   physical / mass-balance diagnostics.

The steady state is the feed condition: concentration `C_f` and the feed
crystal load `M₀_feed` (here one-tenth of the tank seed mass, so
`M₀_feed = M₀(0)/10`). Time-dependent inlet profiles run dynamically but are
not steady inputs.

Run with:  julia --project=examples "examples/Tutorial 9 MSMPR Dynamic versus Steady State.jl"
"""

using CriSTool
using CairoMakie

function main()
    # No nucleation; growth fixed to zero. Fixed kinetics take no parameters.
    nucl = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                                nucleation_order = 1.0)
    gr   = growth_empirical_fixed(growth_coefficient = 0.0, growth_order = 1.0)

    tank_seed = LogNormalInitialCrystals(mass_concentration = 0.5, d43 = 20e-6,
                                         geometric_std = 1.05)
    feed_seed = LogNormalInitialCrystals(mass_concentration = 0.05, d43 = 20e-6,
                                         geometric_std = 1.05)
    operation = MSMPROperation(volume = 2.0, inflow = 0.5,
                               feed = CrystallisationFeed(concentration = 2.0,
                                                          crystals = feed_seed))
    save_grid = collect(0.0:1800.0:21600.0)          # 6 h in seconds

    problem, solution = runsimulation(
        Float64[];                              # fixed kinetics take no parameters
        nucl = nucl, gr = gr, agg = noaggregation(), br = nobreakage(),
        solver = QMOM(nquadrature = 3, reltol = 1e-10, abstol = 1e-12),
        operation = operation, initial_crystals = tank_seed,
        initial_concentration = 8.0, saturation_model = ConstantSolubility(1.0),
        save_idx = save_grid)

    # Dynamic trajectory vs the analytic relaxation.
    tau = operation.volume / operation.inflow      # residence time = 4 s
    c_ss = operation.feed.solvent_state.concentration
    c_oracle(t) = c_ss + (initial_concentration(problem) - c_ss) * exp(-t / tau)
    m0_trajectory = solution.moments[1, :]
    m0_feed_fraction = 0.05 / 0.5                  # feed carries 1/10 of the seed mass
    m0_oracle(t) = m0_trajectory[1] * (m0_feed_fraction +
                                       (1 - m0_feed_fraction) * exp(-t / tau))
    c_relaxed = solution.concentration[end]
    scaled_c_residual = abs(c_relaxed - c_ss) / c_ss
    m0_relaxed_fraction = m0_trajectory[end] / m0_trajectory[1]

    @assert isapprox(c_relaxed, c_oracle(save_grid[end]); rtol = 1e-6)
    @assert isapprox(m0_trajectory[end], m0_oracle(save_grid[end]); rtol = 1e-6)
    @assert all(observable_values(solution, :volume) .≈ operation.volume)
    @assert operation_flows(problem, 3600.0) == (inflow = 0.5, outflow = 0.5)
    @assert isapprox(solution.d43[end], 20e-6; rtol = 1e-4)      # no growth: seed and feed d43 equal

    # Explicit steady-state solve on the same configured problem.
    steady = solve_steadystate(problem)
    @assert steady.success && steady.converged
    @assert isapprox(steady.concentration[1], c_ss; rtol = 1e-6)
    @assert isapprox(steady.moments[1, 1], m0_trajectory[1] * m0_feed_fraction; rtol = 1e-6)
    @assert steady.scaled_residual_norm < 1e-6
    @assert steady.hydraulics == (volume = 2.0, inflow = 0.5, outflow = 0.5,
                                  residence_time = 4.0)
    @assert steady.product_dissolved_solute_flow ≈ operation.inflow * c_ss

    println("MSMPR (V = $(operation.volume) m³, Q = $(operation.inflow) m³/s, τ = $(tau) s)")
    println("  dynamic: C(T) = $c_relaxed kg/m³,  scaled residual |C(T) − C_ss|/C_ss = $scaled_c_residual")
    println("           M₀(T)/M₀(0) = $m0_relaxed_fraction (steady feed fraction $m0_feed_fraction)")
    println("  steady : C_ss = $(steady.concentration[1]) kg/m³,  scaled residual norm = $(steady.scaled_residual_norm)")
    println("           M₀_ss = $(steady.moments[1, 1]) (feed load),  d43 = $(steady.d43[1]) m")
    println("  product dissolved-solute flow = $(steady.product_dissolved_solute_flow) kg/s  (Q·C_ss)")
    println("  hydraulics: ", steady.hydraulics)

    # Time-dependent inlet profiles are transient inputs, not steady inputs.
    var_flow = MSMPROperation(volume = 2.0, inflow = t -> 0.5 + 0.1 * t / 3600,
                              feed = CrystallisationFeed(concentration = 2.0))
    _, transient_solution = runsimulation(
        Float64[];
        nucl = nucl, gr = gr, agg = noaggregation(), br = nobreakage(),
        solver = QMOM(nquadrature = 3, reltol = 1e-10, abstol = 1e-12),
        operation = var_flow, initial_concentration = 8.0,
        saturation_model = ConstantSolubility(1.0),
        save_idx = [0.0, 3600.0])
    @assert transient_solution.success
    println("  callable inflow at t = 3600 s: ", operation_flows(var_flow, 3600.0),
            " (transient only)")

    fig = Figure(size = (900, 550))
    ax_c = CairoMakie.Axis(fig[1, 1], xlabel = "Time (min)", ylabel = "c (kg/m³)")
    lines!(ax_c, solution.time ./ 60, solution.concentration, color = :dodgerblue,
           label = "dynamic C(t)")
    lines!(ax_c, solution.time ./ 60, c_oracle.(solution.time), color = :crimson,
           linestyle = :dash, label = "analytic approach")
    hlines!(ax_c, [c_ss], color = :black, linestyle = :dot, label = "C_ss")
    scatter!(ax_c, [save_grid[end] ./ 60], [steady.concentration[1]],
             color = :seagreen, label = "solve_steadystate")
    axislegend(ax_c; position = :rt)

    ax_m = CairoMakie.Axis(fig[2, 1], xlabel = "Time (min)", ylabel = "M₀ / M₀(0)")
    lines!(ax_m, solution.time ./ 60, m0_trajectory ./ m0_trajectory[1],
           color = :dodgerblue, label = "dynamic M₀(t)")
    lines!(ax_m, solution.time ./ 60, m0_oracle.(solution.time) ./ m0_trajectory[1],
           color = :crimson, linestyle = :dash, label = "relaxation oracle")
    hlines!(ax_m, [m0_feed_fraction], color = :black, linestyle = :dot, label = "M₀_feed/M₀(0)")
    axislegend(ax_m; position = :rt)
    display(fig)
end

main()