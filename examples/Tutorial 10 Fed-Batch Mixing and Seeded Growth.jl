"""
Tutorial 10: fed-batch pure mixing and seeded growth.

A fed-batch has one inlet and no outlet: the volume is a reactor state with
dV/dt = Qin, and every intensive density dilutes. Two short deterministic runs:

Part A — pure mixing of a clear feed into a seeded tank (no growth, no
nucleation):

    V(t) = V₀ + Q t
    C(t) = (V₀ C₀ + Q t C_f) / V(t)
    V(t) M₀(t) = V₀ M₀(0)        (seed inventory conserved by dilution)

Part B — the same fed-batch with constant growth G = 2e-9 m/s (achieved with
`growth_order = 0` under constant supersaturation). Each crystal grows by
G·Δt, so d43 shifts by G·Δt while M₀·V stays conserved and the concentration
drops slightly below the pure-mixing value because solute is consumed.

Run with:  julia --project=examples "examples/Tutorial 10 Fed-Batch Mixing and Seeded Growth.jl"
"""

using CriSTool
using CairoMakie

function main()
    nucl = nucl_empirical_fixed(log10_nucleation_prefactor = -Inf,
                                nucleation_order = 1.0)
    operation = FedBatchOperation(initial_volume = 2.0, inflow = 0.5,
                                  feed = CrystallisationFeed(concentration = 2.0))
    seed = LogNormalInitialCrystals(mass_concentration = 0.5, d43 = 20e-6,
                                    geometric_std = 1.05)
    save_grid = [0.0, 4.0]                       # seconds
    v0 = operation.initial_volume
    c0 = 8.0
    c_f = 2.0

    # ---- Part A: pure mixing (zero growth) ----
    gr_none = growth_empirical_fixed(growth_coefficient = 0.0, growth_order = 1.0)
    _, mixing = runsimulation(
        Float64[];
        nucl = nucl, gr = gr_none, agg = noaggregation(), br = nobreakage(),
        solver = QMOM(nquadrature = 3, reltol = 1e-10, abstol = 1e-12),
        operation = operation, initial_crystals = seed,
        initial_concentration = c0, saturation_model = ConstantSolubility(1.0),
        save_idx = save_grid)

    volume = observable_values(mixing, :volume)
    c_oracle(t) = (v0 * c0 + operation.inflow * t * c_f) / (v0 + operation.inflow * t)
    m0_inventory(v, m0) = v * m0
    @assert isapprox(volume[end], 4.0; rtol = 1e-9)
    @assert isapprox(mixing.concentration[end], c_oracle(4.0); rtol = 1e-9)
    @assert isapprox(m0_inventory(volume[1], mixing.moments[1, 1]),
                     m0_inventory(volume[end], mixing.moments[1, end]); rtol = 1e-8)
    println("Part A — pure mixing (V₀ = $v0 m³, Q = 0.5 m³/s, C₀ = $c0, C_f = $c_f)")
    println("  V(4) = $(volume[end]) m³   C(4) = $(mixing.concentration[end]) kg/m³ (mixing oracle $([c_oracle(0.0), c_oracle(4.0)]))")
    println("  seed inventory V·M₀ conserved: $(m0_inventory(volume[1], mixing.moments[1, 1])) ≈ $(m0_inventory(volume[end], mixing.moments[1, end]))")

    # ---- Part B: seeded growth (constant G = 2e-9 m/s) ----
    gr_grow = growth_empirical_fixed(growth_coefficient = 2e-9, growth_order = 0.0)
    _, growing = runsimulation(
        Float64[];
        nucl = nucl, gr = gr_grow, agg = noaggregation(), br = nobreakage(),
        solver = QMOM(nquadrature = 3, reltol = 1e-10, abstol = 1e-12),
        operation = operation, initial_crystals = seed,
        initial_concentration = c0, saturation_model = ConstantSolubility(1.0),
        save_idx = save_grid)

    growth_rate = 2e-9
    d43_shift = growth_rate * (save_grid[end] - save_grid[1])
    growing_volume = observable_values(growing, :volume)
    @assert isapprox(growing.d43[end], growing.d43[1] + d43_shift; rtol = 0.01)
    @assert isapprox(m0_inventory(growing_volume[1], growing.moments[1, 1]),
                     m0_inventory(growing_volume[end], growing.moments[1, end]); rtol = 1e-8)
    @assert growing.concentration[end] < c_oracle(4.0)      # growth consumes solute
    println("Part B — seeded growth (G = $(growth_rate) m/s)")
    println("  d43(0) = $(growing.d43[1]) m → d43(4) = $(growing.d43[end]) m (growth shift $(d43_shift) m)")
    println("  V·M₀ conserved: $(m0_inventory(growing_volume[1], growing.moments[1, 1])) ≈ $(m0_inventory(growing_volume[end], growing.moments[1, end]))")
    println("  C(4) = $(growing.concentration[end]) kg/m³ vs pure-mixing $(c_oracle(4.0)) kg/m³")

    fig = Figure(size = (900, 550))
    ax_v = CairoMakie.Axis(fig[1, 1], xlabel = "Time (s)", ylabel = "V (m³)")
    lines!(ax_v, save_grid, volume, color = :dodgerblue, label = "pure mixing V(t)")
    lines!(ax_v, save_grid, growing_volume, color = :crimson, linestyle = :dash,
           label = "growing V(t)")
    axislegend(ax_v; position = :lt)

    ax_c = CairoMakie.Axis(fig[2, 1], xlabel = "Time (s)", ylabel = "c (kg/m³)")
    lines!(ax_c, save_grid, mixing.concentration, color = :dodgerblue, label = "pure mixing")
    lines!(ax_c, save_grid, growing.concentration, color = :crimson, label = "seeded growth")
    lines!(ax_c, save_grid, c_oracle.(save_grid), color = :black, linestyle = :dot,
           label = "mixing oracle")
    axislegend(ax_c; position = :rt)
    display(fig)
end

main()