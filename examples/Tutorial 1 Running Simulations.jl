"""
Tutorial 1: Running crystallisation simulations under different temperature profiles.

Calls `runsimulation` three times — constant temperature, a piecewise cooling
ramp, and a custom periodic sweep — and plots T(t), c(t), and d43(t)
side-by-side. For custom kinetics see Tutorial 4; for posterior sampling
see Tutorial 5.
"""

using CriSTool
using CairoMakie

function main()
    # Shared kinetics, ICs, and time grid for all three runs.
    # Flat parameter vector: [Aj, γ, Ag, g] = [nucleation; growth].
    nucl, gr   = nucl_CNT(), growth_empirical()
    agg, br    = noaggregation(), nobreakage()
    params     = [38.0, 0.6, 1.0, 3.0]
    save_grid  = 0.0:6.0:360.0    # minutes

    # 1. Constant temperature.
    const_T = CriSTool.ConstantTemperature(295.0)
    _, sol_const = runsimulation(params; nucl=nucl, gr=gr, agg=agg, br=br,
                                  initial_concentration=18.0, solver=MoM(),
                                  temp_profile=const_T, save_idx=save_grid)

    # 2. Piecewise cooling ramp: hold at 297 K for 120 min, cool linearly to
    #    290 K over the next 180 min, then hold. `CallableTemperature` lets us
    #    write any T(t) we want; `RampTemperature` and `LinearTemperature` are
    #    convenience types for the common shapes.
    cool_T = CriSTool.CallableTemperature(t -> t < 120.0 ? 297.0 :
                                              t < 300.0 ? 297.0 + (290.0 - 297.0) * (t - 120.0) / 180.0 :
                                              290.0)
    _, sol_cool = runsimulation(params; nucl=nucl, gr=gr, agg=agg, br=br,
                                 initial_concentration=18.0, solver=MoM(),
                                 temp_profile=cool_T, save_idx=save_grid)

    # 3. Periodic sweep around 295 K, ±5 K with one full cycle over the run.
    sweep_T = CriSTool.CallableTemperature(t -> 295.0 + 5.0 * sin(2π * t / 360.0))
    _, sol_sweep = runsimulation(params; nucl=nucl, gr=gr, agg=agg, br=br,
                                  initial_concentration=18.0, solver=MoM(),
                                  temp_profile=sweep_T, save_idx=save_grid)

    println("Final c (mg/mL) / d43 (μm):")
    println("  constant 295 K     : $(round(sol_const.concentration[end], digits=3)) / $(round(sol_const.d43[end], digits=3))")
    println("  cooling 297 → 290 K: $(round(sol_cool.concentration[end],  digits=3)) / $(round(sol_cool.d43[end],  digits=3))")
    println("  ±5 K periodic sweep: $(round(sol_sweep.concentration[end], digits=3)) / $(round(sol_sweep.d43[end], digits=3))")

    # Three-row figure: temperature profile, concentration, and d43 vs time.
    fig = Figure(size = (1000, 700))
    Label(fig[0, :], "Tutorial 1: temperature-profile comparison",
          fontsize = 16, halign = :left)

    ax_T = CairoMakie.Axis(fig[1, 1], xlabel = "Time (min)", ylabel = "T (K)")
    lines!(ax_T, sol_const.time, [CriSTool.temperature(const_T, t) for t in sol_const.time],
           color = :dodgerblue, label = "constant")
    lines!(ax_T, sol_cool.time,  [CriSTool.temperature(cool_T,  t) for t in sol_cool.time],
           color = :crimson,    label = "cooling ramp")
    lines!(ax_T, sol_sweep.time, [CriSTool.temperature(sweep_T, t) for t in sol_sweep.time],
           color = :seagreen,   label = "sinusoidal")
    axislegend(ax_T; position = :rb)

    ax_c = CairoMakie.Axis(fig[2, 1], xlabel = "Time (min)", ylabel = "c (mg/mL)")
    lines!(ax_c, sol_const.time, sol_const.concentration, color = :dodgerblue)
    lines!(ax_c, sol_cool.time,  sol_cool.concentration,  color = :crimson)
    lines!(ax_c, sol_sweep.time, sol_sweep.concentration, color = :seagreen)

    ax_d = CairoMakie.Axis(fig[3, 1], xlabel = "Time (min)", ylabel = "d43 (μm)")
    lines!(ax_d, sol_const.time, sol_const.d43, color = :dodgerblue)
    lines!(ax_d, sol_cool.time,  sol_cool.d43,  color = :crimson)
    lines!(ax_d, sol_sweep.time, sol_sweep.d43, color = :seagreen)

    display(fig)
end

main()
