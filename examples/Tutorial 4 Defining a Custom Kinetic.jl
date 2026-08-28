"""
Tutorial 4: Defining a custom kinetic family.

Three pieces let you add a new kinetic without modifying CriSTool:
  1. A struct subtyping the right Abstract* family.
  2. A `paramaxis` method declaring a ComponentArrays Axis.
  3. A `growthrate` / `nucleationrate` / `aggregationrate` / `breakagerate`
     method with the standard signature
     `(fn, parameters, prob::CrystallisationProblem, state, t)`.

The example adds a Michaelis-Menten-style saturation growth law:

    G(S) = A * 1e-9 * (S - 1) / (B + (S - 1))    for S > 1, else 0

where S = supersaturation = state[end] / saturation_concentration(prob, t).
"""

using CriSTool
using CriSTool: AbstractFPScalarGrowthFunction, _named_params
import CriSTool: paramaxis, growthrate   # `import` (not `using`) to extend
using ComponentArrays
using CairoMakie

# 1. Struct.
struct growth_saturation <: AbstractFPScalarGrowthFunction
    nparams::Int64
    string::String
    symbols::Vector{Symbol}
end
growth_saturation() = growth_saturation(2, "Saturation Gr", [:A, :B])

# 2. Axis — powers `p.A` / `p.B` named access in the rate function below.
paramaxis(::growth_saturation) = ComponentArrays.Axis(A = 1, B = 2)

# 3. Rate function. Standard signature: (fn, params, prob, state, t).
#    The rate computes S itself via supersaturation(prob, state, t) and reads
#    temperature from prob.temp_profile — no positional S/temperature/loading.
function growthrate(gf::growth_saturation, parameters::AbstractVector,
                    prob::CrystallisationProblem, state, t)
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? p.A * 1e-9 * (S - 1) / (p.B + (S - 1)) : 0.0
end

function main()
    # Rate function callable in isolation — build a problem and a state.
    prob = CrystallisationProblem(; kinetics_growthfunction = growth_saturation(),
                                  solver = MoM())
    state = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5 * saturation_concentration(prob, 0.0)]
    println("G(S=1.5, A=1.5, B=0.3) = ",
            growthrate(growth_saturation(),
                       ComponentVector(A = 1.5, B = 0.3),
                       prob, state, 0.0), " m/s")

    # Plug into runsimulation. ComponentVector input keeps the layout explicit;
    # a flat [Aj, γ, A, B] vector also works.
    params = ComponentVector(nucl = (Aj = 38.0, γ = 0.6),
                              gr   = (A = 1.5, B = 0.3),
                              agg  = Float64[], br = Float64[])
    _, sol = runsimulation(params;
                            nucl = nucl_CNT(), gr = growth_saturation(),
                            agg = noaggregation(), br = nobreakage(),
                            initial_concentration = 18.0,
                            solver = MoM(),
                            save_idx = 0.0:6.0:360.0)

    println("Final concentration: $(sol.concentration[end]) mg/mL")
    println("Final d43:           $(sol.d43[end]) μm")

    fig = Figure(size = (900, 400))
    Label(fig[0, :], "Tutorial 4: custom growth_saturation kinetic",
          fontsize = 16, halign = :left)
    ax_c = CairoMakie.Axis(fig[1, 1], xlabel = "Time (min)",
                            ylabel = "Concentration (mg/mL)")
    lines!(ax_c, sol.time, sol.concentration, color = :dodgerblue)
    ax_d = CairoMakie.Axis(fig[1, 2], xlabel = "Time (min)",
                            ylabel = "d43 (μm)")
    lines!(ax_d, sol.time, sol.d43, color = :seagreen)
    display(fig)
end

main()