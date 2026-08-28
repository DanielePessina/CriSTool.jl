"""
Generate a synthetic experimental crystallisation dataset for CriSTool's
`load_experiments(workbook, sheet_name, loading)` loader.

Truth model:
    nucl = nucl_CNT(),         params = [Aj=38.0, γ=0.6]
    gr   = growth_empirical(), params = [Ag=1.0, g=3.0]
    agg  = noaggregation(), br = nobreakage(), solver = MoM()

Workbook layout matches `CriSTool.load_experiments(filepath, sheet_name, loading)`:
sheet "Unseeded_PE" with columns
    Exp_ID | System | Temperature [°C] | Loading | Time [min]
    | Concentration [mg/mL] | Concentration_var | PS [μm] | PS_var
PS / PS_var are populated only on the LAST timepoint of each experiment
(loader picks them up at the last row, all earlier rows store -1.0 sentinels).
Temperature is stored in Celsius — the loader adds 273.15 to convert to K.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CriSTool
using XLSX
using DataFrames
using Random
using Statistics

const OUT_PATH = joinpath(@__DIR__, "fake-experimental-dataset.xlsx")
const SHEET_NAME = "Unseeded_PE"

# ----------------------------------------------------------------------
# Truth parameters and kinetics
# ----------------------------------------------------------------------

const θ_TRUTH = [38.0, 0.6, 1.0, 3.0]   # [Aj, γ, Ag, g]
nucl_f = nucl_CNT()
gr_f   = growth_empirical()
agg_f  = noaggregation()
br_f   = nobreakage()
solver = MoM()

# ----------------------------------------------------------------------
# Experimental design – 5 unseeded runs covering 285–305 K and 15–22 mg/mL
# ----------------------------------------------------------------------

# Per-experiment conditions: (T_K, c0_mg_per_mL)
# Temperature range chosen so the truth kinetics produce nontrivial nucleation
# (saturation concentration formula in CriSTool gives cs ≈ 0.37 mg/mL at 0°C
# and ≈ 6.6 mg/mL at 32°C, so at ~10–20°C with c0 ≈ 15–22 mg/mL we get S ≈ 5–25).
exp_conditions = [
    (285.0, 22.0),
    (288.0, 20.0),
    (291.0, 18.0),
    (294.0, 17.0),
    (297.0, 15.0),
]

# Irregular sampling grids: dense early, sparse later
function build_time_grid(rng::AbstractRNG, exp_idx::Int)
    if exp_idx == 1
        # 12 points: every 5 min early, every 30 min mid, every 60 min late
        base = vcat(0.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 130.0, 180.0, 220.0, 260.0, 300.0)
    elseif exp_idx == 2
        # 10 points
        base = vcat(0.0, 8.0, 18.0, 35.0, 60.0, 95.0, 140.0, 200.0, 250.0, 300.0)
    elseif exp_idx == 3
        # 14 points: very dense early
        base = vcat(0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, 85.0, 115.0, 150.0,
                    190.0, 235.0, 280.0, 305.0)
    elseif exp_idx == 4
        # 9 points
        base = vcat(0.0, 10.0, 25.0, 50.0, 90.0, 140.0, 200.0, 260.0, 300.0)
    else
        # 11 points
        base = vcat(0.0, 6.0, 14.0, 28.0, 50.0, 80.0, 120.0, 165.0, 215.0, 265.0, 305.0)
    end
    # Add small jitter (±0.5 min) to interior points to mimic instrument timing
    jittered = copy(base)
    for j in 2:(length(base) - 1)
        jittered[j] += (rand(rng) - 0.5) * 1.0
    end
    return round.(jittered, digits = 2)
end

# Heteroscedastic noise parameters
const SIGMA_C_REL  = 0.03   # 3% relative noise on concentration
const SIGMA_C_FLOOR = 0.02  # mg/mL absolute floor
const SIGMA_D_REL  = 0.08   # 8% relative noise on d43

# ----------------------------------------------------------------------
# Build the full long-format DataFrame
# ----------------------------------------------------------------------

Random.seed!(20260501)

rows = NamedTuple[]

for (idx, (T_K, c0)) in enumerate(exp_conditions)
    rng = Random.Xoshiro(20260501 + idx)
    times = build_time_grid(rng, idx)

    # Run truth simulation at the requested temperature with the given save grid
    problem, solution = runsimulation(
        θ_TRUTH;
        nucl = nucl_f, gr = gr_f, agg = agg_f, br = br_f,
        solver = solver,
        initial_concentration = c0,
        save_idx = times,
        temp_profile = CriSTool.ConstantTemperature(T_K),
    )

    @assert solution.success "Simulation failed for exp $idx"
    @assert length(solution.concentration) == length(times)

    c_true = solution.concentration
    d43_true_final = solution.d43[end]

    # Heteroscedastic concentration noise: σ_c = max(SIGMA_C_FLOOR, SIGMA_C_REL*|c|)
    # Sample as obs = truth * (1 + SIGMA_C_REL * randn()) but also enforce floor
    c_obs = similar(c_true)
    c_var = similar(c_true)
    for j in eachindex(c_true)
        σ = max(SIGMA_C_FLOOR, SIGMA_C_REL * abs(c_true[j]))
        c_obs[j] = c_true[j] + σ * randn(rng)
        c_obs[j] = max(c_obs[j], 0.0)  # nonnegative concentration
        c_var[j] = σ^2
    end

    # Heteroscedastic d43 noise (single value per experiment, attached to last point)
    σ_d = SIGMA_D_REL * d43_true_final
    d43_obs = max(d43_true_final + σ_d * randn(rng), 1e-12)
    d43_var = σ_d^2

    # Convert temperature to Celsius for storage (loader adds 273.15)
    T_C = T_K - 273.15

    for j in eachindex(times)
        is_last = j == length(times)
        push!(rows, (
            Exp_ID            = idx,
            System            = "FAKE_SYS_A",
            Temperature       = round(T_C, digits = 2),
            Loading           = 0.0,
            Time              = times[j],
            Concentration     = round(c_obs[j], digits = 6),
            Concentration_var = round(c_var[j], digits = 8),
            PS                = is_last ? round(d43_obs, digits = 6) : -1.0,
            PS_var            = is_last ? round(d43_var, digits = 8) : -1.0,
        ))
    end
end

df = DataFrame(rows)

@info "Built dataframe" nrows = nrow(df) nexp = length(unique(df.Exp_ID))

# ----------------------------------------------------------------------
# Write workbook
# ----------------------------------------------------------------------

isfile(OUT_PATH) && rm(OUT_PATH)

XLSX.openxlsx(OUT_PATH, mode = "w") do xf
    sheet = xf[1]
    XLSX.rename!(sheet, SHEET_NAME)
    cols = names(df)
    # Header
    for (k, name) in enumerate(cols)
        sheet[1, k] = name
    end
    # Data
    for (i, row) in enumerate(eachrow(df))
        for (k, name) in enumerate(cols)
            sheet[i + 1, k] = row[name]
        end
    end
end

@info "Wrote workbook" path = OUT_PATH

# ----------------------------------------------------------------------
# Verify by loading back through CriSTool.load_experiments
# ----------------------------------------------------------------------

loaded = CriSTool.load_experiments(OUT_PATH, SHEET_NAME, 0.0)

println("\n========== VERIFICATION ==========")
println("Number of measurement objects loaded: ", length(loaded))
for (i, m) in enumerate(loaded)
    conc = m.observables.concentration
    println("\n-- Experiment $(m.exp_id) --")
    println("  T (K)       = ", m.temperature)
    println("  Loading     = ", m.loading)
    println("  N timepoints= ", length(conc.time))
    println("  time grid   = ", conc.time)
    println("  conc mean   = ", round.(conc.mean, digits = 4))
    println("  conc var    = ", round.(conc.variance, digits = 6))
    println("  d43         = ", round(m.observables.d43.value, digits = 4),
            "  d43var = ", round(m.observables.d43.variance, digits = 6))
end
println("===================================")
