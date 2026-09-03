# v1 flagship: a minimal NON-lysozyme system end-to-end.
# Custom saturation model + custom kinetic family (Tutorial-4 pattern) +
# a custom observable, then runsimulation + loss + a PE evaluation.
# Asserts against hand-computed / physical identities, not re-derived logic.

using ComponentArrays
import CriSTool: AbstractFPScalarGrowthFunction, _named_params
import CriSTool: paramaxis, growthrate
import CriSTool: observable_values

# Custom saturation: solubility that scales linearly with temperature
struct TestSolubility <: CriSTool.AbstractSolubilityModel
    slope::Float64
    intercept::Float64
end
function CriSTool.saturation_concentration(sm::TestSolubility, temp_profile, t)
    return sm.slope * (CriSTool.temperature(temp_profile, t) - 273.15) + sm.intercept
end

# Custom growth: G = growth_coefficient * (S - 1)^growth_order in m/s
# but user-defined to prove the Tutorial-4 extension point)
struct growth_custom <: AbstractFPScalarGrowthFunction
    nparams::Int64
    string::String
    symbols::Vector{Symbol}
end
growth_custom() = growth_custom(2, "Custom Gr", [:growth_coefficient, :growth_order])
paramaxis(::growth_custom) = ComponentArrays.Axis(growth_coefficient = 1, growth_order = 2)
function growthrate(gf::growth_custom, parameters,
                    prob::CrystallisationProblem, state, t)
    p = _named_params(gf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? p.growth_coefficient * (S - 1)^p.growth_order : 0.0
end

function observable_values(sol::CriSTool.CrystallisationMoMSolution, name::Symbol)
    name === :mass && return sol.concentration[1] .- sol.concentration
    return invoke(CriSTool.observable_values,
                  Tuple{CriSTool.AbstractSolution, Symbol}, sol, name)
end

solvent_dynamics = (problem, state, time, growth) -> begin
    default_rates = default_solvent_dynamics(problem, state, time, growth)
    (; concentration = default_rates[1], pH = -0.01 * growth)
end

@testset "Bring your own system (non-lysozyme)" begin
    # Saturation: hand-computed value at 293.15 K (20 °C)
    sat = TestSolubility(0.25, 2.0)
    prof = CriSTool.ConstantTemperature(293.15)
    @test saturation_concentration(sat, prof, 0.0) == 0.25 * 20.0 + 2.0

    # Problem with custom saturation + custom growth + empirical nucleation
    problem = CrystallisationProblem(;
        kinetics_nucleationfunction = nucl_empirical(),
        kinetics_growthfunction = growth_custom(),
        kinetics_aggregationfunction = noaggregation(),
        kinetics_breakagefunction = nobreakage(),
        parameterset_nucleation = [8.0, 2.0],
        parameterset_growth = [1e-9 / 60, 2.0],
        saturation_model = sat,
        initial_concentration = 25.0,
        initial_solvent_state = (; concentration = 25.0, pH = 7.0),
        solvent_dynamics = solvent_dynamics,
        solver = MoM())
    @test saturation_concentration(problem, 0.0) == 7.0
    @test supersaturation(problem, [zeros(5); 14.0; 7.0], 0.0) == 2.0
    @test solvent_state(problem, [zeros(5); 14.0; 7.0]) == (concentration = 14.0, pH = 7.0)

    # Simulation runs and consumes solute monotonically
    _, sol = runsimulation([8.0, 2.0, 1e-9 / 60, 2.0];
                           nucl = problem.kinetics_nucleationfunction,
                           gr = problem.kinetics_growthfunction,
                           agg = noaggregation(), br = nobreakage(),
                           initial_concentration = 25.0,
                           save_idx = [0.0, 1800.0, 3600.0, 7200.0],
                           solver = MoM(),
                           saturation_model = sat,
                           initial_solvent_state = problem.initial_solvent_state,
                           solvent_dynamics = problem.solvent_dynamics)
    @test all(diff(sol.concentration) .<= 1e-12)
    @test sol.concentration[end] < 25.0
    @test sol.solvent_state.pH[end] < sol.solvent_state.pH[1]

    _, fv_sol = runsimulation([8.0, 2.0, 1e-9 / 60, 2.0];
                               nucl = problem.kinetics_nucleationfunction,
                               gr = problem.kinetics_growthfunction,
                               agg = noaggregation(), br = nobreakage(),
                               initial_concentration = 25.0,
                               save_idx = [0.0, 1800.0, 3600.0],
                               solver = FiniteVol(meshsize = 30, lmax = 50e-6),
                               saturation_model = sat,
                               initial_solvent_state = problem.initial_solvent_state,
                               solvent_dynamics = problem.solvent_dynamics)
    @test fv_sol.solvent_state.pH[end] < fv_sol.solvent_state.pH[1]

    # Loss with a custom Observable: concentration only (no size observables)
    # exercises the shape-dispatching container with a minimal experiment.
    expt = CrystallisationExperiment(;
        observables = (;
            concentration = Observable(; time = [0.0, 1800.0, 3600.0, 7200.0],
                                       mean = sol.concentration .* 1.05,
                                       variance = fill(0.5, 4)),
            # custom observable beyond the lysozyme set: accumulated crystal mass
            mass = Observable(; time = [0.0, 1800.0, 3600.0, 7200.0],
                              mean = sol.concentration[1] .- sol.concentration),
            pH = Observable(; time = [0.0, 2700.0, 7200.0],
                            mean = [sol.solvent_state.pH[1],
                                    sol.solvent_state.pH[2],
                                    sol.solvent_state.pH[end]] .+ 0.05,
                            variance = fill(0.01, 3)),
            d43 = Observable(; time = [1800.0, 7200.0],
                             mean = [8e-6, 8.5e-6], variance = [1e-12, 1e-12])),
        temperature = 293.15, exp_id = 1)
    @test expt.observables.mass.mean[2] > 0
    L = loss(logMLE(), problem, [8.0, 2.0, 1e-9 / 60, 2.0], [expt])
    @test L > 0
    @test isfinite(L)

    # PE-style evaluation via the prepared setup (remake path)
    setup = prepare_loss(problem, [expt])
    @test loss(logMLE(), setup, [8.0, 2.0, 1e-9 / 60, 2.0]) ≈ L

    # Every measured observable contributes to the loss, including the custom
    # `mass` and `pH` series.
    objectives = CriSTool._experiment_objectives(logMLE(), expt, sol)
    @test objectives isa Vector
    @test length(objectives) == 4

    # The size objective uses every d43 observation, not only the endpoint.
    size_offsets = [1e-6, 2e-6]
    size_measurement = Observable(; time = [1800.0, 7200.0],
                                  mean = [sol.d43[2], sol.d43[end]] .+ size_offsets,
                                  variance = [1e-12, 4e-12])
    size_experiment = CrystallisationExperiment(;
        observables = merge(expt.observables, (; d43 = size_measurement)),
        temperature = expt.temperature, exp_id = expt.exp_id)
    weighted_objectives = CriSTool._experiment_objectives(
        logMLE(weighting = [0.0, 0.0, 0.0, 1.0]), size_experiment, sol)
    expected_size_objective = 0.5 * sum(
        log.(2π .* [1e-12, 4e-12]) .+
        size_offsets .^ 2 ./ [1e-12, 4e-12])
    @test weighted_objectives[4] ≈ expected_size_objective
end
