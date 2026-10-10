# End-to-end simulation: model → conditions → simulate across every solver,
# checked against closed forms (rigid translation, constant-kernel
# aggregation, constant-rate breakage), mass balances and one frozen nonlinear
# regression. Small meshes and short horizons keep the file fast; each
# tolerance is justified next to its assertion.

# Shared helpers (prefix `simulate_` keeps them greppable).
simulate_translation_times = [0.0, 50.0, 100.0]
simulate_mesh_solvers = (FiniteVol(meshsize = 40, lmax = 40e-6), WENO(meshsize = 40, lmax = 40e-6))
# Raw length moment k of a mesh trajectory, per saved time.
simulate_mesh_moment(trajectory, mesh_solver, order) =
    vec(sum(trajectory.population.numberdensity .* mesh_solver.cell_centre .^ order; dims = 1)) .* mesh_solver.cell_dL
simulate_raw_moment(trajectory::CrystallisationSolution{<:QuadraturePopulation}, solver, order) =
    trajectory.population.moments[order + 1, :]
simulate_raw_moment(trajectory::CrystallisationSolution{<:MeshPopulation}, solver, order) =
    simulate_mesh_moment(trajectory, solver, order)

@testset "Translation closed form on every solver" begin
    # Constant G = 1e-8 m/s for 100 s shifts the uniform [10, 20] µm seed by
    # 1 µm; solute lost equals crystal mass gained.
    expected_concentration = fixture_translation_concentration.(simulate_translation_times)
    expected_d43 = fixture_translation_d43.(simulate_translation_times)
    concentration_drop = 20.0 - expected_concentration[end]   # ≈ 0.05 kg/m³
    # Moment methods are exact for constant growth: only ODE tolerance
    # (reltol ≤ 1e-7) remains. Mesh solvers conserve C + ρkv·µ3 exactly, so
    # the concentration drop equals their discrete solid gain. On 40 cells the
    # scheme smears the sharp-edged uniform seed, and the smeared population
    # holds more third moment than the rigidly shifted one: measured 3.5%
    # (FiniteVol) / 3.4% (WENO) excess on the drop, first order in ΔL (0.53% /
    # 0.45% at 160 cells, 0.075% / 0.055% at 640), and ≤ 5e-3 on d43. Asserted
    # with 1.4x / 3x margin; the 160-cell check below confirms convergence.
    for (solver, drop_rtol, d43_rtol) in ((MoM(), 1e-6, 1e-8), (QMOM(), 1e-5, 1e-6), (DQMOM(), 1e-5, 1e-6),
            (simulate_mesh_solvers[1], 5e-2, 1.5e-2), (simulate_mesh_solvers[2], 5e-2, 1.5e-2),
            (FiniteVol(meshsize = 160, lmax = 40e-6), 8e-3, 3e-3))
        translation_trajectory = simulate(fixture_translation_model(), fixture_seeded_conditions();
            solver, saveat = simulate_translation_times)
        @test maximum(abs.(translation_trajectory.concentration .- expected_concentration)) <=
              drop_rtol * concentration_drop
        @test translation_trajectory.d43 ≈ expected_d43 rtol = d43_rtol
        if solver isa QMOM
            # Every tracked raw moment follows µ0·E[(L + Gt)^k].
            seed_number = fixture_seed_mass / 1000.0 / fixture_uniform_moment(3)
            @test translation_trajectory.population.moments[:, end] ≈
                  [seed_number * fixture_uniform_moment(order, 1e-6) for order in 0:5] rtol = 1e-6
        elseif solver isa DQMOM
            # Characteristics: nodes move by G·t = 1 µm, weights are constant.
            translation_rules = quadrature(translation_trajectory)
            @test translation_rules[end].nodes ≈ translation_rules[1].nodes .+ 1e-6 rtol = 1e-6
            @test translation_rules[end].weights ≈ translation_rules[1].weights rtol = 1e-8
        end
    end

    # The low-level parameter-vector interface agrees with simulate for the
    # same configured problem, and a replaced parameter takes effect: doubling
    # the speed doubles the shift (closed form at 2 µm).
    translation_problem = CrystallisationProblem(fixture_translation_model(), fixture_seeded_conditions())
    _, lowlevel_trajectory = runsimulation([0.02], translation_problem; save_idx = [0.0, 100.0])
    @test lowlevel_trajectory.concentration[end] ≈ fixture_translation_concentration(100.0; speed = 0.02) rtol = 1e-9
    @test simulate(translation_problem; saveat = [0.0, 100.0]).concentration[end] ≈ expected_concentration[end] rtol = 1e-9
    @test_throws ArgumentError runsimulation([0.01, 0.0], translation_problem; save_idx = [0.0, 1.0])
end

@testset "Seed descriptions become solver states" begin
    lognormal_seed = LogNormalInitialCrystals(mass_concentration = 0.25, d43 = 12e-6, geometric_std = 1.25)
    gaussian_seed = GaussianInitialCrystals(mass_concentration = 0.25, d43 = 12e-6, standard_deviation = 2e-6)
    # Moment and quadrature states reproduce the requested mass (rho·kv = 1000)
    # and d43 to roundoff. The 1 µm mesh conserves the exact cell-integrated
    # volume, so the centre-rule µ3 used here differs by the midpoint error
    # ~(ΔL/L)²/4 ≈ 2e-3 (asserted 5e-3); d43 is resolved to the 5% cell width.
    for seed_description in (lognormal_seed, gaussian_seed)
        for (solver, mass_rtol, d43_rtol) in ((MoM(), 1e-12, 1e-12), (QMOM(), 1e-10, 1e-10), (DQMOM(), 1e-10, 1e-10),
                (simulate_mesh_solvers[1], 5e-3, 5e-2), (simulate_mesh_solvers[2], 5e-3, 5e-2))
            # Zero speed keeps the compiled translation RHS but freezes the seed.
            seeded_problem = CrystallisationProblem(fixture_translation_model(speed = 0.0),
                RunConditions(temperature = 293.15, concentration = 20.0, crystals = seed_description); solver)
            seeded_start = simulate(seeded_problem; saveat = [0.0, 1.0])
            seeded_mass = solver isa MoM ? 1000.0 * seeded_problem.initial_population[4] :
                          1000.0 * first(simulate_raw_moment(seeded_start, solver, 3))
            @test seeded_mass ≈ 0.25 rtol = mass_rtol
            @test seeded_start.d43 ≈ [12e-6, 12e-6] rtol = d43_rtol
            @test seeded_start.concentration == [20.0, 20.0]
        end
    end

    # A volume-weighted lognormal is the number lognormal shifted by 3σ² in
    # log space; both give the same state. Uniform volume weighting on
    # [10, 20] µm has d43 = 15 µm exactly.
    number_seed = DistributionInitialCrystals(LogNormal(log(12e-6), 0.2); mass_concentration = 0.25)
    volume_seed = DistributionInitialCrystals(LogNormal(log(12e-6) + 3 * 0.2^2, 0.2);
        mass_concentration = 0.25, weighting = :volume)
    @test number_weighted(volume_seed).distribution == number_seed.distribution
    distribution_conditions(seed_description) = RunConditions(temperature = 293.15, concentration = 20.0, crystals = seed_description)
    @test CrystallisationProblem(fixture_translation_model(), distribution_conditions(number_seed)).initial_population ≈
          CrystallisationProblem(fixture_translation_model(), distribution_conditions(volume_seed)).initial_population rtol = 1e-12
    @test DistributionInitialCrystals(Uniform(10e-6, 20e-6); mass_concentration = 0.25, weighting = :volume).d43 ≈
          15e-6 rtol = 1e-9

    # Every entry point resolves a seed the same way: the state built directly
    # from a parametric seed equals the one the model path stores, on a mesh
    # (cell integrals) and for moments.
    for solver in (simulate_mesh_solvers[1], MoM())
        unseeded_problem = CrystallisationProblem(fixture_translation_model(),
            RunConditions(temperature = 293.15, concentration = 20.0); solver)
        @test initial_state_from_characteristics(unseeded_problem, lognormal_seed) ==
              CriSTool._get_initial_state(CrystallisationProblem(fixture_translation_model(),
                  distribution_conditions(lognormal_seed); solver))
    end

    # A single save time returns the initial state on every moment solver.
    single_save_start = simulate(fixture_translation_model(), fixture_seeded_conditions();
        solver = QMOM(), saveat = [0.0])
    @test single_save_start.time == [0.0]
    @test single_save_start.concentration == [20.0]
    @test single_save_start.d43 ≈ [fixture_translation_d43(0.0)] rtol = 1e-10

    # An empty seed (zero mass) is a valid empty population.
    empty_seed_start = simulate(fixture_translation_model(), RunConditions(temperature = 293.15, concentration = 18.0,
        crystals = LogNormalInitialCrystals(mass_concentration = 0.0, d43 = 0.0, geometric_std = 1.25)); saveat = [0.0, 10.0])
    @test empty_seed_start.concentration == [18.0, 18.0]
end

@testset "Aggregation and breakage closed forms" begin
    # Constant aggregation kernel K and constant breakage selection S with
    # uniform-in-volume binary daughters give the logistic number balance
    #   dµ0/dt = S µ0 - (K/2) µ0²  →  µ0(t) = µ0 e^{St} / (1 + (K µ0/2S)(e^{St} - 1)),
    # while both events conserve crystal volume (µ3) and, with zero growth,
    # the solute. K µ0/2S = 0.5 here, so a sign error in either term moves
    # µ0(100 s) far from the 1.46 µ0 closed form. One combined model keeps the
    # compile cost to one RHS per solver.
    aggregation_kernel = 1.5e-13
    breakage_selection = 0.01
    event_model = CrystallisationModel(fixture_system;
        growth = fixture_constant_growth(speed = 0.0),
        aggregation = ConstantAggregation(log10_aggregation_coefficient = log10(aggregation_kernel)),
        breakage = EmpiricalBreakage(breakage_coefficient = breakage_selection, breakage_size_exponent = 0.0))
    logistic_number(initial_number, elapsed_time) = initial_number * exp(breakage_selection * elapsed_time) /
        (1 + aggregation_kernel * initial_number / (2 * breakage_selection) * expm1(breakage_selection * elapsed_time))
    # Quadrature closures are exact for constant kernels (ODE tolerance only;
    # measured ≤ 2e-8). The 40-cell fixed-pivot mesh conserves number per
    # event exactly but loses aggregates beyond lmax and carries the O(ΔL)
    # breakage volume error checked in test_numerics.jl: measured µ0 3e-5,
    # µ3 6e-3.
    for (solver, number_rtol, volume_rtol) in ((QMOM(), 1e-6, 1e-7), (DQMOM(), 1e-6, 1e-7),
            (simulate_mesh_solvers[1], 2e-4, 2e-2))
        event_trajectory = simulate(event_model, fixture_seeded_conditions(); solver, saveat = simulate_translation_times)
        event_number = simulate_raw_moment(event_trajectory, solver, 0)
        @test event_number ≈ logistic_number.(event_number[1], simulate_translation_times) rtol = number_rtol
        @test simulate_raw_moment(event_trajectory, solver, 3) ≈ fill(fixture_seed_mass / 1000.0, 3) rtol = volume_rtol
        @test event_trajectory.concentration == fill(20.0, 3)
    end
end

@testset "QMOM keeps a rank-deficient population under nonlinear growth" begin
    # Two atoms (10, 20 µm, 1e10 /m³ each) in six moments are a deflated
    # rule. Size-independent growth G(C) translates both rigidly: weights and
    # the 10 µm spacing are invariant and C + ρkv·µ3 = 20 + 1000·1e10·(10³ +
    # 20³)·1e-18 = 20.09. Integration error must not make the saved moments
    # unrealizable (G varies with C, so Tsit5 is not exact here).
    two_atom_nodes = [10e-6, 20e-6]
    two_atom_weights = [1e10, 1e10]
    two_atom_model = CrystallisationModel(fixture_system;
        growth = EmpiricalGrowth(growth_coefficient = 1e-8, growth_order = 2.0))
    two_atom_problem = CriSTool._problem_with_initial_values(
        CrystallisationProblem(two_atom_model, RunConditions(temperature = 293.15, concentration = 20.0); solver = QMOM()),
        [sum(two_atom_weights .* two_atom_nodes .^ order) for order in 0:5], (; concentration = 20.0))
    two_atom_trajectory = simulate(two_atom_problem; saveat = collect(0.0:20.0:200.0))
    two_atom_rules = quadrature(two_atom_trajectory)
    @test all(rule -> rule.active_nodes == 2, two_atom_rules)
    @test all(rule -> isapprox(sort(rule.weights), two_atom_weights; rtol = 1e-6), two_atom_rules)
    @test all(rule -> isapprox(-(extrema(rule.nodes)...), -10e-6; rtol = 1e-6), two_atom_rules)
    @test two_atom_trajectory.concentration .+ 1000.0 .* two_atom_trajectory.population.moments[4, :] ≈
          fill(20.09, length(two_atom_trajectory.time)) rtol = 1e-12
    # The atoms did move: C fell, so the measure is not frozen.
    @test two_atom_trajectory.concentration[end] < 19.99
end

@testset "Dissolution to extinction conserves solute" begin
    # C = 5 < C_sat = 10: the 0.25 kg/m³ seed dissolves completely within
    # seconds and cannot regrow (5.25 < 10), so the closed-form end state is
    # an empty population at C = 5 + 0.25.
    extinction_model = CrystallisationModel(fixture_system;
        growth = EmpiricalGrowth(growth_coefficient = 1e-8, growth_order = 1.0),
        dissolution = ArrheniusDissolution(dissolution_coefficient = 1e-6, activation_energy = 0.0, dissolution_order = 1.0))
    extinction_conditions = fixture_seeded_conditions(concentration = 5.0)
    extinction_problem(solver) = CrystallisationProblem(extinction_model, extinction_conditions;
        solver, solid_mass_concentration_threshold = 5e-4)
    moment_extinction(end_time) = simulate(extinction_problem(MoM()); saveat = [0.0, end_time])
    moment_extinction_trajectory = moment_extinction(60.0)
    @test moment_extinction_trajectory.concentration[end] ≈ 5.25 rtol = 1e-10
    # The moment state is reset to an exactly empty population.
    @test all(iszero, moment_extinction_trajectory.final_state[1:5])
    @test moment_extinction_trajectory.d43[end] == 0.0

    # QMOM nodes translate with G < 0; a node reaching L = 0 carries no
    # volume, so removing it changes only µ0 and never C + ρkv·µ3 (5 + 0.25
    # at every saved time, roundoff only). |G| ≤ 1e-6·(1 - 5/10) = 5e-7 m/s,
    # so at 10 s every crystal is still above 5 µm: up to then both moment
    # solvers integrate the same exact translation and agree to ODE tolerance.
    # Extinction happens either through the solid-mass threshold (5e-4) or,
    # with the default 1e-12 threshold, through the last node's removal.
    for extinction_threshold in (5e-4, 1e-12)
        quadrature_extinction = simulate(CrystallisationProblem(extinction_model, extinction_conditions;
            solver = QMOM(), solid_mass_concentration_threshold = extinction_threshold); saveat = [0.0, 10.0, 30.0, 60.0])
        @test quadrature_extinction.concentration[end] ≈ 5.25 rtol = 1e-12
        @test quadrature_extinction.concentration .+ 1000.0 .* quadrature_extinction.population.moments[4, :] ≈
              fill(5.25, length(quadrature_extinction.time)) rtol = 1e-12
        @test all(iszero, quadrature_extinction.final_state[1:6])
        @test quadrature_extinction.concentration[findfirst(==(10.0), quadrature_extinction.time)] ≈
              moment_extinction(10.0).concentration[end] rtol = 1e-7
        # Without nucleation dissolution only removes nodes, one at a time.
        active_node_counts = [rule.active_nodes for rule in quadrature(quadrature_extinction)]
        @test active_node_counts[1] == 3 && active_node_counts[end] == 0
        @test all(diff(active_node_counts) .<= 0)
    end

    # Mesh solvers take the solute from the exact discrete change of the
    # cell-centre third moment plus the crystal volume crossing the size
    # boundaries at face size (zero at lmin = 0), so C + ρkv·µ3 is a linear
    # invariant of the RHS: Runge-Kutta steps preserve it to round-off
    # (measured ≤ 3e-14 relative; 1e-11 allows accumulation over the steps).
    # The inventory is the seed as projected on the 40 cells (0.24975, not
    # 0.25 kg/m³), and at 60 s the first-order tail still holds ≤ 1.2e-5
    # kg/m³, so C itself is within 0.2% of the seed mass of 5.25 (measured 0.1%).
    for mesh_solver in simulate_mesh_solvers
        mesh_extinction = simulate(extinction_problem(mesh_solver); saveat = [0.0, 30.0, 60.0])
        mesh_inventory = mesh_extinction.concentration .+
                         1000.0 .* simulate_mesh_moment(mesh_extinction, mesh_solver, 3)
        @test mesh_inventory ≈ fill(mesh_inventory[1], 3) rtol = 1e-11
        @test abs(mesh_extinction.concentration[end] - 5.25) <= 2e-3 * fixture_seed_mass
    end
    # Scalar dissolution would collapse DQMOM nodes; it is rejected rather
    # than integrated into a non-physical state.
    @test_throws DomainError simulate(extinction_model, extinction_conditions; solver = DQMOM(), saveat = [0.0, 60.0])
end

@testset "Mesh nucleation and growth conserve solute" begin
    # Supersaturated growth (G = 1e-8 m/s at S = 2) with constant nucleation
    # B = 1e10 #/m³/s inserting nuclei at L = 0. Over 60 s no crystal reaches
    # lmax = 40 µm and nothing crosses lmin = 0 with volume, so total solute
    # C + ρkv·µ3 (cell-centre µ3) is invariant. The former continuous
    # 3ρkv·G·µ2 coupling ignored the nuclei and the cell-to-cell volume steps
    # and drifted by 7e-5 of the inventory (4.7% of the solute consumed); the
    # flux-consistent coupling holds it to round-off (measured ≤ 7e-16;
    # 1e-11 as in the extinction check, still 1e-8 of the solute consumed).
    growing_model = CrystallisationModel(fixture_system;
        nucleation = fixture_constant_nucleation(birth_rate = 1e10),
        growth = EmpiricalGrowth(growth_coefficient = 1e-8, growth_order = 1.0))
    for mesh_solver in simulate_mesh_solvers
        growing_trajectory = simulate(growing_model, fixture_seeded_conditions(concentration = 20.0);
            solver = mesh_solver, saveat = [0.0, 30.0, 60.0])
        growing_inventory = growing_trajectory.concentration .+
                            1000.0 .* simulate_mesh_moment(growing_trajectory, mesh_solver, 3)
        # Guard against a vacuous invariant: solute is really consumed.
        @test growing_trajectory.concentration[end] < 20.0 - 0.02
        @test growing_inventory ≈ fill(growing_inventory[1], 3) rtol = 1e-11
    end
end

@testset "Frozen nonlinear lysozyme regression" begin
    # Frozen 2026-10 from MoM (reltol 1e-10) for the README CNT + empirical
    # growth model at 18 kg/m³, 293.15 K. The t = 7200 s value matches the
    # independently frozen WENO-era fixture (11.017412693766), so these
    # numbers guard nonlinear nucleation-growth kinetics end to end.
    lysozyme_times = [0.0, 3600.0, 7200.0]
    lysozyme_trajectory = simulate(fixture_lysozyme_model, fixture_lysozyme_conditions; saveat = lysozyme_times)
    @test lysozyme_trajectory.concentration ≈ [18.0, 16.03303066517201, 11.017412694123134] rtol = 1e-7
    @test lysozyme_trajectory.d43[2:3] ≈ [1.102526806574247e-5, 1.5618879025512232e-5] rtol = 1e-6
    # Moment closure identity: (µ0, µ1, µ2, C) is closed, so tracking only
    # µ0..µ2 gives the same concentration; d43 needs µ4 and is NaN without it.
    closure_trajectory = simulate(fixture_lysozyme_model, fixture_lysozyme_conditions;
        solver = MoM(nmoments = 2), saveat = lysozyme_times)
    @test closure_trajectory.concentration ≈ lysozyme_trajectory.concentration rtol = 1e-8
    @test all(isnan, closure_trajectory.d43)
end

# Tutorial-4 style custom pieces: a temperature-linear solubility and a
# smooth linear-in-supersaturation growth law (no deadband).
struct SimulateLinearSolubility <: CriSTool.AbstractSolubilityModel
    slope_per_kelvin::Float64
    intercept::Float64
end
CriSTool.saturation_concentration(solubility::SimulateLinearSolubility, temperature_profile, elapsed_time) =
    solubility.slope_per_kelvin * (CriSTool.temperature(temperature_profile, elapsed_time) - 273.15) + solubility.intercept

struct SimulateLinearGrowth <: AbstractScalarGrowthFunction end
CriSTool.paramaxis(::SimulateLinearGrowth) = CriSTool.ComponentArrays.Axis(linear_growth_coefficient = 1)
CriSTool.growthrate(::SimulateLinearGrowth, kinetic_parameters, configured_run::CrystallisationProblem, ode_state, elapsed_time) =
    kinetic_parameters[1] * (supersaturation(configured_run, ode_state, elapsed_time) - 1)

@testset "Custom solubility, kinetic law and solvent species" begin
    custom_solubility = SimulateLinearSolubility(0.25, 2.0)
    # Hand value at 303.15 K: 0.25·30 + 2 = 9.5 kg/m³.
    @test saturation_concentration(custom_solubility, ConstantTemperature(303.15), 0.0) == 9.5
    custom_system = CrystallisationSystem(saturation_model = custom_solubility, crystal_density = 1000.0, volume_shape_factor = 1.0)
    custom_model = CrystallisationModel(custom_system;
        growth = KineticModel(SimulateLinearGrowth(); parameters = (linear_growth_coefficient = 1e-5,)))
    # Extra species: d(tracer)/dt = -1e4·G, so tracer(t) = 1 - 1e4·∫G dt, and
    # ∫G dt is exactly the shift of the mean length d10 = µ1/µ0 (no nucleation).
    tracer_dynamics = (configured_run, ode_state, elapsed_time, growth_rate) -> begin
        default_rates = default_solvent_dynamics(configured_run, ode_state, elapsed_time, growth_rate)
        (; concentration = default_rates[1], tracer = -1e4 * growth_rate)
    end
    custom_conditions = RunConditions(temperature = 303.15, concentration = 10.0, solvent = (; tracer = 1.0),
        crystals = fixture_uniform_seed())
    custom_problem = CrystallisationProblem(custom_model, custom_conditions; solvent_dynamics = tracer_dynamics)
    custom_trajectory = simulate(custom_problem; saveat = [0.0, 50.0, 400.0])
    # dC/dt ≈ -3ρkv·k·µ2·(C/9.5 - 1) gives a relaxation time ≲ 20 s, so after
    # 400 s the solute sits at the custom solubility (residual ~e^{-20}).
    @test custom_trajectory.concentration[end] ≈ 9.5 rtol = 1e-6
    @test observable_values(custom_trajectory, :tracer) ≈
          1.0 .- 1e4 .* (custom_trajectory.d10 .- custom_trajectory.d10[1]) rtol = 1e-6
    @test_throws ArgumentError observable_values(custom_trajectory, :unknown_species)

    # Extra material properties reach callable laws through the context:
    # a speed stored on the system reproduces the translation closed form,
    # and a law requiring an absent property is rejected up front.
    property_system = CrystallisationSystem(saturation_model = ConstantSolubility(10.0), crystal_density = 1000.0,
        volume_shape_factor = 1.0, properties = (; seed_translation_speed = 1e-8))
    property_growth(required_names) = CallableGrowth((rate_values, rate_context) -> rate_context.system.properties.seed_translation_speed;
        parameters = (;), required = required_names)
    property_trajectory = simulate(CrystallisationModel(property_system;
        growth = property_growth((:seed_translation_speed,))), fixture_seeded_conditions(); saveat = [0.0, 100.0])
    @test property_trajectory.concentration[end] ≈ fixture_translation_concentration(100.0) rtol = 1e-9
    propertyless_model = CrystallisationModel(fixture_system;
        growth = property_growth((:seed_translation_speed,)))
    @test_throws ArgumentError CrystallisationProblem(propertyless_model, fixture_seeded_conditions())

    # Solvent dynamics also apply to an empty population: a constant rate of
    # -1 kg/m³/s over 2 s, integrated by explicit Euler, is exact.
    draining_problem = CrystallisationProblem(fixture_translation_model(speed = 0.0),
        RunConditions(temperature = 293.15, concentration = 20.0);
        solvent_dynamics = (configured_run, ode_state, elapsed_time, growth_rate) -> (; concentration = -1.0))
    euler_options(iteration_budget) = (; dt = 0.25, adaptive = false, maxiters = iteration_budget)
    drained_trajectory = simulate(draining_problem; saveat = [0.0, 2.0], algorithm = CriSTool.OrdinaryDiffEq.Euler(),
        solve_options = euler_options(8))
    @test drained_trajectory.success
    @test drained_trajectory.concentration ≈ [20.0, 18.0] atol = 1e-12
    # Eight steps are needed; a budget of four fails and is reported, not thrown.
    @test !simulate(draining_problem; saveat = [0.0, 2.0], algorithm = CriSTool.OrdinaryDiffEq.Euler(),
        solve_options = euler_options(4)).success
    # Options owned by the package are refused instead of silently overridden.
    @test_throws ArgumentError simulate(draining_problem; saveat = [0.0, 2.0], solve_options = (; callback = nothing))
    @test_throws ArgumentError simulate(draining_problem; saveat = [0.0, 2.0], solve_options = (; save_idxs = [1]))
end

@testset "WENO applies a solve-local physical timestep cap" begin
    # Empty population at C = 2 > C_sat = 1: the exact solution is constant,
    # so error control alone would take huge steps. With 10 µm cells and
    # |G| = 1 µm/s the CFL cap is 0.9·10 = 9 s.
    cfl_system = CrystallisationSystem(saturation_model = ConstantSolubility(1.0), crystal_density = 1000.0, volume_shape_factor = 1.0)
    cfl_conditions = RunConditions(temperature = 293.15, concentration = 2.0)
    accepted_times = Float64[]
    step_observer = odeproblem -> CriSTool.DiscreteCallback((ode_state, elapsed_time, integrator) -> true,
        integrator -> push!(accepted_times, integrator.t); save_positions = (false, false))
    function largest_accepted_step(cfl_problem; saveat = [0.0, 100.0], solve_options = (;))
        empty!(accepted_times)
        push!(accepted_times, 0.0)
        cfl_trajectory = simulate(cfl_problem; saveat, callback_factory = step_observer, solve_options)
        @test cfl_trajectory.concentration == fill(2.0, length(saveat))
        return maximum(diff(accepted_times))
    end
    for transport_speed in (1.0, -1.0)
        cfl_problem = CrystallisationProblem(
            CrystallisationModel(cfl_system; growth = fixture_constant_growth(speed = transport_speed)),
            cfl_conditions; solver = WENO(meshsize = 10, lmax = 100e-6, timestepping_algorithm = transport_speed < 0 ? CriSTool.OrdinaryDiffEq.SSPRK43() : nothing))
        @test largest_accepted_step(cfl_problem) <= 9.0 + 1e-12
        # A tighter user dtmax wins; it must not leak into the next solve.
        @test largest_accepted_step(cfl_problem; solve_options = (; dtmax = 2.0)) <= 2.0 + 1e-12
        @test 2.0 < largest_accepted_step(cfl_problem) <= 9.0 + 1e-12
        # A fixed user dt larger than the cap is still capped.
        @test largest_accepted_step(cfl_problem; solve_options = (; adaptive = false, dt = 50.0)) <= 9.0 + 1e-12
    end
    # Length-dependent G = 2 µm/s · L/10 µm peaks at the last centre (95 µm):
    # Gmax = 19 µm/s, cap 9/19 s.
    length_cfl_problem = CrystallisationProblem(CrystallisationModel(cfl_system;
            growth = CallableLengthGrowth((rate_values, rate_context, crystal_length) ->
                rate_values.speed * 1e-6 * 2 * crystal_length / 10e-6; parameters = (; speed = 1.0))),
        cfl_conditions; solver = WENO(meshsize = 10, lmax = 100e-6))
    @test largest_accepted_step(length_cfl_problem; saveat = [0.0, 10.0]) <= 9 / 19 + 1e-12
end

@testset "Input contracts that guard user mistakes" begin
    # Keyword binding must name every parameter, and only real ones.
    @test_throws ArgumentError EmpiricalGrowth(growth_order = 2.0)
    @test_throws ArgumentError EmpiricalGrowth(wrong = 1.0, growth_order = 2.0)
    # An unbound law with parameters cannot enter a model.
    @test_throws ArgumentError CrystallisationModel(fixture_system; nucleation = CNTNucleation(), growth = fixture_constant_growth())
    # CNT needs a molecular volume the fixture system does not define.
    @test_throws ArgumentError CrystallisationProblem(CrystallisationModel(fixture_system;
        nucleation = CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4), growth = fixture_constant_growth()),
        fixture_seeded_conditions())
    # Missing concentration is an unresolved observation placeholder.
    @test_throws ArgumentError CrystallisationProblem(fixture_translation_model(), RunConditions(temperature = 293.15))
    @test_throws ArgumentError RunConditions(temperature = 293.15, concentration = 20.0, solvent = (; concentration = 20.0))
    # The seed comes from the conditions; a raw state override is refused.
    @test_throws ArgumentError CrystallisationProblem(fixture_translation_model(), fixture_seeded_conditions(); initial_state = zeros(6))
    # Seeds: negative support, and a mesh that would drop most of the seed.
    @test_throws ArgumentError DistributionInitialCrystals(Normal(12e-6, 1e-6); mass_concentration = 0.25)
    @test_throws ArgumentError CrystallisationProblem(fixture_translation_model(), fixture_seeded_conditions();
        solver = FiniteVol(meshsize = 40, lmax = 13e-6))
    @test_throws ArgumentError CrystallisationProblem(fixture_translation_model(), RunConditions(temperature = 293.15,
        concentration = 20.0, crystals = LogNormalInitialCrystals(mass_concentration = 0.25, d43 = 12e-6, geometric_std = 1.0)))
    # Length-dependent laws need a size coordinate: no MoM.
    @test_throws ArgumentError CrystallisationProblem(CrystallisationModel(fixture_system;
        growth = CallableLengthGrowth((rate_values, rate_context, crystal_length) -> rate_values.speed * 1e-6;
            parameters = (; speed = 0.01))), fixture_seeded_conditions())
    # DQMOM has no empty-population representation.
    @test_throws ArgumentError simulate(fixture_translation_model(), RunConditions(temperature = 293.15, concentration = 20.0);
        solver = DQMOM(), saveat = [0.0, 1.0])
    # Save times must be nonnegative and strictly increasing.
    @test_throws ArgumentError simulate(fixture_translation_model(), fixture_seeded_conditions(); saveat = [10.0, 5.0])
    @test_throws ArgumentError simulate(fixture_translation_model(), fixture_seeded_conditions(); saveat = [-1.0, 5.0])
end
