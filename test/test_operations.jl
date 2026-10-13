# Reactor operations end-to-end through the model API. Every oracle is a
# hand-derived mixing/inventory identity or a closed-form lognormal moment, so
# the assertions do not re-derive solver internals. Uses runtests.jl imports and
# fixtures.jl (fixture_system: rho = 1000 kg/m³, kv = 1).

const operations_density_kv = 1000.0  # rho * kv of fixture_system
const operations_times = collect(0.0:1.0:4.0)
const operations_tank_seed = LogNormalInitialCrystals(mass_concentration = 3.0, d43 = 10e-6,
    geometric_std = 1.2)
const operations_inlet_seed = LogNormalInitialCrystals(mass_concentration = 4.0, d43 = 20e-6,
    geometric_std = 1.1)

# Raw number moments 0:5 of a lognormal seed from its mass, d43 and spread:
# d43 = exp(mu + 3.5 s^2) and m_k = m3 exp((k-3) mu + (k^2-9) s^2 / 2).
function operations_lognormal_moments(seed_description)
    log_spread = log(seed_description.geometric_std)
    log_median = log(seed_description.d43) - 3.5 * log_spread^2
    third_moment = seed_description.mass_concentration / operations_density_kv
    return [third_moment * exp((moment_order - 3) * log_median + 0.5 * (moment_order^2 - 9) * log_spread^2)
            for moment_order in 0:5]
end

# Moments of a population translated rigidly by `length_shift` (binomial expansion).
operations_shifted_moment(initial_moments, moment_order, length_shift) =
    sum(binomial(moment_order, lower_order) * initial_moments[lower_order + 1] *
        length_shift^(moment_order - lower_order) for lower_order in 0:moment_order)

# fixture_seeded_conditions fixes the seed; these runs choose their own.
operations_conditions(; concentration, crystals = nothing, kwargs...) =
    RunConditions(; temperature = 293.15, concentration, crystals, kwargs...)
# Solve over exactly the saved span.
operations_solve(model, conditions, numerical_solver; saveat) =
    solve(CrystallisationProblem(model, conditions, (first(saveat), last(saveat))), numerical_solver; saveat)

const operations_moment_solvers = (MoM(reltol = 1e-10, abstol = 1e-30),
    QMOM(reltol = 1e-10, abstol = 1e-30),
    DQMOM(nquadrature = 3, reltol = 1e-10, abstol = 1e-12))
const operations_mesh_solvers = (FiniteVol(meshsize = 24, lmax = 40e-6, reltol = 1e-10, abstol = 1e-12),
    WENO(meshsize = 24, lmax = 40e-6, reltol = 1e-10, abstol = 1e-12))

# Population representation at every saved time: raw moments 0:4 for moment
# solvers, the density for meshes.
operations_population(trajectory::CrystallisationSolution{<:MeshPopulation}) =
    trajectory.population.numberdensity
operations_population(trajectory::CrystallisationSolution) = trajectory.population.moments[1:5, :]

# Closed-form initial population: lognormal moments for moment solvers, the
# solver's own discretisation (a batch at t = 0) for meshes.
operations_seed_population(numerical_solver, seed_description) =
    numerical_solver isa CriSTool.AbstractDiscretisedSolver ?
    operations_solve(fixture_translation_model(speed = 0.0), RunConditions(temperature = 293.15,
        concentration = 2.0, crystals = seed_description), numerical_solver; saveat = [0.0]).population.numberdensity[:, 1] :
    operations_lognormal_moments(seed_description)[1:5]

@testset "Operation inputs reject unphysical hydraulics" begin
    @test_throws ArgumentError BatchOperation(volume = 0.0)
    @test_throws ArgumentError CrystallisationFeed(concentration = -1.0)
    @test_throws ArgumentError MSMPROperation(volume = 1.0, inflow = -1.0, feed = fixture_clear_feed())
    # Concentration is its own feed field; solvent_state holds only extra species.
    @test_throws ArgumentError CrystallisationFeed(concentration = 1.0, solvent_state = (; concentration = 1.0))
    # An extra tank species with no feed value or transport rule cannot be balanced.
    @test_throws ArgumentError CrystallisationProblem(fixture_translation_model(speed = 0.0),
        RunConditions(temperature = 293.15, concentration = 8.0, solvent = (; pH = 7.0),
            operation = fixture_fed_batch()), (0.0, 1.0))
end

@testset "Mesh fed-batch dilution and MSMPR washout with a seeded feed" begin
    # Zero growth: the tank is a pure mixer, exact on any mesh. With V0 = 2,
    # Q = 0.5, C0 = 8, Cf = 2, and y = C or any number-density cell:
    #   fed batch  V = 2 + Q t,  y(t) = (V0 y0 + Q t yf) / V
    #   MSMPR      V = 2,        y(t) = yf + (y0 - yf) exp(-Q t / V)
    no_growth_model = fixture_translation_model(speed = 0.0)
    fed_volume = 2.0 .+ 0.5 .* operations_times
    washout_factor = exp.(-0.5 .* operations_times ./ 2.0)
    seeded_feed = CrystallisationFeed(concentration = 2.0, crystals = operations_inlet_seed)
    for numerical_solver in operations_mesh_solvers
        tank_population = operations_seed_population(numerical_solver, operations_tank_seed)
        inlet_population = operations_seed_population(numerical_solver, operations_inlet_seed)
        @testset "$(nameof(typeof(numerical_solver)))" begin
            fed_trajectory = operations_solve(no_growth_model,
                operations_conditions(concentration = 8.0, crystals = operations_tank_seed,
                    operation = fixture_fed_batch(; inflow = 0.5, feed = seeded_feed)), numerical_solver; saveat = operations_times)
            @test fed_trajectory.retcode == ReturnCode.Success
            @test observable_values(fed_trajectory, :volume) ≈ fed_volume rtol = 1e-9
            @test fed_trajectory.concentration ≈ (16.0 .+ operations_times) ./ fed_volume rtol = 1e-9
            @test fed_trajectory.population.numberdensity ≈ hcat(((2.0 .* tank_population .+ 0.5 * elapsed_time .* inlet_population) ./
                (2.0 + 0.5 * elapsed_time) for elapsed_time in operations_times)...) rtol = 5e-7 atol = 1e-10
            # No growth means nothing crosses the size-domain boundaries.
            @test all(iszero, reactor_vars(fed_trajectory).size_boundary_upper_solid_mass_flow)
            @test all(iszero, reactor_vars(fed_trajectory).size_boundary_lower_solid_mass_flow)

            msmpr_trajectory = operations_solve(no_growth_model,
                operations_conditions(concentration = 8.0, crystals = operations_tank_seed,
                    operation = fixture_msmpr(; inflow = 0.5, feed = seeded_feed)), numerical_solver; saveat = operations_times)
            @test msmpr_trajectory.retcode == ReturnCode.Success
            @test observable_values(msmpr_trajectory, :volume) == fill(2.0, length(operations_times))
            @test msmpr_trajectory.concentration ≈ 2.0 .+ 6.0 .* washout_factor rtol = 1e-9
            @test msmpr_trajectory.population.numberdensity ≈ hcat((inlet_population .+ (tank_population .- inlet_population) .* factor
                for factor in washout_factor)...) rtol = 5e-7 atol = 1e-10
        end
    end
end

# Composite Simpson rule on [0, upper] (independent of any solver quadrature).
function operations_simpson(integrand, upper; intervals = 400)
    step_size = upper / intervals
    iszero(upper) && return zero(integrand(0.0))
    return step_size / 3 * sum((node_index in (0, intervals) ? 1 : (isodd(node_index) ? 4 : 2)) *
        integrand(node_index * step_size) for node_index in 0:intervals)
end

@testset "Moment solvers: constant growth with a seeded feed under flow" begin
    # G = 5e-7 m/s translates every crystal rigidly. A crystal that entered u
    # seconds ago has grown by G u, so with tank seed moments M and inlet seed
    # moments F (closed-form lognormal):
    #   m_k(t) = dilution(t) M_k(G t) + Q / V(t) * int_0^t survival(u) F_k(G u) du
    # fed batch: dilution = V0/V, survival = 1;  MSMPR: dilution = survival = exp(-t/tau).
    # Solid gained is solute lost, so T = C + rho kv m3 obeys the pure mixing law
    # with inlet total Tf = Cf + rho kv F3, which fixes C(t) in closed form too.
    growth_speed = 0.5
    growth_rate = growth_speed * 1e-6
    growth_model = fixture_translation_model(speed = growth_speed)
    tank_moments = operations_lognormal_moments(operations_tank_seed)
    inlet_moments = operations_lognormal_moments(operations_inlet_seed)
    seeded_feed = CrystallisationFeed(concentration = 2.0, crystals = operations_inlet_seed)
    tank_total = 8.0 + operations_density_kv * tank_moments[4]
    inlet_total = 2.0 + operations_density_kv * inlet_moments[4]
    operation_oracles = (
        (fixture_fed_batch(; inflow = 0.5, feed = seeded_feed),
            elapsed_time -> 2.0 / (2.0 + 0.5 * elapsed_time), (age, elapsed_time) -> 1.0,
            elapsed_time -> 2.0 + 0.5 * elapsed_time,
            elapsed_time -> (2.0 * tank_total + 0.5 * elapsed_time * inlet_total) / (2.0 + 0.5 * elapsed_time)),
        (fixture_msmpr(; inflow = 0.5, feed = seeded_feed),
            elapsed_time -> exp(-elapsed_time / 4.0), (age, elapsed_time) -> exp(-age / 4.0),
            elapsed_time -> 2.0,
            elapsed_time -> inlet_total + (tank_total - inlet_total) * exp(-elapsed_time / 4.0)))
    for (operation, dilution, survival, reactor_volume_value, total_solute) in operation_oracles
        expected_moments = [dilution(elapsed_time) * operations_shifted_moment(tank_moments, moment_order, growth_rate * elapsed_time) +
            0.5 / reactor_volume_value(elapsed_time) * operations_simpson(age -> survival(age, elapsed_time) *
                operations_shifted_moment(inlet_moments, moment_order, growth_rate * age), elapsed_time)
            for moment_order in 0:4, elapsed_time in operations_times]
        expected_concentration = total_solute.(operations_times) .- operations_density_kv .* expected_moments[4, :]
        for numerical_solver in operations_moment_solvers
            @testset "$(nameof(typeof(numerical_solver))) $(nameof(typeof(operation)))" begin
                trajectory = operations_solve(growth_model, operations_conditions(concentration = 8.0,
                    crystals = operations_tank_seed, operation = operation), numerical_solver; saveat = operations_times)
                @test trajectory.retcode == ReturnCode.Success
                @test trajectory.concentration ≈ expected_concentration rtol = 1e-7
                @test trajectory.d43 ≈ expected_moments[5, :] ./ expected_moments[4, :] rtol = 1e-7
                @test operations_population(trajectory) ≈
                      expected_moments[1:5, :] rtol = 1e-7
            end
        end
    end
end

@testset "Time-varying inflow, feed concentration and tracer transport" begin
    # Q(t) = t/4, Cf(t) = 2 + t, tracer feed 2, pH has zero transport, V0 = 2:
    #   V = 2 + t²/8                          -> V(2) = 2.5, V(4) = 4
    #   V C = 16 + (t² + t³/3)/4              -> C(2) = 106/15, C(4) = 19/3
    #   V tracer = 12 + t²/4                  -> tracer(2) = 5.2, tracer(4) = 4
    #   population = (2 n0 + (t²/8) nf) / V   (seeded tank and seeded inlet)
    transport_hook = (transport_problem, numerical_state, simulation_time, feed_values, inlet_flow, volume_value) -> begin
        tank_values = solvent_state(transport_problem, numerical_state)
        (; pH = zero(tank_values.pH), tracer = inlet_flow / volume_value * (feed_values.tracer - tank_values.tracer))
    end
    tracer_feed = CrystallisationFeed(concentration = simulation_time -> 2.0 + simulation_time,
        crystals = operations_inlet_seed, solvent_state = (; tracer = 2.0), transport = transport_hook)
    tracer_operation = FedBatchOperation(initial_volume = 2.0, inflow = simulation_time -> 0.25 * simulation_time,
        feed = tracer_feed)
    tracer_conditions = operations_conditions(concentration = 8.0, crystals = operations_tank_seed,
        solvent = (; pH = 7.0, tracer = 6.0), operation = tracer_operation)
    tracer_times = [0.0, 2.0, 4.0]
    tracer_volume = 2.0 .+ tracer_times .^ 2 ./ 8
    for numerical_solver in (first(operations_moment_solvers), last(operations_moment_solvers),
                             first(operations_mesh_solvers))
        @testset "$(nameof(typeof(numerical_solver)))" begin
            trajectory = operations_solve(fixture_translation_model(speed = 0.0), tracer_conditions, numerical_solver; saveat = tracer_times)
            @test reactor_vars(trajectory).volume ≈ tracer_volume rtol = 1e-9
            @test trajectory.concentration ≈ [8.0, 106.0 / 15.0, 19.0 / 3.0] rtol = 1e-8
            @test trajectory.solvent_state.pH == fill(7.0, 3)
            @test trajectory.solvent_state.tracer ≈ [6.0, 5.2, 4.0] rtol = 1e-8
            tank_population = operations_seed_population(numerical_solver, operations_tank_seed)
            inlet_population = operations_seed_population(numerical_solver, operations_inlet_seed)
            expected_population = hcat(((2.0 .* tank_population .+ elapsed_time^2 / 8 .* inlet_population) ./
                (2.0 + elapsed_time^2 / 8) for elapsed_time in tracer_times)...)
            @test operations_population(trajectory) ≈
                  expected_population rtol = 1e-6 atol = 1e-10
        end
    end
end

@testset "Mesh growth reports the solid mass leaving the size domain" begin
    # A narrow seed at 18 µm grows 3 µm past lmax = 20 µm. Total solute inventory
    # lost from the batch must equal the time-integrated upper-boundary solid
    # flow. Mesh solvers conserve C + ρkv·µ3 up to the size-boundary flows
    # exactly on any mesh, so the only error left is the trapezoid rule on the
    # sampled flow: second order in the 0.025 s spacing (measured ≤ 2.7e-6 of
    # the inventory; 1.3e-4 → 1.3e-6 → 1.3e-8 for 25 / 241 / 2401 samples).
    # The former 3ρkv·G·µ2 coupling left an O(ΔL) defect here (8e-3 / 2.5e-3
    # for FiniteVol on 24 / 48 cells, 9.5e-3 / 2.3e-3 for WENO).
    boundary_seed = GaussianInitialCrystals(mass_concentration = 4.0, d43 = 18e-6, standard_deviation = 0.5e-6)
    boundary_times = collect(range(0.0, 6.0; length = 241))
    trapezoid_integral(samples) = sum((boundary_times[index + 1] - boundary_times[index]) *
        (samples[index + 1] + samples[index]) / 2 for index in 1:(length(boundary_times) - 1))
    for solver_type in (FiniteVol, WENO)
        balance_errors = map((24, 48)) do cell_count
            mesh_solver = solver_type(meshsize = cell_count, lmin = 0.0, lmax = 20e-6, reltol = 1e-9, abstol = 1e-12)
            trajectory = operations_solve(fixture_translation_model(speed = 0.5),
                operations_conditions(concentration = 10.0, crystals = boundary_seed,
                    operation = BatchOperation(volume = 2.0)), mesh_solver; saveat = boundary_times)
            solid_mass = [operations_density_kv * mesh_solver.cell_dL *
                          sum(trajectory.population.numberdensity[:, time_index] .* mesh_solver.cell_centre .^ 3)
                          for time_index in eachindex(boundary_times)]
            total_solute = trajectory.concentration .+ solid_mass
            boundary_flows = reactor_vars(trajectory)
            @test all(iszero, boundary_flows.size_boundary_lower_solid_mass_flow)
            reactor_volume_value = 2.0
            boundary_loss = trapezoid_integral(boundary_flows.size_boundary_upper_solid_mass_flow)
            inventory_loss = reactor_volume_value * (first(total_solute) - last(total_solute))
            # Most of the seed mass must actually leave for the balance to mean anything.
            @test inventory_loss > 0.5 * reactor_volume_value * first(solid_mass)
            abs(inventory_loss - boundary_loss) / (reactor_volume_value * first(total_solute))
        end
        @test all(<(1e-5), balance_errors)
    end
end
