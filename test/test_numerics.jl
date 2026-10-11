# Numerical kernels checked against independent truth: hand-evaluated values,
# known quadratures, convergence orders and conservation identities that the
# end-to-end simulations in test_simulate.jl cannot localise.

# Minimal problem carrying only what the kernels read: solubility (C_sat = 10),
# solver mesh, and (for CNT) the lysozyme molecular volume at 293.15 K.
const numerics_system = LysozymeSystem(saturation_model = ConstantSolubility(10.0))
numerics_problem(; solver = FiniteVol(meshsize = 3, lmax = 3e-6), concentration = 20.0) =
    CrystallisationProblem(CrystallisationModel(numerics_system;
            growth = EmpiricalGrowth(growth_coefficient = 1e-9 / 60, growth_order = 1.0)),
        RunConditions(temperature = 293.15, concentration = concentration); solver)
# Population block of `population_count` zeros followed by concentration C;
# with C_sat = 10 the supersaturation is C / 10.
numerics_state(population_count, concentration) = [zeros(population_count); concentration]

@testset "WENO reconstruction" begin
    # A linear profile is reproduced exactly at the face between cells 3 and 4.
    @test CriSTool.weno_flux([2.0 + 0.4 * centre for centre in -2:2], 3) ≈ 2.2 atol = 2e-15

    # Smooth data: exact cell averages of 2 + sin(L). Fifth-order WENO must
    # shrink the face error by ~2^5 = 32 per halving (assert > 20).
    reconstruction_errors = map((0.2, 0.1, 0.05)) do cell_width
        cell_centres = [1.0 + stencil_offset * cell_width for stencil_offset in -2:2]
        cell_averages = [2.0 + (cos(centre - cell_width / 2) - cos(centre + cell_width / 2)) / cell_width
                         for centre in cell_centres]
        abs(CriSTool.weno_flux(cell_averages, 3) - (2.0 + sin(1.0 + cell_width / 2)))
    end
    @test reconstruction_errors[1] / reconstruction_errors[2] > 20.0
    @test reconstruction_errors[2] / reconstruction_errors[3] > 20.0
    @test reconstruction_errors[3] < 1e-7

    # Inlet ghosts [3, 3] (nucleation flux 6 / speed 2) and stencil [3, 3, 1, 2, 4]:
    # the hand-evaluated Jiang-Shu left trace is 1.3155079987669 (flux ×2).
    positive_flux = zeros(5)
    padded_density = zeros(8)
    CriSTool._fill_positive_weno_flux!(positive_flux, [1.0, 2.0, 4.0, 5.0], 2.0, 6.0, padded_density)
    @test padded_density[1:2] == [3.0, 3.0]
    @test positive_flux[1] == 6.0
    @test positive_flux[2] ≈ 2.631015997533807 atol = 2e-14
    @test positive_flux[end] == 11.0   # outlet: 2 × linear trace 5.5

    # Signed transport: exact cell averages of 1 + 0.2 sin(2πL) on [0,1]; both
    # the left and mirrored right traces must reproduce the face value.
    meshsize = 64
    cell_width = 1.0 / meshsize
    average_factor = sinpi(cell_width) / (pi * cell_width)
    smooth_density = [1.0 + 0.2 * sin(2pi * (index - 0.5) * cell_width) * average_factor for index in 1:meshsize]
    smooth_padded = [fill(smooth_density[1], 2); smooth_density; zeros(2)]
    exact_face_value = 1.0 + 0.2 * sin(2pi * 31 * cell_width)
    @test CriSTool._weno_flux_left(smooth_padded, 33) ≈ exact_face_value atol = 2e-6
    @test CriSTool._weno_flux_right(smooth_padded, 34) ≈ exact_face_value atol = 2e-6

    # Negative speed on the linear profile 1 + 0.1 i: face 16 sits between
    # cells 15 and 16, value 2.55. Lower face is open outflow G·n₁; the upper
    # face has no inflow.
    linear_density = [1.0 + 0.1 * index for index in 1:32]
    signed_flux = zeros(33)
    CriSTool._fill_signed_weno_flux!(signed_flux, linear_density, -2.0, 0.0, zeros(36))
    @test signed_flux[1] == -2.0 * 1.1
    @test signed_flux[16] ≈ -2.0 * 2.55
    @test signed_flux[end] == 0.0

    # Face-wise rates +1 | -1: nucleation inlet 3, and the face where the
    # averaged rate is exactly zero carries no flux.
    split_flux = zeros(25)
    CriSTool._fill_signed_weno_flux!(split_flux, collect(range(0.5, 2.0; length = 24)),
        vcat(fill(1.0, 12), fill(-1.0, 12)), 3.0, zeros(28))
    @test split_flux[1] == 3.0
    @test split_flux[13] == 0.0
    @test split_flux[end] == 0.0

    # Positivity limiter: a negative trace h beside a nonempty upwind cell u
    # blends with θ = -h/(u - h), and θu + (1 - θ)h = 0. On far-tail values the
    # state and its ForwardDiff partials must stay exactly zero (the blend's
    # quotient rule would square u - h ≈ 1e-162 to zero and return NaN).
    @test CriSTool._weno_nonnegative_state(-0.5, 2.0) == 0.0
    @test CriSTool._weno_nonnegative_state(0.7, 2.0) == 0.7
    tail_trace = CriSTool._weno_nonnegative_state(ForwardDiff.Dual(-1e-164, 1.0, 0.0),
                                                  ForwardDiff.Dual(1e-162, 0.0, 0.0))
    @test ForwardDiff.value(tail_trace) == 0.0
    @test collect(ForwardDiff.partials(tail_trace)) == [0.0, 0.0]
end

@testset "FV limiter and reconstruction" begin
    @test CriSTool.fluxlimiter_ospre(1.0) == 1.0
    @test CriSTool.fluxlimiter_ospre(0.0) == 0.0
    @test CriSTool.fluxlimiter_ospre(-2.0) == 0.0
    # A limited linear profile is reproduced exactly in both slope directions:
    # unit speed, inlet flux equal to the extrapolated ghost.
    reconstructed_flux = zeros(5)
    CriSTool._fill_fv_scalar_flux!(reconstructed_flux, [4.0, 3.0, 2.0, 1.0], 1.0, 4.5)
    @test reconstructed_flux ≈ [4.5, 3.5, 2.5, 1.5, 0.5]
    CriSTool._fill_fv_scalar_flux!(reconstructed_flux, [1.0, 2.0, 3.0, 4.0], 1.0, 0.5)
    @test reconstructed_flux ≈ [0.5, 1.5, 2.5, 3.5, 4.5]
    # At a local peak the reconstruction falls back to the cell value.
    CriSTool._fill_fv_scalar_flux!(reconstructed_flux, [1.0, 3.0, 2.0, 1.0], 1.0, 0.0)
    @test reconstructed_flux[3] == 3.0
end

@testset "QMOM moment inversion" begin
    qmom_solver = QMOM(nquadrature = 3, coordinate_scale = 1e-6, realizability_tolerance = 1e-9)
    # Moments of independently chosen atomic measures; their hand values are
    # asserted first so the inversion is checked against known quadratures.
    atomic_moments(nodes, weights) = [sum(weights .* nodes .^ order) for order in 0:5]
    for (atom_nodes, atom_weights, hand_moments, expected_status) in (
            ([0.8e-6], [2.5e12], [2.5e12, 2.0e6, 1.6, 1.28e-6, 1.024e-12, 8.192e-19], :deflated),
            ([0.5e-6, 1.5e-6], [2.0e12, 3.0e12], [5.0e12, 5.5e6, 7.25, 1.0375e-5, 1.53125e-11, 2.284375e-17], :deflated),
            ([0.4e-6, 1.0e-6, 1.8e-6], [1.2e12, 2.0e12, 0.8e12],
             [4.0e12, 3.92e6, 4.784, 6.7424e-6, 1.04288e-11, 1.7128832e-17], :ok))
        @test atomic_moments(atom_nodes, atom_weights) ≈ hand_moments
        rule = invert_moments(hand_moments, qmom_solver)
        node_order = sortperm(rule.nodes)
        @test rule.active_nodes == length(atom_nodes)
        @test rule.nodes[node_order] ≈ atom_nodes rtol = 1e-7
        @test rule.weights[node_order] ≈ atom_weights rtol = 1e-7
        @test rule.diagnostics.status == expected_status
    end
    @test invert_moments(zeros(6), qmom_solver).diagnostics.status == :empty
    zero_size_rule = invert_moments([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], qmom_solver)
    @test (zero_size_rule.nodes[1], zero_size_rule.weights[1]) == (0.0, 2.0)
    # Unrealisable or malformed moment sets are rejected, not silently clipped.
    @test_throws ArgumentError invert_moments([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], qmom_solver)
    @test_throws ArgumentError invert_moments([1.0, 0.0, -1.0, 0.0, 0.0, 0.0], qmom_solver)
    @test_throws ArgumentError invert_moments([1.0, NaN, 0.0, 0.0, 0.0, 0.0], qmom_solver)
    @test_throws ArgumentError invert_moments([1.0, -0.5e-6, 0.25e-12, -0.125e-18, 0.0625e-24, -0.03125e-30], qmom_solver)
    @test_throws ArgumentError invert_moments(zeros(5), qmom_solver)
end

@testset "Moment sources on quadratures" begin
    # Two nodes (0.5, 1.5 µm), weights (2, 3). Constant kernel K = 1: event
    # count ½(2+3)² = 12.5 removes one particle each and conserves M₃.
    source_rule = QMOMQuadrature([0.5e-6, 1.5e-6], [2.0, 3.0])
    aggregation_source = aggregation_moment_source(ConstantAggregation(), [0.0], source_rule, 3)
    @test aggregation_source[1] ≈ -12.5
    @test aggregation_source[4] ≈ 0.0 atol = 1e-24
    # Unit-frequency binary breakage: +1 particle per parent (5), M₃ conserved.
    breakage_source = breakage_moment_source(UniformBreakage(), [0.0, 0.0], source_rule, 3)
    @test breakage_source[1] ≈ 5.0
    @test breakage_source[4] ≈ 0.0 atol = 1e-24

    # Constant growth on M_k = 2·(1e-6)^k: dM_k/dt = k G M_{k-1}.
    growth_source = CriSTool._qmom_scalar_growth_moment_source(
        [2.0 * 1e-6^order for order in 0:5], 3.0e-9, 0.0, 5)
    @test growth_source[1:4] ≈ [0.0, 6.0e-9, 12.0e-15, 18.0e-21]

    # DQMOM normalised weights (0.25, 0.75), pair scale 1e12: total collision
    # rate W/2 with K = 1; unit-frequency breakage adds one daughter.
    dqmom_aggregation = CriSTool._aggregation_moment_source(ConstantAggregation(), [0.0], [0.5e-6, 1.5e-6],
        [0.25, 0.75], 3; shape_factor = 0.81, pair_weight_scale = 1e12)
    @test dqmom_aggregation[1] ≈ -0.5e12
    @test dqmom_aggregation[4] ≈ 0.0 atol = 1e-20
    dqmom_breakage = CriSTool._breakage_moment_source(EmpiricalBreakage(), [1.0, 0.0],
        [0.5e-6, 1.5e-6], [0.25, 0.75], 3)
    @test dqmom_breakage[1] ≈ 1.0
    @test dqmom_breakage[4] == 0.0

    # DQMOM projection for nodes (0.5, 1.5): rows k = 0..3 of
    # [(1-k) Lᵢᵏ | k Lᵢᵏ⁻¹], evaluated by hand.
    @test CriSTool._dqmom_projection_matrix([0.5, 1.5]) ≈
          [1.0 1.0 0.0 0.0; 0.0 0.0 1.0 1.0; -0.25 -2.25 1.0 3.0; -0.25 -6.75 0.75 6.75]
end

@testset "FV aggregation and breakage kernels" begin
    # Three unit cells on [0, 3]: centres 0.5, 1.5, 2.5; faces cubed 0, 1, 8, 27.
    kernel_problem(meshsize = 3, lmax = 3.0) = numerics_problem(; solver = FiniteVol(; meshsize, lmax), concentration = 1.0)
    kernel_solver = kernel_problem().solver

    # One parent in cell 1, K = 1: hand rate [-27/52, 1/52, 0] (fixed-pivot
    # split of L³ = 0.25 between pivots 0.125 and 3.375), scaled by each
    # kernel's value at (0.5, 0.5). Number falls by ½K, volume conserved.
    for (aggregation_law, kernel_scale) in ((ConstantAggregation(), 1.0), (SizeSumAggregation(), 1.0),
            (VolumeSumAggregation(), 0.2025), (AverageAggregation(), 0.5))
        single_parent_rate = CriSTool.aggregationrate(aggregation_law, [0.0], kernel_problem(), [1.0, 0.0, 0.0, 1.0], 0.0)
        @test single_parent_rate ≈ kernel_scale .* [-27 / 52, 1 / 52, 0.0]
        @test sum(kernel_solver.cell_centre .^ 3 .* single_parent_rate) ≈ 0.0 atol = 1e-12
    end
    # Populations (1, 2, 0): hand-summed collision rates per kernel.
    for (aggregation_law, event_rate) in ((ConstantAggregation(), 4.5), (SizeSumAggregation(), 10.5),
            (VolumeSumAggregation(), 16.70625), (AverageAggregation(), 5.25))
        two_cell_rate = CriSTool.aggregationrate(aggregation_law, [0.0], kernel_problem(), [1.0, 2.0, 0.0, 1.0], 0.0)
        @test sum(two_cell_rate) ≈ -event_rate
        @test sum(kernel_solver.cell_centre .^ 3 .* two_cell_rate) ≈ 0.0 atol = 1e-12
    end
    # 2.5³ + 2.5³ = 31.25 > 27: the product leaves the domain, not the last pivot.
    @test CriSTool.aggregationrate(ConstantAggregation(), [0.0], kernel_problem(), [0.0, 0.0, 1.0, 1.0], 0.0) ≈ [0.0, 0.0, -1.0]

    # Parent at 1.5 with unit frequency, uniform-in-volume daughters:
    # 2ΔV/V per overlapped cell, minus the parent → [16/27, 11/27, 0].
    middle_parent_state = [0.0, 1.0, 0.0, 1.0]
    @test CriSTool.breakagerate(EmpiricalBreakage(), [1.0, 0.0], kernel_problem(), middle_parent_state, 0.0) ≈
          [16 / 27, 11 / 27, 0.0]
    @test CriSTool.breakagerate(UniformBreakage(), [0.0, 0.0], kernel_problem(), middle_parent_state, 0.0) ≈
          [16 / 27, 11 / 27, 0.0]
    # Exactly one net daughter per event wherever the parent sits.
    for parent_index in 1:3
        parent_state = zeros(4)
        parent_state[parent_index] = 1.0
        @test sum(CriSTool.breakagerate(EmpiricalBreakage(), [1.0, 0.0], kernel_problem(), parent_state, 0.0)) ≈ 1.0
    end
    # Uniform population, constant selection: continuous M₃ source is exactly
    # zero, so the discrete M₃ error must converge (better than first order).
    volume_moment_errors = map((20, 40, 80)) do meshsize
        refined_solver = kernel_problem(meshsize, 1.0).solver
        refined_rate = CriSTool.breakagerate(EmpiricalBreakage(), [1.0, 0.0],
            kernel_problem(meshsize, 1.0), [ones(meshsize); 1.0], 0.0)
        abs(sum(refined_solver.cell_centre .^ 3 .* refined_rate) * refined_solver.cell_dL)
    end
    @test volume_moment_errors[2] < volume_moment_errors[1] / 3
    @test volume_moment_errors[3] < volume_moment_errors[2] / 3
end

@testset "Kinetic rates at hand-evaluated points" begin
    rate_problem = numerics_problem()
    supersaturated_state = numerics_state(3, 15.0)   # S = 1.5
    undersaturated_state = numerics_state(3, 5.0)    # S = 0.5
    # 10^10 (1.5 - 1)^2 and (1e-9/60)(0.5)^3.
    @test CriSTool.nucleationrate(EmpiricalNucleation(), [10.0, 2.0], rate_problem, supersaturated_state, 0.0) ≈ 2.5e9
    @test CriSTool.growthrate(EmpiricalGrowth(), [1e-9 / 60, 3.0], rate_problem, supersaturated_state, 0.0) ≈ 1e-9 / 480
    @test CriSTool.growthrate(EmpiricalGrowth(), [1e-9 / 60, 3.0], rate_problem, undersaturated_state, 0.0) == 0.0
    # CNT without the leading S factor differs by exactly S = 1.5.
    @test CriSTool.nucleationrate(CNTNucleationNoSupersaturation(), [38.0, 0.0007], rate_problem, supersaturated_state, 0.0) * 1.5 ≈
          CriSTool.nucleationrate(CNTNucleation(), [38.0, 0.0007], rate_problem, supersaturated_state, 0.0)
    # Dissolution with zero activation energy: -(2e-9/60)(1 - 0.5)^1.5; inside
    # the deadband (S = 0.9995) the rate is exactly zero.
    @test CriSTool.dissolutionrate(ArrheniusDissolution(), [2e-9 / 60, 0.0, 1.5], rate_problem, undersaturated_state, 0.0) ≈
          -(2e-9 / 60) * sqrt(0.125)
    @test CriSTool.dissolutionrate(ArrheniusDissolution(), [2e-9 / 60, 0.0, 1.5], rate_problem,
        numerics_state(3, 9.995), 0.0) == 0.0
    # Arrhenius growth with zero activation energy: 10^-8 (1.5 - 1)^2.
    @test CriSTool.growthrate(ArrheniusGrowth(), [-8.0, 0.0, 2.0], rate_problem, supersaturated_state, 0.0) ≈ 2.5e-9
    # Combined nucleation sums its components: 10^10·0.5² + 10^9·0.5¹.
    combined_nucleation = CombinedNucleation(EmpiricalNucleation(), EmpiricalNucleation())
    @test CriSTool.nucleationrate(combined_nucleation, [10.0, 2.0, 9.0, 1.0], rate_problem,
        supersaturated_state, 0.0) ≈ 3.0e9
end

@testset "Kinetic parameter schema" begin
    # paramaxis is the only parameter declaration; an empty axis counts zero.
    @test CriSTool._parameter_count(NoDissolution()) == 0
    @test CriSTool._parameter_count(CombinedNucleation(CNTNucleation(), SecondaryNucleation())) == 5
    bound_combination = CombinedNucleation(CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energy = 6e-4),
        EmpiricalNucleation(log10_nucleation_prefactor = 9.0, nucleation_order = 1.0))
    @test bound_combination.parameters ==
          (primary = (ln_nucleation_prefactor = 38.0, surface_energy = 6e-4),
           secondary = (log10_nucleation_prefactor = 9.0, nucleation_order = 1.0))
    # Partial or misspelled keywords name both the missing and unknown entries.
    partial_binding_error = try
        CNTNucleation(ln_nucleation_prefactor = 38.0, surface_energi = 6e-4)
    catch binding_error
        binding_error
    end
    @test partial_binding_error isa ArgumentError
    @test occursin("missing (:surface_energy,)", partial_binding_error.msg)
    @test occursin("unknown (:surface_energi,)", partial_binding_error.msg)
    # Configuration keywords stay on the law; parameter keywords bind it.
    @test EmpiricalLengthGrowth(Lref = 2e-6).Lref == 2e-6
end

@eval begin
    # Custom length dissolution counting scalar-fallback evaluations.
    mutable struct NumericsCountingLengthDissolution <: CriSTool.AbstractLengthDissolutionFunction
        evaluations::Base.RefValue{Int}
    end
    function CriSTool.dissolutionrate_at_length(counting_law::NumericsCountingLengthDissolution, parameters,
            problem::CriSTool.CrystallisationProblem, state, time, crystal_length::Real)
        counting_law.evaluations[] += 1
        return -parameters[1] * crystal_length
    end
end

@testset "Length-dependent rate kernels" begin
    length_mesh = [0.5e-6, 1.0e-6, 2.0e-6]
    rate_problem = numerics_problem()
    supersaturated_state = numerics_state(3, 15.0)
    undersaturated_state = numerics_state(3, 5.0)
    # 3e-8 · 0.5² · (1 + 0.5 L/1µm)²  at L = 0.5, 1, 2 µm.
    length_growth = EmpiricalLengthGrowth(Lref = 1e-6)
    growth_destination = zeros(3)
    CriSTool.growthrate!(growth_destination, length_growth, [3e-8, 2.0, 0.5, 2.0], rate_problem,
        supersaturated_state, 0.0, length_mesh)
    @test growth_destination ≈ [1.171875e-8, 1.6875e-8, 3.0e-8] rtol = 1e-14
    @test CriSTool.growthrate_at_length(length_growth, [3e-8, 2.0, 0.5, 2.0], rate_problem,
        supersaturated_state, 0.0, 1.0e-6) ≈ 1.6875e-8 rtol = 1e-14
    CriSTool.growthrate!(growth_destination, length_growth, [3e-8, 2.0, 0.5, 2.0], rate_problem,
        numerics_state(3, 10.009), 0.0, length_mesh)
    @test all(iszero, growth_destination)   # inside the S < 1.001 deadband
    # A negative size factor 1 - 0.5·2.5 < 0 is a domain error, not NaN.
    @test_throws DomainError CriSTool.growthrate!(growth_destination, length_growth, [3e-8, 2.0, -0.5, 2.0],
        rate_problem, supersaturated_state, 0.0, [2.5e-6, 2.6e-6, 2.7e-6])

    # -6e-8 · 0.5 · (1 + 0.5 L/1µm)²; undersaturated growth contributes zero.
    length_dissolution = ArrheniusLengthDissolution(Lref = 1e-6)
    net_destination = zeros(3)
    CriSTool.net_growth_rate!(net_destination, length_growth, [3e-8, 2.0, 0.5, 2.0], length_dissolution,
        [6e-8, 0.0, 1.0, 0.5, 2.0], rate_problem, undersaturated_state, 0.0, length_mesh)
    @test net_destination ≈ [-4.6875e-8, -6.75e-8, -1.2e-7] rtol = 1e-14

    # Custom scalar fallback: -0.03 L, evaluated once per mesh point.
    counting_law = NumericsCountingLengthDissolution(Ref(0))
    CriSTool.net_growth_rate!(net_destination, EmpiricalGrowth(), [3e-8, 2.0], counting_law, [0.03],
        rate_problem, undersaturated_state, 0.0, length_mesh)
    @test net_destination ≈ [-1.5e-8, -3.0e-8, -6.0e-8] rtol = 1e-14
    @test counting_law.evaluations[] == 3
end

# Boundary-flow kernels: nucleation proportional to (1 + t/10), constant
# scalar speed, and a per-cell length law with context dependence.
struct NumericsBoundaryNucleation <: CriSTool.AbstractNucleationFunction end
CriSTool.paramaxis(::NumericsBoundaryNucleation) = CriSTool.ComponentArrays.Axis(boundary_birth = 1)
CriSTool.nucleationrate(::NumericsBoundaryNucleation, parameters, problem, state, time) = parameters[1] * (1 + time / 10)

struct NumericsBoundaryScalarGrowth <: CriSTool.AbstractScalarGrowthFunction end
CriSTool.paramaxis(::NumericsBoundaryScalarGrowth) = CriSTool.ComponentArrays.Axis(boundary_speed = 1)
CriSTool.growthrate(::NumericsBoundaryScalarGrowth, parameters, problem, state, time) = parameters[1]

struct NumericsBoundaryLengthGrowth <: CriSTool.AbstractLengthGrowthFunction
    boundary_cell_rate_count::Int
end
CriSTool.paramaxis(law::NumericsBoundaryLengthGrowth) =
    CriSTool.ComponentArrays.Axis(boundary_cell_rates = 1:law.boundary_cell_rate_count)
function CriSTool.growthrate!(destination::AbstractVector, ::NumericsBoundaryLengthGrowth, parameters, problem,
        state, time, mesh::AbstractVector)
    cell_count = length(mesh)
    concentration = CriSTool.solvent_state(problem, state).concentration
    volume = CriSTool.reactor_volume(problem, state)
    for cell_index in eachindex(mesh, destination)
        destination[cell_index] = parameters[cell_index] + parameters[cell_count + cell_index] * time +
                                  1e-9 * concentration + 1e-10 * volume + 1e-3 * mesh[cell_index]
    end
    return destination
end

# Four cells on [1, 5] (faces 1..5), rho·kv = 1500·0.5 = 750, batch volume 2.5.
const numerics_boundary_system = CrystallisationSystem(saturation_model = ConstantSolubility(1.0),
    crystal_density = 1500.0, volume_shape_factor = 0.5)
function numerics_boundary_problem(growth_law, growth_parameters::NamedTuple;
        solver = WENO(meshsize = 4, lmin = 1.0, lmax = 5.0), operation = BatchOperation(volume = 2.5))
    boundary_model = CrystallisationModel(numerics_boundary_system;
        nucleation = KineticModel(NumericsBoundaryNucleation(); parameters = (; boundary_birth = 8.0)),
        growth = KineticModel(growth_law; parameters = growth_parameters))
    return CrystallisationProblem(boundary_model, RunConditions(; temperature = 293.15, concentration = 5.0, operation); solver)
end
numerics_saved_solution(problem, saved_times, saved_states; parameters = CriSTool.kinetic_parameter_vector(problem)) =
    (; t = saved_times, u = saved_states, prob = (; p = parameters))
numerics_boundary_flows(problem, saved_times, saved_states; kwargs...) = CriSTool._mesh_boundary_flow_state(
    problem.solver, problem, numerics_saved_solution(problem, saved_times, saved_states; kwargs...))

@testset "Mesh boundary mass flows" begin
    # Density [2,3,4,5]: upper linear trace 5.5; lower face L³ = 1, upper 125.
    # Mass flows are rho·kv·V·L³·flux = 1875 · L³ · flux.
    scalar_problem = numerics_boundary_problem(NumericsBoundaryScalarGrowth(), (; boundary_speed = 0.25))
    scalar_flows = numerics_boundary_flows(scalar_problem, [0.0, 2.0], [[2.0, 3.0, 4.0, 5.0, 5.0], [2.0, 3.0, 4.0, 5.0, 7.0]])
    @test scalar_flows.size_boundary_lower_solid_mass_flow ≈ [-15_000.0, -18_000.0]   # birth 8, 9.6
    @test scalar_flows.size_boundary_upper_solid_mass_flow ≈ fill(322_265.625, 2)     # 0.25·5.5·125·1875
    # Trace 1 + 0.5(1 - 4) < 0 is clipped: no upper outflow.
    @test numerics_boundary_flows(scalar_problem, [0.0], [[2.0, 3.0, 4.0, 1.0, 5.0]]).size_boundary_upper_solid_mass_flow == [0.0]
    # Dissolving (G = -0.5): lower outflow 0.5·2·1·1875, no upper inflow.
    dissolving_flows = numerics_boundary_flows(numerics_boundary_problem(NumericsBoundaryScalarGrowth(), (; boundary_speed = -0.5)),
        [0.0], [[2.0, 3.0, 4.0, 5.0, 5.0]])
    @test dissolving_flows.size_boundary_lower_solid_mass_flow == [1875.0]
    @test dissolving_flows.size_boundary_upper_solid_mass_flow == [0.0]

    # Fed batch: the flows must use each saved volume (2 then 3 m³), not a
    # template value. Hand rates: lower ≈ 1.5e-3 > 0 (birth 8, 9.6), upper
    # 4.5000053e-3 and 4.4999043e-3 with traces 5.5 and 6.5.
    length_parameters = [-0.2e-6, 0.1e-6, -0.1e-6, 0.3e-6, 0.2e-6, 0.0, 0.0, -0.2e-6]
    fed_problem = numerics_boundary_problem(NumericsBoundaryLengthGrowth(8), (; boundary_cell_rates = length_parameters);
        operation = FedBatchOperation(initial_volume = 2.0, inflow = 0.5, feed = CrystallisationFeed(concentration = 4.0)))
    fed_flows = numerics_boundary_flows(fed_problem, [0.0, 2.0], [[2.0, 3.0, 4.0, 5.0, 2.0, 5.0], [3.0, 4.0, 5.0, 6.0, 3.0, 4.0]])
    @test fed_flows.size_boundary_lower_solid_mass_flow ≈ [-12_000.0, -21_600.0]
    @test fed_flows.size_boundary_upper_solid_mass_flow ≈ [4640.9397375, 8226.387548437498]

    # Candidate kinetics override the template for both FV and WENO.
    for mesh_solver in (FiniteVol(meshsize = 4, lmin = 1.0, lmax = 5.0), WENO(meshsize = 4, lmin = 1.0, lmax = 5.0))
        candidate_template = numerics_boundary_problem(NumericsBoundaryScalarGrowth(), (; boundary_speed = 0.25); solver = mesh_solver)
        growing_flows = numerics_boundary_flows(candidate_template, [0.0], [[2.0, 3.0, 4.0, 5.0, 5.0]];
            parameters = (; nucl = [16.0], gr = [0.5], diss = Float64[]))
        @test growing_flows.size_boundary_lower_solid_mass_flow == [-30_000.0]    # 16·1875
        @test growing_flows.size_boundary_upper_solid_mass_flow == [644_531.25]   # 0.5·5.5·125·1875
        shrinking_flows = numerics_boundary_flows(candidate_template, [0.0], [[2.0, 3.0, 4.0, 5.0, 5.0]];
            parameters = (; nucl = [16.0], gr = [-0.5], diss = Float64[]))
        @test shrinking_flows.size_boundary_lower_solid_mass_flow == [1875.0]
        @test shrinking_flows.size_boundary_upper_solid_mass_flow == [0.0]
    end
end

@testset "Mesh size metrics" begin
    # Centres 1, 3, 5 µm (width 2 µm); columns are two saved densities.
    # Hand moments: column 1 → μ1/μ0 = 5/3, μ3/μ2 = 29/11, μ4/μ3 = 83/29.
    mesh_sizes = CriSTool._momentsizes(LinRange(1.0e-6, 5.0e-6, 3), [2.0 1.0; 1.0 3.0; 0.0 2.0])
    @test mesh_sizes.moment2 ≈ [22.0e-18, 156.0e-18] rtol = 1e-14
    @test mesh_sizes.d10 ≈ [5 / 3, 10 / 3] .* 1e-6 rtol = 1e-14
    @test mesh_sizes.d32 ≈ [29 / 11, 166 / 39] .* 1e-6 rtol = 1e-14
    @test mesh_sizes.d43 ≈ [83 / 29, 4.5] .* 1e-6 rtol = 1e-14
end
