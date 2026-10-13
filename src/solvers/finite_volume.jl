"""
    _mesh_growth_layout(problem) -> Val{Bool}

`Val(true)` when the growth or dissolution law depends on crystal length, so
mesh transport needs a per-cell net growth-rate field; `Val(false)` when one
scalar rate drives every cell. Evaluated once per ODE construction.
"""
_mesh_growth_layout(problem::CrystallisationProblem) = Val(_uses_length_rates(problem))

"""
    _mesh_net_growth(growth_layout, growth_rate_cache, parameters, problem, state, time)

Net (growth plus dissolution) rate that drives mesh transport: a scalar for
`Val(false)`, or the per-cell field written into `growth_rate_cache` (a `DiffCache` or a
plain vector) for `Val(true)`. Without a dissolution law the scalar path calls `growthrate`
directly.
"""
@inline _mesh_net_growth(::Val{false}, growth_rate_cache, parameters, problem, state, time) =
    net_growth_rate(problem.model.growth.law, parameters.gr,
                    problem.model.dissolution.law, parameters.diss,
                    problem, state, time)

@inline _mesh_net_growth(::Val{true}, growth_rate_cache, parameters, problem, state, time) =
    net_growth_rate!(get_tmp(growth_rate_cache, state), problem.model.growth.law, parameters.gr,
                     problem.model.dissolution.law, parameters.diss,
                     problem, state, time, problem.solver.cell_centre)

"""Fill the legacy high-resolution FV scalar-growth flux in place."""
@inline function _fill_fv_scalar_flux!(flux::AbstractVector,
                                       numberdensity::AbstractVector,
                                       scalar_growth_rate,
                                       nucleation_rate)
    if scalar_growth_rate > zero(scalar_growth_rate)
        # Preserve the established high-resolution lower-boundary/nucleation
        # convention and OSPRE limiter at every interior face.
        flux[1] = nucleation_rate
        flux[2] = scalar_growth_rate * 0.5 * (numberdensity[1] + numberdensity[2])

        @inbounds for cell_index in 3:length(numberdensity)
            gradient_upstream = numberdensity[cell_index - 1] - numberdensity[cell_index - 2]
            gradient_downstream = numberdensity[cell_index] - numberdensity[cell_index - 1]
            gradient_floor = eps(eltype(numberdensity))
            gradient_denominator = abs(gradient_downstream) > gradient_floor ?
                gradient_downstream :
                (gradient_downstream < zero(gradient_downstream) ? -gradient_floor : gradient_floor)
            gradient_ratio = gradient_upstream / gradient_denominator
            flux[cell_index] = scalar_growth_rate *
                               (numberdensity[cell_index - 1] +
                                0.5 * fluxlimiter_ospre(gradient_ratio) *
                                gradient_downstream)
        end

        flux[length(numberdensity) + 1] = scalar_growth_rate *
                                          (numberdensity[end] +
                                           0.5 * (numberdensity[end] -
                                                  numberdensity[end - 1]))
    elseif scalar_growth_rate < zero(scalar_growth_rate)
        # Negative growth is outflow at L=lmin and zero-inflow at lmax.
        @inbounds begin
            flux[1] = scalar_growth_rate * numberdensity[1]
            for face_index in 2:length(numberdensity)
                flux[face_index] = scalar_growth_rate * numberdensity[face_index]
            end
            flux[end] = zero(scalar_growth_rate)
        end
    else
        fill!(flux, zero(scalar_growth_rate))
    end
    return flux
end

"""
    _fill_fv_flux!(flux, numberdensity, net_growth, nucleation_rate)

FV face fluxes for the transport rate shape: the high-resolution OSPRE flux
for a scalar rate, first-order signed upwinding for a per-cell rate field.
"""
@inline _fill_fv_flux!(flux, numberdensity, scalar_growth_rate::Number, nucleation_rate) =
    _fill_fv_scalar_flux!(flux, numberdensity, scalar_growth_rate, nucleation_rate)
@inline _fill_fv_flux!(flux, numberdensity, net_growth_rates::AbstractVector, nucleation_rate) =
    _fill_signed_first_order_flux!(flux, numberdensity, net_growth_rates, nucleation_rate)

"""
    _mesh_boundary_fluxes(solver, flux_cache, numberdensity, net_growth, nucleation_rate)
        -> (lower, upper)

Lower- and upper-face number fluxes that the solver's RHS produces. The FV
flux is filled into `flux_cache` and read at both ends; WENO evaluates the two
boundary faces without reconstructing the interior (`flux_cache` unused).
"""
function _mesh_boundary_fluxes(solver::FiniteVol, flux_cache, numberdensity, net_growth,
                               nucleation_rate)
    _fill_fv_flux!(flux_cache, numberdensity, net_growth, nucleation_rate)
    return first(flux_cache), last(flux_cache)
end

"""Boundary solid-mass flow samples using the exact boundary fluxes of the RHS.

The signed lower and upper flows are positive out of the size domain and use
kg/s. Batch operation volume is the explicit reactor volume; the default
1 m³ value is its documented unit-volume reference.
"""
function _mesh_boundary_flow_state(solver::AbstractDiscretisedSolver, problem, ode_solution)
    saved_times = ode_solution.t
    sample_state = first(ode_solution.u)
    lower_mass_flow = Vector{typeof(zero(eltype(sample_state)))}(undef, length(saved_times))
    upper_mass_flow = similar(lower_mass_flow)
    flux_cache = similar(sample_state, solver.meshsize + 1)
    # Length-dependent laws get the full mesh, state and time context; custom
    # laws need not support pointwise endpoint calls.
    growth_rate_cache = similar(sample_state, solver.meshsize)
    growth_layout = _mesh_growth_layout(problem)
    # Prepared inference remakes p without replacing the physical template.
    parameters = ode_solution.prob.p
    lower_length = first(solver.cell_face)
    upper_length = last(solver.cell_face)
    mass_scale = problem.model.system.crystal_density * problem.model.system.volume_shape_factor

    for time_index in eachindex(saved_times)
        numerical_state = ode_solution.u[time_index]
        simulation_time = saved_times[time_index]
        nucleation_rate_value = nucleationrate(problem.model.nucleation.law,
                                               parameters.nucl, problem,
                                               numerical_state, simulation_time)
        net_growth = _mesh_net_growth(growth_layout, growth_rate_cache, parameters,
                                      problem, numerical_state, simulation_time)
        lower_boundary_flux, upper_boundary_flux =
            _mesh_boundary_fluxes(solver, flux_cache, crystal_state(problem, numerical_state),
                                  net_growth, nucleation_rate_value)

        reactor_volume_value = reactor_volume(problem, numerical_state)
        # The RHS flux buffer holds face values at the state element type;
        # convert before applying the physical mass-flow scaling.
        lower_boundary_flux = convert(eltype(sample_state), lower_boundary_flux)
        upper_boundary_flux = convert(eltype(sample_state), upper_boundary_flux)
        lower_mass_flow[time_index] = -lower_boundary_flux * lower_length^3 *
                                      mass_scale * reactor_volume_value
        upper_mass_flow[time_index] = upper_boundary_flux * upper_length^3 *
                                      mass_scale * reactor_volume_value
    end

    return (; size_boundary_lower_solid_mass_flow = lower_mass_flow,
              size_boundary_upper_solid_mass_flow = upper_mass_flow)
end

"""Physical mesh solution shared by the `FiniteVol` and `WENO` solvers."""
function _wrap_solution(solver::AbstractDiscretisedSolver, problem::CrystallisationProblem,
                        ode_solution, boundary_flow_state::NamedTuple)
    cell_centre = solver.cell_centre
    nd_matrix = ode_solution[_population_state_range(problem), :]
    vol_weighted_dens = volumeweighteddensity(cell_centre, nd_matrix,
                                              problem.model.system.volume_shape_factor)
    population = MeshPopulation(nd_matrix, vol_weighted_dens,
                                quantilecalculator(cell_centre, vol_weighted_dens, 0.1),
                                quantilecalculator(cell_centre, vol_weighted_dens, 0.5),
                                quantilecalculator(cell_centre, vol_weighted_dens, 0.9))
    return _crystallisation_solution(problem, ode_solution, population,
                                     _momentsizes(cell_centre, nd_matrix),
                                     vec(ode_solution[:, end]),
                                     merge(_reactor_solution_state(problem, ode_solution),
                                           boundary_flow_state))
end

"""
    _fill_mesh_flux!(solver, solver_cache, face_flux, numberdensity, net_growth, nucleation_rate)

Fill the face fluxes of the solver's transport scheme; `solver_cache` is the
solver's extra per-state scratch (see `_mesh_solver_cache`).
"""
@inline _fill_mesh_flux!(::FiniteVol, ::Nothing, face_flux, numberdensity, net_growth,
                         nucleation_rate) =
    _fill_fv_flux!(face_flux, numberdensity, net_growth, nucleation_rate)

"""Extra `DiffCache` scratch a mesh solver needs beside the flux and growth buffers."""
_mesh_solver_cache(::FiniteVol, problem) = nothing

@inline _mesh_solver_cache_view(::Nothing, state) = nothing
@inline _mesh_solver_cache_view(solver_cache_dc::DiffCache, state) =
    get_tmp(solver_cache_dc, state)

"""
    _mesh_population_rhs(problem, flux_cache_dc, growth_rate_cache_dc, solver_cache_dc)

In-place ODE right-hand side shared by the `FiniteVol` and `WENO` solvers:
net growth, face flux (`_fill_mesh_flux!`), transport divergence, aggregation,
breakage, solvent and operation terms. The caches are `DiffCache` buffers so
ForwardDiff dual states retype them.
"""
function _mesh_population_rhs(CryProblem::CrystallisationProblem, flux_cache_dc,
                              growth_rate_cache_dc, solver_cache_dc)
    solver = CryProblem.solver
    feed_population = _operation_feed_population(CryProblem)
    aggregation_table = _mesh_aggregation_table(CryProblem.model.aggregation.law, solver)
    growth_layout = _mesh_growth_layout(CryProblem)
    cell_centre_cubed = solver.cell_centre .^ 3

    function mesh_rhs!(dstdt, st, p, t)
        face_flux = get_tmp(flux_cache_dc, st)
        net_growth = _mesh_net_growth(growth_layout, growth_rate_cache_dc, p,
                                      CryProblem, st, t)
        _fill_mesh_flux!(solver, _mesh_solver_cache_view(solver_cache_dc, st), face_flux,
                         crystal_state(CryProblem, st), net_growth,
                         nucleationrate(CryProblem.model.nucleation.law, p.nucl,
                                        CryProblem, st, t))

        population_derivative = crystal_state(CryProblem, dstdt)
        solid_volume_rate = _mesh_transport_divergence!(population_derivative, face_flux,
                                                        solver, cell_centre_cubed)
        _add_aggregation_rate!(population_derivative, CryProblem.model.aggregation.law, p.agg,
                               CryProblem, st, t, aggregation_table)
        _add_breakage_rate!(population_derivative, CryProblem.model.breakage.law, p.br,
                            CryProblem, st, t)

        _write_mesh_solvent_derivatives!(dstdt, CryProblem, st, t, net_growth,
                                         solid_volume_rate)
        _add_operation_transport!(dstdt, st, CryProblem, t, feed_population)
        return nothing
    end
    return mesh_rhs!
end

"""
    crystallisation_odeproblem(solver::FiniteVol, problem, saveat, kinetic_values)

High-resolution (OSPRE-limited) transport for a scalar net growth rate,
first-order signed upwinding when growth or dissolution depends on length
(see `_fill_fv_flux!`). The default time stepper is `Tsit5` for
scalar growth laws and `SSPRK43` for length-dependent growth laws.
"""
function crystallisation_odeproblem(solver::FiniteVol, CryProblem::CrystallisationProblem,
                                    saveat, kinetic_values)
    flux_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize + 1))
    growth_rate_cache_dc = DiffCache(zeros(CryProblem.solver.meshsize))
    finite_volume_rhs! = _mesh_population_rhs(CryProblem, flux_cache_dc, growth_rate_cache_dc,
                                              _mesh_solver_cache(solver, CryProblem))

    # Plain state at the parameter element type (do NOT convert to the
    # ComponentArray parameter container).
    u0_typed = eltype(kinetic_values).(_get_initial_state(CryProblem))
    ODEprob = ODEProblem(finite_volume_rhs!, u0_typed, (saveat[1], saveat[end]),
                         kinetic_values)
    default_algorithm = _is_length_law(CryProblem.model.growth.law) ? SSPRK43() : Tsit5()
    return (ODEprob, _timestepping_algorithm(solver, default_algorithm))
end
