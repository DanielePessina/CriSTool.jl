#### Kinetics
## Nucleation
"""
    nucleationrate(::CNTNucleation, parameters::AbstractVector,
                   problem::CrystallisationProblem, state, t) -> Real

Calculate nucleation rate using Classical Nucleation Theory (CNT).

# Arguments
- `parameters`: Vector containing [ln_nucleation_prefactor, surface_energy]
  where the first entry is a natural-log prefactor and `surface_energy` is in
  J/m².
- `problem`: Crystallisation problem providing the saturation model, molecular
  volume, and physical constants.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::CNTNucleation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        exp(p.ln_nucleation_prefactor) *
        S *
        exp(-16π * p.surface_energy^3 * prob.molecular_volume^2 /
            (3(prob.boltzmann_constant * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(::EmpiricalNucleation, parameters::AbstractVector,
                   problem::CrystallisationProblem, state, t) -> Real

Calculate nucleation rate using an empirical power law model.

# Arguments
- `parameters`: Vector containing [log10_nucleation_prefactor,
  nucleation_order], where the first entry is a base-10 logarithmic prefactor.
- `problem`: Crystallisation problem providing the saturation model.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::EmpiricalNucleation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    return S > 1.001 ? 10^p.log10_nucleation_prefactor *
                       (S - 1)^p.nucleation_order : 0.0
end

"""
    nucleationrate(::ArrheniusNucleation, parameters::AbstractVector,
                   problem::CrystallisationProblem, state, t) -> Real

Calculate nucleation rate using empirical model with activation energy.

# Arguments
- `parameters`: Vector [ln_nucleation_prefactor, activation_energy,
  nucleation_order], with activation energy in J/mol.
- `problem`: Crystallisation problem providing the saturation model and gas
  constant.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::ArrheniusNucleation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ?
           exp(p.ln_nucleation_prefactor) *
           exp(-p.activation_energy / (prob.R_gas_constant * temp)) *
           (S - 1)^p.nucleation_order : 0.0
end

"""
    nucleationrate(::CNTNucleationNoSupersaturation, parameters::AbstractVector,
                   problem::CrystallisationProblem, state, t) -> Real

Calculate nucleation rate using Classical Nucleation Theory (CNT) without S factor in pre-exponential term.

# Arguments
- `parameters`: Vector containing [ln_nucleation_prefactor, surface_energy]
  where the first entry is a natural-log prefactor and `surface_energy` is in
  J/m².
- `problem`: Crystallisation problem providing the saturation model, molecular
  volume, and physical constants.
- `state`: Current solver state, including the named solvent-state values.
- `t`: Time in seconds.

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::CNTNucleationNoSupersaturation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        exp(p.ln_nucleation_prefactor) *
        exp(-16π * p.surface_energy^3 * prob.molecular_volume^2 /
            (3(prob.boltzmann_constant * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(::SecondaryNucleation, parameters::AbstractVector,
                   problem::CrystallisationProblem, state, t) -> Real

Calculate secondary nucleation rate proportional to third moment (crystal mass).

# Arguments
- `parameters`: Vector [ln_nucleation_prefactor, activation_energy,
  nucleation_order], with activation energy in J/mol.
- `problem`: Crystallisation problem providing the saturation model, solver,
  and physical constants.
- `state`: Current solver state, including the named solvent-state values and
  crystal population.
- `t`: Time in seconds.

# Returns
- Secondary nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::SecondaryNucleation, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        base_rate = exp(p.ln_nucleation_prefactor) *
                    exp(-p.activation_energy / (prob.R_gas_constant * temp)) *
                    (S - 1)^p.nucleation_order
        base_rate * _secondary_third_moment(prob.solver, prob,
                                            crystal_state(prob, state))
    else
        0.0
    end
end

"""
    nucleationrate(combined_law::CombinedNucleation, parameters,
                   problem::CrystallisationProblem, state, t) -> Real

Sum of the primary and secondary component rates, each evaluated with its
nested parameter block.
"""
function nucleationrate(combined_law::CombinedNucleation, parameters::AbstractVector,
                        prob::CrystallisationProblem, state, t)
    named_values = _named_params(combined_law, parameters)
    return nucleationrate(combined_law.primary, named_values.primary, prob, state, t) +
           nucleationrate(combined_law.secondary, named_values.secondary, prob, state, t)
end

"""
    _secondary_third_moment(solver, prob, nd) -> Real

Third-moment contribution to the secondary nucleation rate, dispatched on the
solver: volume-density quadrature for discretised solvers, `max(0, nd[4])`
(the raw third moment, in the solver's metre-based convention) for the MoM
state.
"""
_secondary_third_moment(solver::AbstractDiscretisedSolver, prob, nd) =
    momentcalculator(solver.cell_centre, nd, 3)
_secondary_third_moment(solver::MoM, prob, nd) = max(0, nd[4])  # µ3 stored at index 4
_secondary_third_moment(solver::QMOM, prob, nd) =
    max(zero(eltype(nd)), nd[4])  # µ3 stored at index 4
