#### Kinetics
## Nucleation
"""
    nucleationrate(::nucl_CNT, parameters::AbstractVector, S::Real,
                   system, temperature, numberdensity)

Calculate nucleation rate using Classical Nucleation Theory (CNT).

# Arguments
- `parameters`: Vector containing [A_j, γ] where:
  - A_j: Pre-exponential factor
  - γ: Surface tension (mJ/m²)
- `S`: Supersaturation ratio
 - `system`: Crystallisation system parameters (molecular volume, constants, etc.)
 - `temperature`: Instantaneous temperature in Kelvin (use `temperature(sys.temp_profile, t)`)
 - `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_CNT, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (60 * exp(p.Aj)) *
        S *
        exp(-16π * ((p.γ * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(::nucl_empirical, parameters::AbstractVector, S::Real,
                   system, temperature, numberdensity)

Calculate nucleation rate using an empirical power law model.

# Arguments
- `parameters`: Vector containing [A_j, j] where:
  - A_j: Pre-exponential factor (log10 scale)
  - j: Power law exponent
- `S`: Supersaturation ratio
 - `system`: Crystallisation system parameters
 - `temperature`: Instantaneous temperature in Kelvin
 - `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_empirical, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ? (60 * 10^(p.Aj)) * (S - 1)^p.j : 0.0
end

"""
    nucleationrate(::nucl_empirical_energy, parameters::AbstractVector, S::Real,
                   system, temperature, loading, numberdensity) -> Real

Calculate nucleation rate using empirical model with activation energy.

# Arguments
- `parameters`: Vector [Aj, Ea, j] where Aj is pre-exponential, Ea is activation energy (kJ/mol), j is exponent
- `S`: Supersaturation ratio
- `system`: Crystallisation system parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_empirical_energy, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return S > 1.001 ?
           (60 * exp(p.Aj)) * exp(-(p.Ea * 1e3) / (8.314 * temp)) *
           (S - 1)^p.j : 0.0
end

"""
    nucleationrate(::nucl_CNTnoS, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, numberdensity::AbstractVector)

Calculate nucleation rate using Classical Nucleation Theory (CNT) without S factor in pre-exponential term.

# Arguments
- `parameters`: Vector containing [A_j, γ] where:
  - A_j: Pre-exponential factor
  - γ: Surface tension (mJ/m²)
- `S`: Supersaturation ratio
 - `system`: Crystallisation system parameters
 - `temperature`: Instantaneous temperature in Kelvin
 - `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_CNTnoS, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (60 * exp(p.Aj)) *
        S *
        exp(-16π * ((p.γ * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(::nucl_secondary, parameters::AbstractVector, S::Real,
                   system, temperature, loading, numberdensity::AbstractVector) -> Real

Calculate secondary nucleation rate proportional to third moment (crystal mass).

# Arguments
- `parameters`: Vector [Aj, Ea, j] where Aj is pre-exponential, Ea is activation energy (kJ/mol), j is exponent
- `S`: Supersaturation ratio
- `system`: Crystallisation system parameters
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Secondary nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(nf::nucl_secondary, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        base_rate = (60 * exp(p.Aj)) *
                    exp(-(p.Ea * 1e3) / (8.314 * temp)) *
                    (S - 1)^p.j
        base_rate * _secondary_third_moment(prob.solver, prob,
                                            crystal_state(prob, state))
    else
        0.0
    end
end

"""
    nucleationrate(::nucl_prim_plus_second, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate combined primary (empirical) and secondary nucleation rate.

# Arguments
- `parameters`: Vector [Aj_prim, Ea_prim, j_prim, Aj_sec, Ea_sec, j_sec] (6 parameters total)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Combined nucleation rate (number/m³/s)
"""
function nucleationrate(nf::nucl_prim_plus_second, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    prim_rate = nucleationrate(nucl_empirical_energy(), p.prim, prob, state, t)
    sec_rate = nucleationrate(nucl_secondary(), p.sec, prob, state, t)
    return prim_rate + sec_rate
end

"""
    nucleationrate(::nucl_CNT_plus_second, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate combined CNT primary and secondary nucleation rate.

# Arguments
- `parameters`: Vector [Aj_CNT, γ, Aj_sec, Ea_sec, j_sec] (5 parameters total)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Combined nucleation rate (number/m³/s)
"""
function nucleationrate(nf::nucl_CNT_plus_second, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    p = _named_params(nf, parameters)
    cnt_rate = nucleationrate(nucl_CNT(), p.cnt, prob, state, t)
    sec_rate = nucleationrate(nucl_secondary(), p.sec, prob, state, t)
    return cnt_rate + sec_rate
end

"""
    nucleationrate(NuF::nucl_CNT_fixed, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate CNT nucleation rate using pre-fixed parameters embedded in the struct.

# Arguments
- `NuF`: Fixed CNT nucleation function with embedded Aj and γ parameters
- `parameters`: Ignored (parameters are taken from NuF)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(NuF::nucl_CNT_fixed, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    return if S > 1.001
        (60 * exp(NuF.Aj)) *
        S *
        exp(-16π * ((NuF.γ * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end

"""
    nucleationrate(NuF::nucl_empirical_fixed, parameters::AbstractVector, S::Real,
                   system::CrystallisationProblem, temperature, loading,
                   numberdensity::AbstractVector) -> Real

Calculate empirical nucleation rate using pre-fixed parameters embedded in the struct.

# Arguments
- `NuF`: Fixed empirical nucleation function with embedded Aj and j parameters
- `parameters`: Ignored (parameters are taken from NuF)
- `S`: Supersaturation ratio
- `system`: Crystallisation problem
- `temperature`: Temperature in Kelvin
- `loading`: Loading value
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) if S > 1.001, otherwise 0
"""
function nucleationrate(NuF::nucl_empirical_fixed, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    S = supersaturation(prob, state, t)
    return S > 1.001 ? (60 * 10^(NuF.Aj)) * (S - 1)^NuF.j : 0.0
end

"""
    nucleationrate(::nucl_CNT_multiloading, parameters::AbstractVector, S::Real,
                   system, temperature, loading, numberdensity)

Calculate nucleation rate using Classical Nucleation Theory (CNT) with loading-dependent parameters.

# Arguments
- `parameters`: Vector containing [A1, γ1, A2, γ2, ...] where each pair corresponds to a unique loading
- `S`: Supersaturation ratio
- `system`: Crystallisation system parameters (molecular volume, constants, etc.)
- `temperature`: Instantaneous temperature in Kelvin
- `loading`: Current loading value (used to select appropriate A and γ parameters)
- `numberdensity`: Current crystal size distribution

# Returns
- Nucleation rate (number/m³/s) using the parameters corresponding to the current loading
"""
function nucleationrate(nucl_func::nucl_CNT_multiloading, parameters::T, prob::CrystallisationProblem, state, t) where {T <: AbstractVector}
    S = supersaturation(prob, state, t)
    temp = temperature(prob.temp_profile, t)
    # Find which prob.loading corresponds to the current prob.loading value
    loading_idx = findfirst(==(prob.loading), nucl_func.unique_loadings)

    if loading_idx === nothing
        error("Loading value $prob.loading not found in unique_loadings: $(nucl_func.unique_loadings)")
    end

    # Extract the relevant A and γ parameters for this prob.loading
    param_idx = 2 * (loading_idx - 1) + 1
    A_param = parameters[param_idx]
    γ_param = parameters[param_idx + 1]

    # Apply the standard CNT nucleation rate formula
    return if S > 1.001
        (60 * exp(A_param)) *
        S *
        exp(-16π * ((γ_param * 1e-3)^3) * ((prob.molecular_volume)^2) /
            (3(prob.kb * temp)^3 * (log(S))^2))
    else
        0.0
    end
end


"""
    _secondary_third_moment(solver, prob, nd) -> Real

Third-moment contribution to the secondary nucleation rate, dispatched on the
solver: volume-density quadrature for discretised solvers, `max(0, nd[4])`
(µ3) for the MoM state.
"""
_secondary_third_moment(solver::AbstractDiscretisedSolver, prob, nd) =
    momentcalculator(solver.cell_centre, nd, 3)
_secondary_third_moment(solver::MoM, prob, nd) = max(0, nd[4])  # µ3 stored at index 4
_secondary_third_moment(solver::QMOM, prob, nd) =
    max(zero(eltype(nd)), nd[4])  # µ3 stored at index 4
