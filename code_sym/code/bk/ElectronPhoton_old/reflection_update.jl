include("bosonic_action.jl")
@doc raw"""
    reflection_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                       Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                       electron_photon_parameters::ElectronPhotonParameters{T,E};
                       fermion_path_integral_up::FermionPathIntegral{T,E},
                       fermion_path_integral_dn::FermionPathIntegral{T,E},
                       fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                       Bup::Vector{P}, Bdn::Vector{P}, rng::AbstractRNG,
                       photon_types = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Randomly sample a photon mode in the lattice, and propose an update that reflects all the photon fields associated with that photon mode ``x \rightarrow -x.``
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{T}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::E`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::T`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{T}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::E`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::T`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `electron_photon_parameters::ElectronPhotonParameters{T,E}`: Electron-photon parameters, including the current photon configuration.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `photon_types = nothing`: Collection of photon types in the unit cell to randomly sample a photon mode from. If `nothing` then all photon modes in the unit cell are considered.
"""
function reflection_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                            Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                            electron_photon_parameters::ElectronPhotonParameters{T,E};
                            fermion_path_integral_up::FermionPathIntegral{T,E},
                            fermion_path_integral_dn::FermionPathIntegral{T,E},
                            fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                            Bup::Vector{P}, Bdn::Vector{P}, rng::AbstractRNG,
                            photon_types = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′
    photon_parameters = electron_photon_parameters.photon_parameters::PhotonParameters{E}
    holstein_parameters_up = electron_photon_parameters.holstein_parameters_up::HolsteinParameters{E}
    holstein_parameters_dn = electron_photon_parameters.holstein_parameters_dn::HolsteinParameters{E}
    ssh_parameters_up = electron_photon_parameters.ssh_parameters_up::SSHParameters{T}
    ssh_parameters_dn = electron_photon_parameters.ssh_parameters_dn::SSHParameters{T}
    min_parameters_up = electron_photon_parameters.min_parameters_up::MinParameters{T}
    min_parameters_dn = electron_photon_parameters.min_parameters_dn::MinParameters{T}
    x = electron_photon_parameters.x

    # make sure stabilization frequencies match
    if fermion_greens_calculator_up.n_stab != fermion_greens_calculator_up_alt.n_stab
        resize!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up.n_stab)
    end

    # make sure stabilization frequencies match
    if fermion_greens_calculator_dn.n_stab != fermion_greens_calculator_dn_alt.n_stab
        resize!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn.n_stab)
    end

    # get the mass associated with each photon
    M = photon_parameters.M

    # get the number of photon modes per unit cell
    nphoton = photon_parameters.nphoton

    # total number of photon modes
    Nphoton = photon_parameters.Nphoton

    # number of unit cells
    Nunitcells = Nphoton ÷ nphoton

    # sample random photon mode
    photon_mode = _sample_photon_mode(rng, nphoton, Nunitcells, M, photon_types)

    # whether the exponentiated on-site energy matrix needs to be updated with the photon field,
    # true if photon mode appears in holstein coupling
    calculate_exp_V = (photon_mode in holstein_parameters_up.coupling_to_photon)

    # whether the exponentiated hopping matrix needs to be updated with the photon field,
    # true if photon mode appears in SSH coupling
    calculate_exp_K = (photon_mode in ssh_parameters_up.coupling_to_photon) || (min_parameters_up.Nmin > 0)

    # get the corresponding photon fields
    x_i = @view x[photon_mode, :]

    # calculate the initial bosonic action
    Sb = bosonic_action(electron_photon_parameters)

    # substract off the effect of the current photon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral_up, holstein_parameters_up, x, -1)
        update!(fermion_path_integral_dn, holstein_parameters_dn, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral_up, ssh_parameters_up, x, -1)
        update!(fermion_path_integral_dn, ssh_parameters_dn, x, -1)
        update!(fermion_path_integral_up, min_parameters_up, x, -1)
        update!(fermion_path_integral_dn, min_parameters_dn, x, -1)
    end

    # reflection photon fields for chosen mode
    @. x_i = -x_i

    # calculate the final bosonic action
    Sb′ = bosonic_action(electron_photon_parameters)

    # caclulate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update the fermion path integrals to reflect new photon field configuration
    if calculate_exp_V
        update!(fermion_path_integral_up, holstein_parameters_up, x, +1)
        update!(fermion_path_integral_dn, holstein_parameters_dn, x, +1)
    end
    if calculate_exp_K
        update!(fermion_path_integral_up, ssh_parameters_up, x, +1)
        update!(fermion_path_integral_dn, ssh_parameters_dn, x, +1)
        update!(fermion_path_integral_up, min_parameters_up, x, +1)
        update!(fermion_path_integral_dn, min_parameters_dn, x, +1)
    end

    # update the spin up and spin down propagators to reflect current photon configuration
    calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
    calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)

    # update the Green's function to reflect the new photon configuration
    logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
    logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(Gup)/det(Gup′)|⋅|det(Gdn)/det(Gdn′)|
    #                                    = exp(-ΔS_b)⋅|det(Mup′)/det(Mup)|⋅|det(Mdn′)/det(Mdn)|
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    else
        P_i = 0.0
    end

    # accept or reject the update
    if rand(rng) < P_i
        logdetGup = logdetGup′
        logdetGdn = logdetGdn′
        sgndetGup = sgndetGup′
        sgndetGdn = sgndetGdn′
        copyto!(Gup, Gup′)
        copyto!(Gdn, Gdn′)
        copyto!(fermion_greens_calculator_up, fermion_greens_calculator_up_alt)
        copyto!(fermion_greens_calculator_dn, fermion_greens_calculator_dn_alt)
        accepted = true
    else
        # substract off the effect of the current photon configuration on the fermion path integrals
        if calculate_exp_V
            update!(fermion_path_integral_up, holstein_parameters_up, x, -1)
            update!(fermion_path_integral_dn, holstein_parameters_dn, x, -1)
        end
        if calculate_exp_K
            update!(fermion_path_integral_up, ssh_parameters_up, x, -1)
            update!(fermion_path_integral_dn, ssh_parameters_dn, x, -1)
            update!(fermion_path_integral_up, min_parameters_up, x, -1)
            update!(fermion_path_integral_dn, min_parameters_dn, x, -1)
        end
        # revert to the original photon configuration
        @. x_i = -x_i
        # update the fermion path integrals to reflect new photon field configuration
        if calculate_exp_V
            update!(fermion_path_integral_up, holstein_parameters_up, x, +1)
            update!(fermion_path_integral_dn, holstein_parameters_dn, x, +1)
        end
        if calculate_exp_K
            update!(fermion_path_integral_up, ssh_parameters_up, x, +1)
            update!(fermion_path_integral_dn, ssh_parameters_dn, x, +1)
            update!(fermion_path_integral_up, min_parameters_up, x, +1)
            update!(fermion_path_integral_dn, min_parameters_dn, x, +1)
        end
        # update the fermion path integrals to reflect the original photon configuration
        calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    reflection_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                       electron_photon_parameters::ElectronPhotonParameters{T,E};
                       fermion_path_integral::FermionPathIntegral{T,E},
                       fermion_greens_calculator::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                       B::Vector{P}, rng::AbstractRNG,
                       photon_types = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Randomly sample a photon mode in the lattice, and propose an update that reflects all the photon fields associated with that photon mode ``x \rightarrow -x.``
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{T}`: Eqaul-time Greens function matrix.
- `logdetG::E`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::T`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `electron_photon_parameters::ElectronPhotonParameters{T,E}`: Electron-photon parameters, including the current photon configuration.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `photon_types = nothing`: Collection of photon types in the unit cell to randomly sample a photon mode from. If `nothing` then all photon modes in the unit cell are considered.
"""
function reflection_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                            electron_photon_parameters::ElectronPhotonParameters{T,E};
                            fermion_path_integral::FermionPathIntegral{T,E},
                            fermion_greens_calculator::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                            B::Vector{P}, rng::AbstractRNG,
                            photon_types = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    G′ = fermion_greens_calculator_alt.G′
    photon_parameters = electron_photon_parameters.photon_parameters::PhotonParameters{E}
    holstein_parameters = electron_photon_parameters.holstein_parameters_up::HolsteinParameters{E}
    ssh_parameters = electron_photon_parameters.ssh_parameters_up::SSHParameters{T}
    min_parameters = electron_photon_parameters.min_parameters_up::MinParameters{T}
    
    x = electron_photon_parameters.x

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # get the mass associated with each photon
    M = photon_parameters.M

    # get the number of photon modes per unit cell
    nphoton = photon_parameters.nphoton

    # total number of photon modes
    Nphoton = photon_parameters.Nphoton

    # number of unit cells
    Nunitcells = Nphoton ÷ nphoton

    # sample random photon mode
    photon_mode = _sample_photon_mode(rng, nphoton, Nunitcells, M, photon_types)

    # whether the exponentiated on-site energy matrix needs to be updated with the photon field,
    # true if photon mode appears in holstein coupling
    calculate_exp_V = (photon_mode in holstein_parameters.coupling_to_photon)

    # whether the exponentiated hopping matrix needs to be updated with the photon field,
    # true if photon mode appears in SSH coupling
    calculate_exp_K = (photon_mode in ssh_parameters.coupling_to_photon) || (min_parameters_up.Nmin > 0)

    # get the corresponding photon fields
    x_i = @view x[photon_mode, :]

    # calculate the initial bosonic action
    Sb = bosonic_action(electron_photon_parameters)

    # substract off the effect of the current photon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral, holstein_parameters, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral, ssh_parameters, x, -1)
        update!(fermion_path_integral, min_parameters, x, -1)
    end

    # reflection photon fields for chosen mode
    @. x_i = -x_i

    # calculate the final bosonic action
    Sb′ = bosonic_action(electron_photon_parameters)

    # caclulate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update the fermion path integrals to reflect new photon field configuration
    if calculate_exp_V
        update!(fermion_path_integral, holstein_parameters, x, +1)
    end
    if calculate_exp_K
        update!(fermion_path_integral, ssh_parameters, x, +1)
        update!(fermion_path_integral, min_parameters, x, +1)
    end

    # update the spin up and spin down propagators to reflect current photon configuration
    calculate_propagators!(B, fermion_path_integral, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)

    # update the Green's function to reflect the new photon configuration
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(G)/det(G′)|²
    #                                    = exp(-ΔS_b)⋅|det(M′)/det(M)|²
    if isfinite(logdetG′)
        P_i = exp(-ΔSb + 2*logdetG - 2*logdetG′)
    else
        P_i = 0.0
    end

    # accept or reject the update
    if rand(rng) < P_i
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        accepted = true
    else
        # substract off the effect of the current photon configuration on the fermion path integrals
        if calculate_exp_V
            update!(fermion_path_integral, holstein_parameters, x, -1)
        end
        if calculate_exp_K
            update!(fermion_path_integral, ssh_parameters, x, -1)
            update!(fermion_path_integral, min_parameters, x, -1)
        end
        # revert to the original photon configuration
        @. x_i = -x_i
        # update the fermion path integrals to reflect new photon field configuration
        if calculate_exp_V
            update!(fermion_path_integral, holstein_parameters, x, +1)
        end
        if calculate_exp_K
            update!(fermion_path_integral, ssh_parameters, x, +1)
            update!(fermion_path_integral, min_parameters, x, +1)
        end
        # update the fermion path integrals to reflect the original photon configuration
        calculate_propagators!(B, fermion_path_integral, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end


# sample a single random photon mode
function _sample_photon_mode(rng::AbstractRNG, nphoton::Int, Nunitcells::Int, masses::Vector{T}, photon_types = nothing) where {T<:AbstractFloat}

    # initialize photon mode to zero
    photon_mode = 0

    # initialize photon mass
    mass = one(T)

    # if set of photon types unit cell is not specified set to all photon modes in unit cell
    if isnothing(photon_types)
        photon_types = 1:nphoton
    end

    # sample photon mode with finite mass
    while iszero(photon_mode) || isinf(mass)

        # randomly sample photon type
        photon_type = rand(rng, photon_types)

        # randomly sample unit cell
        unit_cell = rand(rng, 1:Nunitcells)

        # get the photon mode
        photon_mode = (photon_type - 1) * Nunitcells + unit_cell

        # get photon mass
        mass = masses[photon_mode]
    end

    return photon_mode
end