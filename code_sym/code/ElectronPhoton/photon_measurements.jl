#######################################
## PHOTON KINETIC ENERGY MEASUREMENT ##
#######################################

@doc raw"""
    measure_photon_kinetic_energy(electron_photon_parameters::ElectronPhotonParameters{T,E},
                                  n::Int) where {T<:Number, E<:AbstractFloat}

Evaluate the average photon kinetic energy for photon mode `n`.
The measurement is made using the expression
```math
\langle K \rangle = \frac{1}{2\Delta\tau} - \frac{M}{2}\bigg\langle\frac{(x_{l+1}-x_{l})^{2}}{\Delta\tau^{2}}\bigg\rangle. 
```
"""
function measure_photon_kinetic_energy(electron_photon_parameters::ElectronPhotonParameters{T,E},
                                       n::Int) where {T<:Number, E<:AbstractFloat}

    (; x, Δτ) = electron_photon_parameters
    photon_parameters = electron_photon_parameters.photon_parameters::PhotonParameters{E}

    # calculate photon kinetic energy
    K = measure_photon_kinetic_energy(photon_parameters, x, Δτ, n)

    return K
end

function measure_photon_kinetic_energy(photon_parameters::PhotonParameters{T},
                                       x::Matrix{T}, Δτ::T, n::Int) where {T<:AbstractFloat}

    (; M, nphoton, Nphoton) = photon_parameters

    # initialize photon kinetic energy to zero
    K = zero(T)

    # length of imaginary time axis
    Lτ = size(x,2)

    # number of unit cells in lattice
    Nunitcells = Nphoton ÷ nphoton

    # reshape photon field
    x′ = reshape(x, (Nunitcells, nphoton, Lτ))
    M′ = reshape(M, (Nunitcells, nphoton))

    # only non-zero kinetic energy if finite mass
    if isfinite(M′[1,n])
        # calculate (M/2)⋅⟨(x[l+1]-x[l])²/Δτ²⟩
        # iterate over imaginary time slice
        @fastmath @inbounds for l in axes(x′, 3)
            # iterate over unit cells
            for u in axes(x′, 1)
                # calculate K = 1/(2Δτ) - (M/2)⋅⟨(x[l+1]-x[l])²/Δτ²⟩
                K += 1/(2*Δτ) - M′[u,n]/2 * (x′[u,n,mod1(l+1,Lτ)] - x′[u,n,l])^2 / Δτ^2
            end
        end

        # normalize the kinetic energy
        K = K / (Nunitcells * Lτ)
    end

    return K
end

#########################################
## PHOTON POTENTIAL ENERGY MEASUREMENT ##
#########################################

@doc raw"""
    measure_photon_potential_energy(electron_photon_parameters::ElectronPhotonParameters{T,E},
                                    n::Int) where {T<:Number, E<:AbstractFloat}

Calculate the average photon potential energy, given by
```math
U = \frac{1}{2} M \Omega^2 \langle \hat{X}^2 \rangle + \frac{1}{24} M \Omega_4^2 \langle \hat{X}^4 \rangle,
```
for photon mode `n` in the unit cell.
"""
function measure_photon_potential_energy(electron_photon_parameters::ElectronPhotonParameters{T,E},
                                         n::Int) where {T<:Number, E<:AbstractFloat}

    photon_parameters = electron_photon_parameters.photon_parameters::PhotonParameters{E}
    (; x) = electron_photon_parameters
    (; Ω, Ω4, M, nphoton) = photon_parameters

    U = measure_photon_potential_energy(x, M, Ω, Ω4, nphoton, n)

    return U
end

function measure_photon_potential_energy(x::Matrix{T}, M::Vector{T}, Ω::Vector{T}, Ω4::Vector{T},
                                         nphoton::Int, n::Int) where {T<:AbstractFloat}

    # length of imaginary time axis
    Lτ = size(x, 2)

    # total number of photon modes in lattice
    Nphoton = size(x, 1)

    # number of unit cells in lattice
    Nunitcell = Nphoton ÷ nphoton

    # initialize photon potential energy to zero
    U = zero(T)

    # reshape arrays
    x′  = reshape(x,  (Nunitcell, nphoton, Lτ))
    M′  = reshape(M,  (Nunitcell, nphoton))
    Ω′  = reshape(Ω,  (Nunitcell, nphoton))
    Ω4′ = reshape(Ω4, (Nunitcell, nphoton))

    # make sure photon mass is finite
    if isfinite(M′[1,n])
        # iterate over imaginary time
        @fastmath @inbounds for l in axes(x′,3)
            # iterate over unit cells
            for u in axes(x′,1)
                # calcualte photon potential energy
                U += M′[u,n]*Ω′[u,n]^2*x′[u,n,l]^2/2 + M′[u,n]*Ω4′[u,n]^2*x′[u,n,l]^4/24
            end
        end
        # normalize photon potential energy measurement
        U /= (Nunitcell * Lτ)
    end

    return U
end

########################################
## MEASURE MOMENTS OF PHOTON POSITION ##
########################################

@doc raw"""
    measure_photon_position_moment(electron_photon_parameters::ElectronPhotonParameters{T,E},
                                   n::Int, m::Int) where {T<:Number, E<:AbstractFloat}

Measure ``\langle X^m \rangle`` for photon mode `n` in the unit cell.
"""
function measure_photon_position_moment(electron_photon_parameters::ElectronPhotonParameters{T,E},
                                        n::Int, m::Int) where {T<:Number, E<:AbstractFloat}

    photon_parameters = electron_photon_parameters.photon_parameters::PhotonParameters{E}
    nphoton = photon_parameters.nphoton::Int
    x = electron_photon_parameters.x::Matrix{E}
    xm = measure_photon_position_moment(x, nphoton, n, m)

    return xm
end

function measure_photon_position_moment(x::Matrix{T}, nphoton::Int, n::Int, m::Int) where {T<:AbstractFloat}

    # length of imaginary time axis
    Lτ = size(x,2)

    # total number of photon modes in lattice
    Nphoton = size(x,1)

    # number of unit cells in lattice
    Nunitcell = Nphoton ÷ nphoton

    # reshape photon fields
    x′ = reshape(x, (Nunitcell, nphoton, Lτ))

    # photon moment
    xm = zero(T)

    # iterate over imaginary time slices
    for l in axes(x′,3)
        # iterate over unit cells
        for u in axes(x′,1)
            xm += x′[u,n,l]^m
        end
    end

    # normalize measurement
    xm /= (Nunitcell * Lτ)

    return xm
end