include("PhotonParameters.jl")
include("HolsteinParameters.jl")
include("SSHParameters.jl")
include("MinParameters.jl")
include("DispersionParameters.jl")
using SmoQyDQMC

# import SmoQyDQMC.JDQMCFramework    as dqmcf
using JDQMCFramework
# export ElectronphotonParameters, update!
@doc raw"""
    ElectronPhotonParameters{T<:Number, E<:AbstractFloat}

Describes all parameters in the electron-photon model.

# Fields

- `β::E`: Inverse temperature.
- `Δτ::E`: Discretization in imaginary time.
- `Lτ::Int`: Length of imaginary time axis.
- `x::Matrix{E}`: Photon fields, where each column represents the photon fields for a given imaginary time slice.
- `photon_parameters::PhotonParameters{E}`: Refer to [`PhotonParameters`](@ref).
- `holstein_parameters_up::HolsteinParameters{E}`: Spin up [`HolsteinParameters`](@ref).
- `holstein_parameters_dn::HolsteinParameters{E}`: Spin down [`HolsteinParameters`](@ref).
- `ssh_parameters_up::SSHParameters{T}`: Spin up [`SSHParameters`](@ref).
- `ssh_parameters_dn::SSHParameters{T}`: Spin down [`SSHParameters`](@ref).
- `dispersion_parameters::DispersionParameters{E}`: Refer to [`DispersionParameters`](@ref).
"""
struct ElectronPhotonParameters{T<:Number, E<:AbstractFloat}

    # inverse temperature
    β::E

    # discretization in imaginary time
    Δτ::E

    # length of imaginary time axis
    Lτ::Int

    # photon fields
    x::Matrix{E}

    # all the photon parameters
    photon_parameters::PhotonParameters{E}

    # all the photon dispersion parameters
    dispersion_parameters::DispersionParameters{E}
    
    # all the spin-up holstein coupling parameters
    holstein_parameters_up::HolsteinParameters{E}

    # all the spin-down holstein coupling parameters
    holstein_parameters_dn::HolsteinParameters{E}

    # all the spin-up ssh coupling parameters
    ssh_parameters_up::SSHParameters{T}

    # all the spin-down ssh coupling parameters
    ssh_parameters_dn::SSHParameters{T}
    
    # all the spin-up ssh coupling parameters
    min_parameters_up::MinParameters{T}

    # all the spin-down ssh coupling parameters
    min_parameters_dn::MinParameters{T}
end

@doc raw"""
    ElectronPhotonParameters(;
        β::E, Δτ::E,
        model_geometry::ModelGeometry{D,E},
        tight_binding_parameters::Union{TightBindingParameters{T,E}, Nothing} = nothing,
        tight_binding_parameters_up::Union{TightBindingParameters{T,E}, Nothing} = nothing,
        tight_binding_parameters_dn::Union{TightBindingParameters{T,E}, Nothing} = nothing,
        electron_photon_model::ElectronPhotonModel{T,E,D},
        rng::AbstractRNG
    ) where {T,E,D}

Initialize and return an instance of [`ElectronPhotonParameters`](@ref).
"""
function ElectronPhotonParameters(;
    β::E, Δτ::E,
    model_geometry::ModelGeometry{D,E},
    tight_binding_parameters::Union{TightBindingParameters{T,E}, Nothing} = nothing,
    tight_binding_parameters_up::Union{TightBindingParameters{T,E}, Nothing} = nothing,
    tight_binding_parameters_dn::Union{TightBindingParameters{T,E}, Nothing} = nothing,
    electron_photon_model::ElectronPhotonModel{T,E,D},
    PBCx,PBCy,init::Int,
    rng::AbstractRNG
) where {T,E,D}

    # specify spin-up and spin-down tight binding parameters if need
    if !isnothing(tight_binding_parameters)

        tight_binding_parameters_up = tight_binding_parameters
        tight_binding_parameters_dn = tight_binding_parameters
    end

    # initialize photon parameters
    photon_parameters = PhotonParameters(model_geometry = model_geometry,
                                         electron_photon_model = electron_photon_model,
                                         rng = rng)

    # initialize photon dispersion parameters
    dispersion_parameters = DispersionParameters(
        model_geometry = model_geometry,
        electron_photon_model = electron_photon_model,
        photon_parameters = photon_parameters,
        rng = rng
    )

    # initialize spin-down holstein parameters
    holstein_parameters_up, holstein_parameters_dn = HolsteinParameters(
        model_geometry = model_geometry,
        electron_photon_model = electron_photon_model,
        rng = rng
    )

    # initialize spin-up ssh parameters
    ssh_parameters_up, ssh_parameters_dn = SSHParameters(
        model_geometry = model_geometry,
        electron_photon_model = electron_photon_model,
        tight_binding_parameters_up = tight_binding_parameters_up,
        tight_binding_parameters_dn = tight_binding_parameters_dn,
        rng = rng
    )

    # initialize spin-up min parameters
    min_parameters_up, min_parameters_dn = MinParameters(
        model_geometry = model_geometry,
        electron_photon_model = electron_photon_model,
        tight_binding_parameters_up = tight_binding_parameters_up,
        tight_binding_parameters_dn = tight_binding_parameters_dn,
        PBCx = PBCx, PBCy = PBCy,
        rng = rng
    )
    # @show min_parameters_up
    # evaluate length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get relevant photon parameters
    (; Nphoton, M, Ω) = photon_parameters

    # allocate photon fields
    x = zeros(E, Nphoton, Lτ)

    ## iterate over photons
    # for photon in 1:Nphoton
    #     # if finite photon mass
    #     if isfinite(M[photon])
    #         # get the photon fields
    #         x_p = @view x[photon,:]
    #         # @show Ω[photon]
    #         # @show std_x_qho(β, 1.0, M[photon])
    #         # initialize photon field
    #         if iszero(Ω[photon])
    #             # uncertainty in photon position
    #             Δx = std_x_qho(β, 1.0, M[photon])
    #             # assign initial photon position
    #             x0 = Δx * randn(rng)
    #             @. x_p = x0
    #         else
    #             # uncertainty in photon position
    #             Δx = std_x_qho(β, Ω[photon], M[photon])
    #             # assign initial photon position
    #             x0 = Δx * randn(rng)
    #             @. x_p = x0
    #         end
    #     end
    # end

    for photon in 1:Nphoton
        # if finite photon mass
        if isfinite(M[photon])
            # get the photon fields
            # x_p = @view x[photon,:]
            # @show size(x_p)
            if iszero(Ω[photon])
                # uncertainty in photon position
                Δx = std_x_qho(β, 1.0, M[photon])
                # assign initial photon position
                # x0 = Δx * randn(rng)
                # @. x_p = x0
                # x_p = Δx*randn(rng,size(x_p,2))
                x[photon,:] = Δx*randn(rng,size(x[photon,:]))
            else
                # uncertainty in photon position
                Δx = std_x_qho(β, Ω[photon], M[photon])
                # assign initial photon position
                # x0 = Δx * randn(rng)
                # @. x_p = x0
                # @show Δx,randn(rng,size(x[photon,:]))
                x[photon,:] = Δx*randn(rng,size(x[photon,:]))
            end
        end
    end
    # dasd
    # # @show "initialllll",size(x),x
  
    for i in 1:size(x,2)
        if init == 1
            x[i] = 0
        elseif init == 2
            x[i] = pi/(min_parameters_up.α[model_geometry.lattice.L[1]+1])
        elseif init == 3
            x[i] += pi/(min_parameters_up.α[model_geometry.lattice.L[1]+1])
        elseif init == 4
            x[i] *= 2
        elseif init == 5
            x[i] *= 4
        elseif init == 6
            x[i] *= 8
        elseif init == 7
            x[i] *= 16
        # elseif init == 8
        #     x[i] *= 9
        end
    end
    if init == 8
        for photon in 1:Nphoton
            # @show size(x[photon,:])
            # @show randn(rng,size(x[photon,:]))
            # @show rand(rng,size(x[photon,:])[1])
            x[photon,:] = pi * rand(rng,size(x[photon,:])[1])
        end
    end

    @show init, "initial",x
    # initialize electron-photon parameters
    electron_photon_parameters = ElectronPhotonParameters(β
        , Δτ, Lτ, x,
        photon_parameters
        ,
        dispersion_parameters,
        holstein_parameters_up, holstein_parameters_dn,
        ssh_parameters_up, ssh_parameters_dn,
        min_parameters_up,min_parameters_dn
        )
    
    # return electron_photon_parameters
end

# @doc raw"""
#     initialize!(
#         fermion_path_integral_up::FermionPathIntegral{T,E},
#         fermion_path_integral_dn::FermionPathIntegral{T,E},
#         electron_photon_parameters::ElectronPhotonParameters{T,E}
#     ) where {T,E}

# Initialize the contribution of an [`ElectronPhotonParameters`](@ref) to a [`FermionPathIntegral`](@ref).
# """
function initialize!(
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    electron_photon_parameters::ElectronPhotonParameters{T,E}
) where {T,E}

    # initialize spin up fermion path integral
    initialize!(fermion_path_integral_up, electron_photon_parameters, spin = +1)

    # initialize spin down fermion path integral
    initialize!(fermion_path_integral_dn, electron_photon_parameters, spin = -1)

    return nothing
end

@doc raw"""
    initialize!(
        fermion_path_integral::FermionPathIntegral{T,E},
        electron_photon_parameters::ElectronPhotonParameters{T,E};
        spin::Int = +1,
    ) where {T,E}

Initialize the contribution of an [`ElectronPhotonParameters`](@ref) to a [`FermionPathIntegral`](@ref).
"""
function initialize!(
    fermion_path_integral::FermionPathIntegral{T,E},
    electron_photon_parameters::ElectronPhotonParameters{T,E};
    spin::Int = +1
) where {T,E}

    x = electron_photon_parameters.x
    if isone(spin)
        holstein_parameters = electron_photon_parameters.holstein_parameters_up
        ssh_parameters = electron_photon_parameters.ssh_parameters_up
        min_parameters = electron_photon_parameters.min_parameters_up
    else
        holstein_parameters = electron_photon_parameters.holstein_parameters_dn
        ssh_parameters = electron_photon_parameters.ssh_parameters_dn
        min_parameters = electron_photon_parameters.min_parameters_dn
    end

    # update fermion path integral based on holstein interaction
    update!(fermion_path_integral, holstein_parameters, x, 1)

    # update fermion path integral based on ssh interaction
    update!(fermion_path_integral, ssh_parameters, x, 1)
    # @show "here",min_parameters
    # update fermion path integral based on minimal coupling interaction
    update!(fermion_path_integral, min_parameters, x, 1)
    return nothing
end


@doc raw"""
    update!(
        fermion_path_integral_up::FermionPathIntegral{T,E},
        fermion_path_integral_dn::FermionPathIntegral{T,E},
        electron_photon_parameters::ElectronPhotonParameters{T,E},
        x′::Matrix{E},
        x::Matrix{E}
    ) where {T,E}

Update a [`FermionPathIntegral`](@ref) to reflect a change in the photon configuration from `x` to `x′`.
"""
function update!(
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    electron_photon_parameters::ElectronPhotonParameters{T,E},
    x′::Matrix{E},
    x::Matrix{E}
) where {T,E}

    # update spin up fermion path integral
    update!(fermion_path_integral_up, electron_photon_parameters, x′, x, spin = +1)

    # update spin down fermion path integral
    update!(fermion_path_integral_dn, electron_photon_parameters, x′, x, spin = -1)

    return nothing
end

@doc raw"""
    update!(
        fermion_path_integral::FermionPathIntegral{T,E},
        electron_photon_parameters::ElectronPhotonParameters{T,E},
        x′::Matrix{E},
        x::Matrix{E};
        spin::Int = +1
    ) where {T,E}

Update a [`FermionPathIntegral`](@ref) to reflect a change in the photon configuration from `x` to `x′`.
"""
function update!(
    fermion_path_integral::FermionPathIntegral{T,E},
    electron_photon_parameters::ElectronPhotonParameters{T,E},
    x′::Matrix{E},
    x::Matrix{E};
    spin::Int = +1
) where {T,E}

    if isone(spin)
        holstein_parameters = electron_photon_parameters.holstein_parameters_up
        ssh_parameters = electron_photon_parameters.ssh_parameters_up
        min_parameters = electron_photon_parameters.min_parameters_up
    else
        holstein_parameters = electron_photon_parameters.holstein_parameters_dn
        ssh_parameters = electron_photon_parameters.ssh_parameters_dn
        min_parameters = electron_photon_parameters.min_parameters_dn
    end

    # update fermion path integral based on holstein interaction and new photon configration
    update!(fermion_path_integral, holstein_parameters, x, -1)
    update!(fermion_path_integral, holstein_parameters, x′, +1)

    # update fermion path integral based on ssh interaction and new photon configration
    update!(fermion_path_integral, ssh_parameters, x, -1)
    update!(fermion_path_integral, ssh_parameters, x′, +1)
    
    # update fermion path integral based on minimal coupling interaction and new photon configration
    update!(fermion_path_integral, min_parameters, x, -1)
    update!(fermion_path_integral, min_parameters, x′, +1)

    return nothing
end

@doc raw"""
    update!(fermion_path_integral::FermionPathIntegral{T,E},
        electron_photon_parameters::ElectronPhotonParameters{T,E},
        x::Matrix{E},
        sgn::Int;
        spin::Int = +1
    ) where {T,E}

Update a [`FermionPathIntegral`](@ref) according to `sgn * x`.
"""
function update!(fermion_path_integral::FermionPathIntegral{T,E},
    electron_photon_parameters::ElectronPhotonParameters{T,E},
    x::Matrix{E},
    sgn::Int;
    spin::Int = +1
) where {T,E}

    if isone(spin)
        holstein_parameters = electron_photon_parameters.holstein_parameters_up
        ssh_parameters = electron_photon_parameters.ssh_parameters_up
        min_parameters = electron_photon_parameters.min_parameters_up
    else
        holstein_parameters = electron_photon_parameters.holstein_parameters_dn
        ssh_parameters = electron_photon_parameters.ssh_parameters_dn
        min_parameters = electron_photon_parameters.min_parameters_dn
    end

    # update fermion path integral based on holstein interaction and new photon configration
    update!(fermion_path_integral, holstein_parameters, x, sgn)

    # update fermion path integral based on ssh interaction and new photon configration
    update!(fermion_path_integral, ssh_parameters, x, sgn)
    # @show "hhhere",min_parameters
    # update fermion path integral based on minimal coupling interaction and new photon configration
    update!(fermion_path_integral, min_parameters, x, sgn)
    return nothing
end


# Given a quantum harmonic oscillator with frequency Ω and mass M at an
# inverse temperature of β, return the standard deviation of the equilibrium
# distribution for the photon position.
function std_x_qho(β::T, Ω::T, M::T) where {T<:AbstractFloat}

    ΔX = inv(sqrt(2 * M * Ω * tanh(β*Ω/2)))
    return ΔX
end


# Calculate the reduced mass given the mass of two photons `M` and `M′`.
function reduced_mass(M::T, M′::T) where {T<:AbstractFloat}

    if !isfinite(M)
        M″ = M′
    elseif !isfinite(M′)
        M″ = M
    else
        M″ = (M*M′)/(M+M′)
    end

    return M″
end
