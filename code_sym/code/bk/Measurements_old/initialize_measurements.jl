include("CorrelationContainer.jl")
######################################
## INITIALIZE MEASUREMENT CONTAINER ##
######################################

@doc raw"""
    initialize_measurement_container(
        model_geometry::ModelGeometry{D,T,N},
        β::T, Δτ::T
    ) where {T<:AbstractFloat, D, N}

Initialize and return a measurement container of type `NamedTuple`.
"""
function initialize_measurement_container(
    model_geometry::ModelGeometry{D,T,N},
    β::T, Δτ::T
) where {T<:AbstractFloat, D, N}

    lattice   = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,T,N}
    bonds     = model_geometry.bonds::Vector{Bond{D}}

    # number of orbitals per unit cell
    norbitals = unit_cell.n

    # length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # size of lattice in unit cells in direction of each lattice vector
    L = lattice.L

    # initialize global measurements
    global_measurements = Dict{String, Complex{T}}(k => zero(Complex{T}) for k in GLOBAL_MEASUREMENTS)

    # initialize local measurement
    local_measurements = Dict{String, Vector{Complex{T}}}(
        "density"    => zeros(Complex{T}, norbitals), # average density for each orbital species
        "density_up" => zeros(Complex{T}, norbitals), # average density for each orbital species
        "density_dn" => zeros(Complex{T}, norbitals), # average density for each orbital species
        "double_occ" => zeros(Complex{T}, norbitals), # average double occupancy for each orbital
    )

    # initialize time displaced correlation measurement dictionary
    equaltime_correlations = Dict{String, CorrelationContainer{D,T}}()

    # initialize time displaced correlation measurement dictionary
    time_displaced_correlations = Dict{String, CorrelationContainer{D+1,T}}()

    # initialize integrated correlation measurement dictionary
    integrated_correlations = Dict{String, CorrelationContainer{D,T}}()

    # initialize measurement container
    measurement_container = (
        global_measurements         = global_measurements,
        local_measurements          = local_measurements,
        equaltime_correlations      = equaltime_correlations,
        time_displaced_correlations = time_displaced_correlations,
        integrated_correlations     = integrated_correlations,
        L                           = L,
        Lτ                          = Lτ,
        a                           = zeros(Complex{T}, L..., Lτ),
        a′                          = zeros(Complex{T}, L..., Lτ),
        a″                          = zeros(Complex{T}, L..., Lτ),
    )

    return measurement_container
end


############################################################
## INITIALIZE MEASUREMENTS ASSOCIATED WITH VARIOUS MODELS ##
############################################################

@doc raw"""
    initialize_measurements!(
        measurement_container::NamedTuple,
        tight_binding_model_up::TightBindingModel{T,E},
        tight_binding_model_dn::TightBindingModel{T,E},
    ) where {T<:Number, E<:AbstractFloat}

    initialize_measurements!(
        measurement_container::NamedTuple,
        tight_binding_model::TightBindingModel{T,E}
    ) where {T<:Number, E<:AbstractFloat}

Initialize tight-binding model related measurements.

# Initialized Measurements

- `onsite_energy`: Refer to [`measure_onsite_energy`](@ref).
- `onsite_energy_up`: Refer to [`measure_onsite_energy`](@ref).
- `onsite_energy_dn`: Refer to [`measure_onsite_energy`](@ref).
- `hopping_energy`: Refer to [`measure_hopping_energy`](@ref).
- `hopping_energy_up`: Refer to [`measure_hopping_energy`](@ref).
- `hopping_energy_dn`: Refer to [`measure_hopping_energy`](@ref).
"""
function initialize_measurements!(
    measurement_container::NamedTuple,
    tight_binding_model_up::TightBindingModel{T,E},
    tight_binding_model_dn::TightBindingModel{T,E},
) where {T<:Number, E<:AbstractFloat}

    initialize_measurements!(measurement_container, tight_binding_model_up)

    return nothing
end

function initialize_measurements!(
    measurement_container::NamedTuple,
    tight_binding_model::TightBindingModel{T,E}
) where {T<:Number, E<:AbstractFloat}

    (; local_measurements, global_measurements) = measurement_container

    # number of orbitals per unit cell
    norbital = length(tight_binding_model.ϵ_mean)

    # number of types of hoppings
    nhopping = length(tight_binding_model.t_bond_ids)

    # initialize chemical potential as global measurement
    global_measurements["chemical_potential"] = zero(Complex{E})

    # initialize on-site energy measurement
    local_measurements["onsite_energy"]    = zeros(Complex{E}, norbital)
    local_measurements["onsite_energy_up"] = zeros(Complex{E}, norbital)
    local_measurements["onsite_energy_dn"] = zeros(Complex{E}, norbital)

    # initialize hopping related measurements
    if nhopping > 0

        local_measurements["hopping_energy"]           = zeros(Complex{E}, nhopping)
        local_measurements["hopping_energy_up"]        = zeros(Complex{E}, nhopping)
        local_measurements["hopping_energy_dn"]        = zeros(Complex{E}, nhopping)

        local_measurements["bare_hopping_energy"]      = zeros(Complex{E}, nhopping)
        local_measurements["bare_hopping_energy_up"]   = zeros(Complex{E}, nhopping)
        local_measurements["bare_hopping_energy_dn"]   = zeros(Complex{E}, nhopping)

        local_measurements["hopping_amplitude"]        = zeros(Complex{E}, nhopping)
        local_measurements["hopping_amplitude_up"]     = zeros(Complex{E}, nhopping)
        local_measurements["hopping_amplitude_dn"]     = zeros(Complex{E}, nhopping)

        local_measurements["hopping_inversion"]        = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_up"]     = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_dn"]     = zeros(Complex{E}, nhopping)

        local_measurements["hopping_inversion_avg"]    = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_avg_up"] = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_avg_dn"] = zeros(Complex{E}, nhopping)
    end

    return nothing
end


@doc raw"""
    initialize_measurements!(
        measurement_container::NamedTuple,
        hubbard_model::HubbardModel{T}
    ) where {T<:AbstractFloat}

Initialize Hubbard model related measurements.

# Initialized Measurements:

- `hubbard_energy`: Refer to [`measure_hopping_energy`](@ref).
"""
function initialize_measurements!(
    measurement_container::NamedTuple,
    hubbard_model::HubbardModel{T}
) where {T<:AbstractFloat}

    (; local_measurements) = measurement_container

    # number of orbitals in unit cell
    norbital = length(local_measurements["density"])

    # initialize hubbard energy measurement U⋅nup⋅ndn
    local_measurements["hubbard_energy"] = zeros(Complex{T}, norbital)

    return nothing
end

@doc raw"""
    initialize_measurements!(
        measurement_container::NamedTuple,
        electron_photon_model::ElectronPhotonModel{T, E, D}
    ) where {T<:Number, E<:AbstractFloat, D}

Initialize electron-photon model related measurements.

# Initialized Measurements:

- `photon_kinetic_energy`: Refer to [`measure_photon_kinetic_energy`](@ref).
- `photon_potential_energy`: Refer to [`measure_photon_potential_energy`](@ref).
- `X`: Measure ``\langle \hat{X} \rangle``, refer to [`measure_photon_position_moment`](@ref).
- `X2`: Measure ``\langle \hat{X}^2 \rangle``, refer to [`measure_photon_position_moment`](@ref).
- `X3`: Measure ``\langle \hat{X}^3 \rangle``, refer to [`measure_photon_position_moment`](@ref).
- `X4`: Measure ``\langle \hat{X}^4 \rangle``, refer to [`measure_photon_position_moment`](@ref).
- `holstein_energy`: Refer to [`measure_holstein_energy`](@ref).
- `holstein_energy_up`: Refer to [`measure_holstein_energy`](@ref).
- `holstein_energy_dn`: Refer to [`measure_holstein_energy`](@ref).
- `ssh_energy`: Refer to [`measure_ssh_energy`](@ref).
- `ssh_energy_up`: Refer to [`measure_ssh_energy`](@ref).
- `ssh_energy_dn`: Refer to [`measure_ssh_energy`](@ref).
- `ssh_sgn_switch_up`: Refer to [`measure_ssh_sgn_switch`](@ref).
- `ssh_sgn_switch_dn`: Refer to [`measure_ssh_sgn_switch`](@ref).
- `dispersion_energy`: Refer to [`measure_dispersion_energy`](@ref).
"""
function initialize_measurements!(
    measurement_container::NamedTuple,
    electron_photon_model::ElectronPhotonModel{T, E, D}
) where {T<:Number, E<:AbstractFloat, D}

    (; local_measurements) = measurement_container
    (; photon_modes, holstein_couplings_up, ssh_couplings_up, min_couplings_up, photon_dispersions) = electron_photon_model

    _initialize_measurements!(local_measurements, photon_modes)
    _initialize_measurements!(local_measurements, holstein_couplings_up)
    _initialize_measurements!(local_measurements, ssh_couplings_up)
    _initialize_measurements!(local_measurements, min_couplings_up)
    _initialize_measurements!(local_measurements, photon_dispersions)

    return nothing
end

# photon mode related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{T}}},
                                   photon_modes::Vector{PhotonMode{T}}) where {T<:AbstractFloat}

    # number of photon modes
    n_modes = length(photon_modes)

    # add measurements
    local_measurements["photon_kin_energy"] = zeros(Complex{T}, n_modes)
    local_measurements["photon_pot_energy"] = zeros(Complex{T}, n_modes)
    local_measurements["X"]  = zeros(Complex{T}, n_modes)
    local_measurements["X2"] = zeros(Complex{T}, n_modes)
    local_measurements["X3"] = zeros(Complex{T}, n_modes)
    local_measurements["X4"] = zeros(Complex{T}, n_modes)

    return nothing
end

# holstein related coupling measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{T}}},
                                   holstein_couplings::Vector{HolsteinCoupling{T,D}}) where {T<:AbstractFloat, D}

    # number of photon modes
    n_couplings = length(holstein_couplings)

    # add measurements
    local_measurements["holstein_energy"]    = zeros(Complex{T}, n_couplings)
    local_measurements["holstein_energy_up"] = zeros(Complex{T}, n_couplings)
    local_measurements["holstein_energy_dn"] = zeros(Complex{T}, n_couplings)

    return nothing
end

# ssh coupling related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                   ssh_couplings::Vector{SSHCoupling{T,E,D}}) where {T<:Number, E<:AbstractFloat, D}

    # number of photon modes
    n_couplings = length(ssh_couplings)

    # add measurements
    local_measurements["ssh_energy"]        = zeros(Complex{E}, n_couplings)
    local_measurements["ssh_energy_up"]     = zeros(Complex{E}, n_couplings)
    local_measurements["ssh_energy_dn"]     = zeros(Complex{E}, n_couplings)

    return nothing
end

# min coupling related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                   min_couplings::Vector{MinCoupling{T,E,D}}) where {T<:Number, E<:AbstractFloat, D}

# number of photon modes
n_couplings = length(min_couplings)

# add measurements
local_measurements["min_energy"]        = zeros(Complex{E}, n_couplings)
local_measurements["min_energy_up"]     = zeros(Complex{E}, n_couplings)
local_measurements["min_energy_dn"]     = zeros(Complex{E}, n_couplings)

return nothing
end

# photon dispersion related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{T}}},
                                   photon_dispersions::Vector{PhotonDispersion{T,D}}) where {T<:AbstractFloat, D}

    # number of photon modes
    n_couplings = length(photon_dispersions)

    # add measurements
    local_measurements["dispersion_energy"] = zeros(Complex{T}, n_couplings)

    return nothing
end


#########################################
## INITIALIZE CORRELATION MEASUREMENTS ##
#########################################

@doc raw"""
    initialize_correlation_measurements!(;
        measurement_container::NamedTuple,
        model_geometry::ModelGeometry{D,T,N},
        correlation::String, pairs::AbstractVector{NTuple{2,Int}},
        time_displaced::Bool,
        integrated::Bool = false
    )  where {T<:AbstractFloat, D, N}

Initialize measurements of `correlation` for all pairs of bond ID's in `pairs`.
The name `correlation` must appear in `CORRELATION_FUNCTIONS`.
If `time_displaced = true` then time-displaced and integrated correlation measurements are made.
If `time_displaced = false` and `integrated = false`, then just equal-time correlation measurements are made.
If `time_displaced = false` and `integrated = true`, then both equal-time and integrated correlation measurements are made.
"""
function initialize_correlation_measurements!(;
    measurement_container::NamedTuple,
    model_geometry::ModelGeometry{D,T,N},
    correlation::String, pairs::AbstractVector{NTuple{2,Int}},
    time_displaced::Bool,
    integrated::Bool = false,
    translation::Bool = true
)  where {T<:AbstractFloat, D, N}

    # iterate over all bond/orbial ID pairs
    for pair in pairs
        initialize_correlation_measurement!(measurement_container = measurement_container,
                                            model_geometry = model_geometry,
                                            correlation = correlation,
                                            pair = pair,
                                            time_displaced = time_displaced,
                                            integrated = integrated,
                                            translation = translation)
    end

    return nothing
end

# initialize a single correlation measurement
function initialize_correlation_measurement!(; measurement_container::NamedTuple,
                                             model_geometry::ModelGeometry{D,T,N},
                                             correlation::String, pair::NTuple{2,Int},
                                             time_displaced::Bool,
                                             integrated::Bool = false,translation::Bool = true)  where {T<:AbstractFloat, D, N}

    (; time_displaced_correlations, integrated_correlations, equaltime_correlations) = measurement_container

    # check to make sure valid correlation measurement
    @assert correlation in keys(CORRELATION_FUNCTIONS)

    # extent of lattice in unit cells
    L = measurement_container.L

    # length of imaginary time axis
    Lτ = measurement_container.Lτ

    # if time displaced or integrated measurement should be made
    if time_displaced || integrated

        # add time-displaced and integrated correlation key if not present
        if !haskey(time_displaced_correlations, correlation)
            time_displaced_correlations[correlation] = CorrelationContainer(D+1, T, time_displaced)
            integrated_correlations[correlation] = CorrelationContainer(D, T, false)
        end

        # add time-dispalced correlation measurement
        push!(time_displaced_correlations[correlation].id_pairs, pair)
        if (CORRELATION_FUNCTIONS[correlation] == "BOND_ID") || (CORRELATION_FUNCTIONS[correlation] == "ORBITAL_ID")
            push!(time_displaced_correlations[correlation].bond_id_pairs, pair) # record bond ID pair as same as correlation ID pair
        else
            push!(time_displaced_correlations[correlation].bond_id_pairs, (0,0)) # record null bond ID pair to be filled in later
        end
        push!(time_displaced_correlations[correlation].correlations, zeros(Complex{T}, L..., Lτ+1))

        # add integrated correlation measurement
        push!(integrated_correlations[correlation].id_pairs, pair)
        if (CORRELATION_FUNCTIONS[correlation] == "BOND_ID") || (CORRELATION_FUNCTIONS[correlation] == "ORBITAL_ID")
            push!(integrated_correlations[correlation].bond_id_pairs, pair) # record bond ID pair as same as correlation ID pair
        else
            push!(integrated_correlations[correlation].bond_id_pairs, (0,0)) # record null bond ID pair to be filled in later
        end
        push!(integrated_correlations[correlation].correlations, zeros(Complex{T}, L...))
    end

    # if equal-time measurement should be made
    if !time_displaced

        # add equal-time correlation key if not present
        if !haskey(equaltime_correlations, correlation)
            equaltime_correlations[correlation] = CorrelationContainer(D, T, false)
        end

        # add equal-time correlation measurement
        push!(equaltime_correlations[correlation].id_pairs, pair)
        if (CORRELATION_FUNCTIONS[correlation] == "BOND_ID") || (CORRELATION_FUNCTIONS[correlation] == "ORBITAL_ID")
            push!(equaltime_correlations[correlation].bond_id_pairs, pair) # record bond ID pair as same as correlation ID pair
        else
            push!(equaltime_correlations[correlation].bond_id_pairs, (0,0)) # record null bond ID pair to be filled in later
        end
        if translation == true
            push!(equaltime_correlations[correlation].correlations, zeros(Complex{T}, L...))
        else
            push!(equaltime_correlations[correlation].correlations, zeros(Complex{T}, prod(L),prod(L)))
        end
    end

    return nothing
end


################################################
## INITIALIZE MEASUREMENT DIRECTORY STRUCTURE ##
################################################

@doc raw"""
    initialize_measurement_directories(;
        simulation_info::SimulationInfo,
        measurement_container::NamedTuple
    )

Initialize the measurement directories for simulation.
"""
function initialize_measurement_directories(;
    simulation_info::SimulationInfo,
    measurement_container::NamedTuple
)

    (; datafolder, resuming, pID) = simulation_info
    (; time_displaced_correlations, equaltime_correlations, integrated_correlations) = measurement_container

    # only initialize folders if pID = 0
    if iszero(pID) && !resuming

        # make global measurements directory
        global_directory = joinpath(datafolder, "global")
        mkdir(global_directory)

        # make local measurements directory
        local_directory = joinpath(datafolder, "local")
        mkdir(local_directory)

        # make equaltime correlation directory
        eqaultime_directory = joinpath(datafolder, "equal-time")
        mkdir(eqaultime_directory)

        # iterate over equal-time correlation measurements
        for correlation in keys(equaltime_correlations)

            # make directory for each individual eqaul-time correlation measurement
            equaltime_correlation_directory = joinpath(eqaultime_directory, correlation)
            mkdir(equaltime_correlation_directory)

            # create sub-directories for position and momentum space data
            mkdir(joinpath(equaltime_correlation_directory, "position"))
            mkdir(joinpath(equaltime_correlation_directory, "momentum"))
        end

        # make time-displaced correlation directory
        time_displaced_directory = joinpath(datafolder, "time-displaced")
        mkdir(time_displaced_directory)

        # make integrated correlation directory
        integrated_directory = joinpath(datafolder, "integrated")
        mkdir(integrated_directory)

        # iterate over integrated correlation measurements
        for correlation in keys(integrated_correlations)

            # make directory for integrated correlation measurement
            integrated_correlation_directory = joinpath(integrated_directory, correlation)
            mkdir(integrated_correlation_directory)

            # create sub-directories for position and momentum space time-displaced correlation measurements
            mkdir(joinpath(integrated_correlation_directory, "position"))
            mkdir(joinpath(integrated_correlation_directory, "momentum"))

            # check if also a time-displaced measurement should also be made
            if time_displaced_correlations[correlation].time_displaced

                # make directory for time-displaced correlation measurement
                time_displaced_correlation_directory = joinpath(time_displaced_directory, correlation)
                mkdir(time_displaced_correlation_directory)

                # create sub-directories for position and momentum space time-displaced correlation measurements
                mkdir(joinpath(time_displaced_correlation_directory, "position"))
                mkdir(joinpath(time_displaced_correlation_directory, "momentum"))
            end
        end
    end

    return nothing
end