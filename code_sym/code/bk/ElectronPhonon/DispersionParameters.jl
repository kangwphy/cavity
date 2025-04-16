@doc raw"""
    DispersionParameters{E<:AbstractFloat}

Defines the dispersive photon coupling parameters in the lattice.

# Fields

- `ndispersion::Int`: Number of types of dispersive couplings.
- `Ndispersion::Int`: Number of dispersive couplings in the lattice.
- `Ω::Vector{E}`: Frequency of dispersive photon coupling.
- `Ω4::Vector{E}`: Quartic coefficient for the photon dispersion.
- `dispersion_to_photon::Matrix{Int}`: Pair of photon modes in lattice coupled by dispersive coupling.
- `init_photon_to_coupling::Vector{Vector{Int}}`: Maps initial photon mode to corresponding dispersive photon coupling.
- `final_photon_to_coupling::Vector{Vector{Int}}`: Maps final photon mode to corresponding dispersive photon coupling.
"""
struct DispersionParameters{E<:AbstractFloat}

    # number of types of dispersive couplings
    ndispersion::Int

    # number of dispersive couplings
    Ndispersion::Int

    # photon frequency
    Ω::Vector{E}

    # quartic coefficient for photon potential energy (X⁴)
    Ω4::Vector{E}

    # map dispersion to photon mode
    dispersion_to_photon::Matrix{Int}

    # initial photon mapping to dispersion
    init_photon_to_dispersion::Vector{Vector{Int}}

    # final photon mapping to dispersion
    final_photon_to_dispersion::Vector{Vector{Int}}
end

@doc raw"""
    DispersionParameters(; model_geometry::ModelGeometry{D,E},
                         electron_photon_model::ElectronPhotonModel{T,E,D},
                         photon_parameters::PhotonParameters{E},
                         rng::AbstractRNG) where {T,E,D}

Initialize and return an instance of [`DispersionParameters`](@ref).
"""
function DispersionParameters(; model_geometry::ModelGeometry{D,E},
                              electron_photon_model::ElectronPhotonModel{T,E,D},
                              photon_parameters::PhotonParameters{E},
                              rng::AbstractRNG) where {T,E,D}

    photon_dispersions = electron_photon_model.photon_dispersions::Vector{PhotonDispersion{E,D}}
    photon_modes = electron_photon_model.photon_modes::Vector{PhotonMode{E}}
    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}

    # the number of dispersive photon coupling definitions
    ndispersion = length(photon_dispersions)

    if ndispersion > 0

        # get number of types of photon models
        nphoton = length(photon_modes)

        # get photon to size map
        photon_to_site = photon_parameters.photon_to_site

        # get the number of unit cells in the lattice
        Ncells = lattice.N

        # get total number of photon modes
        Nphoton = nphoton * Ncells

        # total number of disperson photon couplings in lattice
        Ndispersion = ndispersion * Ncells

        # get dispersive bonds
        dispersion_bonds = [photon_dispersion.bond for photon_dispersion in photon_dispersions]

        # build site neighbor table
        site_neighbor_table = build_neighbor_table(dispersion_bonds, unit_cell, lattice)

        # dispersion to photon mapping
        dispersion_to_photon = similar(site_neighbor_table)

        # allocate dispersive coupling coefficients
        Ω  = zeros(E, Ndispersion)
        Ω4 = zeros(E, Ndispersion) 

        # iterate over dispersive coupling definition
        dipsersion_counter = 0 # count dispersive couplings
        for n in 1:ndispersion
            # get the dispersive coupling definition
            photon_dispersion = photon_dispersions[n]
            # get the two photon mode definitions that are coupled
            photon_mode_i = photon_dispersion.photon_modes[1]
            photon_mode_f = photon_dispersion.photon_modes[2]
            # iterate over unit cells
            for uc in 1:Ncells
                # increment dispersive coupling counter
                dipsersion_counter += 1
                # initialize dispersive coupling coefficient
                Ω[dipsersion_counter]  = photon_dispersion.Ω_mean  + photon_dispersion.Ω_std  * randn(rng)
                Ω4[dipsersion_counter] = photon_dispersion.Ω4_mean + photon_dispersion.Ω4_std * randn(rng)
                # record the initial photon getting coupled to
                photon_i = (photon_mode_i - 1) * Ncells + uc
                dispersion_to_photon[1, dipsersion_counter] = photon_i
                # get view into photons photon_to_site that corresponds to final photon type
                photons = @view photon_to_site[(photon_mode_f-1)*Ncells + 1 : photon_mode_f*Ncells]
                # get the terminating site of the photon dispersion
                site_f = site_neighbor_table[2,dipsersion_counter]
                # get the corresponding photon in the lattice
                photon_f = findfirst(i -> i==site_f, photons) + (photon_mode_f - 1) * Ncells
                dispersion_to_photon[2, dipsersion_counter] = photon_f
            end
        end

        # construct photon to dispersive coupling map
        init_photon_to_dispersion  = Vector{Int}[]
        final_photon_to_dispersion = Vector{Int}[]
        for photon in 1:Nphoton
            dispersion_to_init_photon  = @view dispersion_to_photon[1,:]
            dispersion_to_final_photon = @view dispersion_to_photon[2,:]
            push!(init_photon_to_dispersion, findall(i -> i==photon, dispersion_to_init_photon))
            push!(final_photon_to_dispersion, findall(i -> i==photon, dispersion_to_final_photon))
        end

        # initialize dispersion parameters
        dispersion_parameters = DispersionParameters(ndispersion, Ndispersion, Ω, Ω4, dispersion_to_photon,
                                                     init_photon_to_dispersion, final_photon_to_dispersion)
    else

        # initialize null dispersion parameters
        dispersion_parameters = DispersionParameters(electron_photon_model)
    end

    return dispersion_parameters
end

@doc raw"""
    DispersionParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

Initialize and return null (empty) instance of [`DispersionParameters`](@ref).
"""
function DispersionParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

    return DispersionParameters(0, 0, E[], E[], Matrix{Int}(undef,2,0), Vector{Int}[], Vector{Int}[])
end