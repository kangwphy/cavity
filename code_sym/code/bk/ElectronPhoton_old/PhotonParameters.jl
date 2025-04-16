@doc raw"""
    PhotonParameters{E<:AbstractFloat}

Defines the parameters for each photon in the lattice, includes the photon field configuration.

# Fields

- `nphoton::Int`: Number of type of photon modes.
- `Nphoton::Int`: Total number of photon modes in finite lattice.
- `M::Int`: Mass of each photon mode.
- `Ω::Int`: Frequency of each photon mode.
- `Ω4::Int`: Quartic photon coefficient for each photon mode.
- `photon_to_site::Vector{Int}`: Map each photon to the site it lives on in the lattice.
- `site_to_photons::Vector{Vector{Int}}`: Maps the site to the photon modes on it, allowing for multiple modes to reside on a single site.
"""
struct PhotonParameters{E<:AbstractFloat}

    # number of types of photon modes
    nphoton::Int

    # number of photon modes
    Nphoton::Int

    # photon masses
    M::Vector{E}

    # photon frequency
    Ω::Vector{E}

    # quartic coefficient for photon potential energy (X⁴)
    Ω4::Vector{E}

    # map photon field to site in lattice
    photon_to_site::Vector{Int}

    # map sites to photon fields (note that multiple fields may live on a single site)
    site_to_photons::Vector{Vector{Int}}
end

@doc raw"""
    PhotonParameters(; model_geometry::ModelGeometry{D,E},
                     electron_photon_model::ElectronPhotonModel{T,E,D},
                     rng::AbstractRNG) where {T,E,D}

Initialize and return an instance of [`PhotonParameters`](@ref).
"""
function PhotonParameters(; model_geometry::ModelGeometry{D,E},
                          electron_photon_model::ElectronPhotonModel{T,E,D},
                          rng::AbstractRNG) where {T,E,D}

    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}

    # get totals number of sites/orbitals in lattice
    Nsites = nsites(unit_cell, lattice)

    # get number of unit cells
    Ncells = lattice.N

    # get the photon mode defintions
    photon_modes = electron_photon_model.photon_modes::Vector{PhotonMode{E}}

    # get the number of photon mode definitions
    nphoton = length(photon_modes)

    # get the total number of photon modes in the lattice
    Nphoton = nphoton * 1

    # allocate array of masses for each photon mode
    M = zeros(E,Nphoton)

    # allocate array of photon frequncies for each photon mode
    Ω = zeros(E,Nphoton)

    # allocate array of quartic coefficient for each photon mode
    Ω4 = zeros(E,Nphoton)

    # allocate photon_to_site
    photon_to_site = zeros(Int, Nphoton)

    # allocate site_to_photons
    site_to_photons = [Int[] for i in 1:Nsites]

    # iterate over photon modes
    photon = 0 # photon counter
    for nph in 1:nphoton
        # get the photon mode
        photon_mode = photon_modes[nph]::PhotonMode{E}
        # get the orbital species associated with photon mode
        orbital = photon_mode.orbital
        # iterate over unit cells in lattice
        for uc in 1:1
            # increment photon counter
            photon += 1
            # get site associated with photon mode
            site = loc_to_site(uc, orbital, unit_cell)
            # record photon ==> site
            photon_to_site[photon] = site
            # record site ==> photon
            push!(site_to_photons[site], photon)
            # assign photon mass
            M[photon] = photon_mode.M
            # assign photon freuqency
            Ω[photon] = photon_mode.Ω_mean + photon_mode.Ω_std * randn(rng)
            # assign quartic photon coefficient
            Ω4[photon] = photon_mode.Ω4_mean + photon_mode.Ω4_std * randn(rng)
        end
    end

    return PhotonParameters(nphoton, Nphoton, M, Ω, Ω4, photon_to_site, site_to_photons)
end