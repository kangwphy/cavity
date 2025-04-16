@doc raw"""
    HolsteinParameters{E<:AbstractFloat}

Defines the Holstein coupling parameters in lattice.

# Fields

- `nholstein::Int`: The number of type of holstein couplings.
- `Nholstein::Int`: Total number of Holstein couplings in lattice.
- `α::Vector{T}`: Linear Holstein coupling.
- `α2::Vector{T}`: Quadratic Holstein coupling.
- `α3::Vector{T}`: Cubic Holstein coupling.
- `α4::Vector{T}`: Quartic Holstein coupling.
- `shifted::Vector{Bool}`: If the density multiplying the odd powered interaction terms is shifted.
- `neighbor_table::Matrix{Int}`: Neighbor table where the first row specifies the site where the photon mode is located, and the second row specifies the site corresponds to the density getting coupled to.
- `coupling_to_photon::Vector{Int}`: Maps each Holstein coupling in the lattice to the corresponding photon mode.
- `photon_to_coupling::Vector{Vector{Int}}`: Maps each photon model to correspond Holstein couplings.
"""
struct HolsteinParameters{E<:AbstractFloat}

    # number of type of holstien couplings
    nholstein::Int

    # number of Holstein couplings
    Nholstein::Int

    # linear coupling
    α::Vector{E}

    # quadratic coupling
    α2::Vector{E}

    # cubic coupling
    α3::Vector{E}

    # quartic coupling
    α4::Vector{E}

    # whether the density multiplying the odd powered interaction terms are shifted
    shifted::Vector{Bool}

    # neighbor table for couplings where first row is site photon lives on,
    # and the second row is the site where density getting coupled to is
    neighbor_table::Matrix{Int}

    # map coupling to photon
    coupling_to_photon::Vector{Int}

    # map photon to coupling
    photon_to_coupling::Vector{Vector{Int}}
end

@doc raw"""
    HolsteinParameters(;
        model_geometry::ModelGeometry{D,E},
        electron_photon_model::ElectronPhotonModel{T,E,D},
        rng::AbstractRNG,
    ) where {T,E,D}

Initialize and return an instance of [`HolsteinParameters`](@ref).
"""
function HolsteinParameters(;
    model_geometry::ModelGeometry{D,E},
    electron_photon_model::ElectronPhotonModel{T,E,D},
    rng::AbstractRNG,
) where {T,E,D}

    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}
    photon_modes = electron_photon_model.photon_modes::Vector{PhotonMode{E}}
    holstein_couplings_up = electron_photon_model.holstein_couplings_up::Vector{HolsteinCoupling{E,D}}
    holstein_couplings_dn = electron_photon_model.holstein_couplings_dn::Vector{HolsteinCoupling{E,D}}

    # number holstein coupling definitions
    nholstein = length(holstein_couplings_up)

    if nholstein > 0

        # get number of types of photon models
        nphoton = length(photon_modes)
            
        # get the number of unit cells in the lattice
        Ncells = lattice.N

        # total number of holstein couplings
        Nholstein = nholstein * Ncells

        # get total number of photon modes
        Nphoton = nphoton * Ncells

        # build the neighbor table for the holstein couplings
        holstein_bonds = [holstein_coupling.bond for holstein_coupling in holstein_couplings_up]
        neighbor_table = build_neighbor_table(holstein_bonds, unit_cell, lattice)

        # allocate arrays for holstein coupling parameters
        α_up  = zeros(E, Nholstein)
        α2_up = zeros(E, Nholstein)
        α3_up = zeros(E, Nholstein)
        α4_up = zeros(E, Nholstein)
        α_dn  = zeros(E, Nholstein)
        α2_dn = zeros(E, Nholstein)
        α3_dn = zeros(E, Nholstein)
        α4_dn = zeros(E, Nholstein)

        # whether type of holstein coupling term is shifted
        shifted_up = [holstein_coupling.shifted for holstein_coupling in holstein_couplings_up]
        shifted_dn = [holstein_coupling.shifted for holstein_coupling in holstein_couplings_dn]

        # allocate arrays mapping holstein coupling to photon in lattice
        coupling_to_photon = zeros(Int, Nholstein)

        # iterate over holstein coupling defintitions
        holstein_counter = 0 # holstein coupling counter
        for hc in 1:nholstein

            # get the holstein coupling definition
            holstein_coupling_up = holstein_couplings_up[hc]
            holstein_coupling_dn = holstein_couplings_dn[hc]
            # get the photon mode definition/ID associated with holstein coupling
            photon_mode = holstein_coupling_up.photon_mode

            # iterate over unit cells
            for uc in 1:Ncells

                # increment holstein coupling counter
                holstein_counter += 1
                # get the photon mode getting coupled to
                photon = Ncells*(photon_mode-1) + uc
                # record the photon mode associated with the coupling
                coupling_to_photon[holstein_counter] = photon
                # initialize coupling parameters
                α_up[holstein_counter]  = holstein_coupling_up.α_mean  + holstein_coupling_up.α_std  * randn(rng)
                α2_up[holstein_counter] = holstein_coupling_up.α2_mean + holstein_coupling_up.α2_std * randn(rng)
                α3_up[holstein_counter] = holstein_coupling_up.α3_mean + holstein_coupling_up.α3_std * randn(rng)
                α4_up[holstein_counter] = holstein_coupling_up.α4_mean + holstein_coupling_up.α4_std * randn(rng)
                α_dn[holstein_counter]  = holstein_coupling_dn.α_mean  + holstein_coupling_dn.α_std  * randn(rng)
                α2_dn[holstein_counter] = holstein_coupling_dn.α2_mean + holstein_coupling_dn.α2_std * randn(rng)
                α3_dn[holstein_counter] = holstein_coupling_dn.α3_mean + holstein_coupling_dn.α3_std * randn(rng)
                α4_dn[holstein_counter] = holstein_coupling_dn.α4_mean + holstein_coupling_dn.α4_std * randn(rng)
            end
        end

        # construct photon to coupling map
        photon_to_coupling = Vector{Int}[]
        for photon in 1:Nphoton
            push!(photon_to_coupling, findall(i -> i==photon, coupling_to_photon))
        end

        # initialize holstein parameters
        holstein_parameters_up = HolsteinParameters(nholstein, Nholstein, α_up, α2_up, α3_up, α4_up, shifted_up, neighbor_table, coupling_to_photon, photon_to_coupling)
        holstein_parameters_dn = HolsteinParameters(nholstein, Nholstein, α_dn, α2_dn, α3_dn, α4_dn, shifted_dn, neighbor_table, coupling_to_photon, photon_to_coupling)
    else

        # initialize null holstein parameters
        holstein_parameters_up = HolsteinParameters(electron_photon_model)
        holstein_parameters_dn = HolsteinParameters(electron_photon_model)
    end

    return holstein_parameters_up, holstein_parameters_dn
end

@doc raw"""
    HolsteinParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

Initialize and return null (empty) instance of [`HolsteinParameters`](@ref).
"""
function HolsteinParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

    return HolsteinParameters(0, 0, E[], E[], E[], E[], Bool[], Matrix{Int}(undef,2,0), Int[], Vector{Int}[])
end

# Update the on-site energy matrix for each time-slice based on the Holstein interaction
# and the photon field configuration `x`, where `sgn = ±1` determines whether the Holstein
# contribution to the on-site energy matrix is either being added or subtracted.
function update!(fermion_path_integral::FermionPathIntegral{T,E},
                 holstein_parameters::HolsteinParameters{E},
                 x::Matrix{E}, sgn::Int) where {T,E}

    (; V, Lτ) = fermion_path_integral
    (; Nholstein, α, α2, α3, α4, coupling_to_photon, neighbor_table) = holstein_parameters

    # if holstein interaction present
    if Nholstein > 0
        # iterate over imaginary time slice
        @fastmath @inbounds for l in 1:Lτ
            # iterate over holstein couplings
            for i in 1:Nholstein
                # get the photon mode associated with the photon coupling
                p = coupling_to_photon[i]
                # get the orbital/site whose denisty the photon is coupling to
                s = neighbor_table[2,i] # second row corresponds to site with density getting coupled to,
                                        # the first row give the site the photon lives on, which we don't need here
                # update diagonal on-site energy matrix
                V[s,l] += sgn * (α[i]*x[p,l] + α2[i]*x[p,l]^2 + α3[i]*x[p,l]^3 + α4[i]*x[p,l]^4)
            end
        end
    end

    return nothing
end
