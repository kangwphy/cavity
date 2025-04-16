@doc raw"""
    SSHParameters{T<:Number}

Defines the SSH coupling parameters in lattice.

# Fields

- `nssh::Int`: Number of types of SSH couplings.
- `Nssh::Int`: Number of SSH couplings in lattice.
- `α::Vector{T}`: Linear SSH coupling.
- `α2::Vector{T}`: Quadratic SSH coupling.
- `α3::Vector{T}`: Cubic SSH coupling.
- `α4::Vector{T}`: Quartic SSH coupling.`
- `neighbor_table::Matrix{Int}`: Neighbor table to SSH coupling.
- `coupling_to_photon::Matrix{Int}`: Maps each SSH coupling onto that pair of coupled photons.
- `init_photon_to_coupling::Vector{Vector{Int}}`: Maps initial photon mode to corresponding SSH coupling(s).
- `final_photon_to_coupling::Vector{Vector{Int}}`: Maps final photon mode to corresponding SSH coupling(s).
- `hopping_to_couplings::Vector{Vector{Int}}`: Maps hopping in the tight-binding model onto SSH couplings.
- `coupling_to_hopping::Vector{Int}`: Maps each SSH coupling onto the corresponding hopping in the tight-binding model.
"""
struct SSHParameters{T<:Number}

    # number of types of SSH couplings
    nssh::Int

    # number of ssh couplings in lattice
    Nssh::Int

    # linear coupling
    α::Vector{T}

    # quadratic coupling
    α2::Vector{T}

    # cubic coupling
    α3::Vector{T}

    # quartic coupling
    α4::Vector{T}

    # ssh neighbor table
    neighbor_table::Matrix{Int}

    # map ssh coupling to photon mode
    coupling_to_photon::Matrix{Int}

    # initial photon to coupling
    init_photon_to_coupling::Vector{Vector{Int}}

    # initial photon to coupling
    final_photon_to_coupling::Vector{Vector{Int}}

    # map hopping in bare tight binding model to ssh coupling
    hopping_to_couplings::Vector{Vector{Int}}
    
    # map coupling to bare hopping in tight binding model
    coupling_to_hopping::Vector{Int}
end

@doc raw"""
    SSHParameters(;
        model_geometry::ModelGeometry{D,E},
        electron_photon_model::ElectronPhotonModel{T,E,D},
        tight_binding_parameters_up::TightBindingParameters{T,E},
        tight_binding_parameters_dn::TightBindingParameters{T,E},
        rng::AbstractRNG
    ) where {T,E,D}

Initialize and return an instance of [`SSHParameters`](@ref).
"""
function SSHParameters(;
    model_geometry::ModelGeometry{D,E},
    electron_photon_model::ElectronPhotonModel{T,E,D},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E},
    rng::AbstractRNG
) where {T,E,D}

    ssh_couplings_up = electron_photon_model.ssh_couplings_up::Vector{SSHCoupling{T,E,D}}
    ssh_couplings_dn = electron_photon_model.ssh_couplings_dn::Vector{SSHCoupling{T,E,D}}
    photon_modes = electron_photon_model.photon_modes::Vector{PhotonMode{E}}
    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}

    # number holstein coupling definitions
    nssh = length(ssh_couplings_up)

    if nssh>0

        # get number of types of photon models
        nphoton = length(photon_modes)

        # get the number of unit cells in the lattice
        Ncells = lattice.N

        # total number of holstein couplings
        Nssh = nssh * Ncells

        # get total number of photon modes
        Nphoton = nphoton * Ncells

        # get the bare tight binding model hopping neighbor table
        hopping_neighbor_table = tight_binding_parameters_up.neighbor_table

        # get the total number of hoppings in lattice
        Nhoppings = size(hopping_neighbor_table,2)

        # get bare hopping bond ids
        hopping_bond_ids = tight_binding_parameters_up.bond_ids::Vector{Int}

        # get the slice of hopping neighbor table associated with each bond ID
        hopping_bond_slices = tight_binding_parameters_up.bond_slices::Vector{UnitRange{Int}}

        # allocate arrays of ssh coupling parameters
        α_up  = zeros(T, Nssh)
        α2_up = zeros(T, Nssh)
        α3_up = zeros(T, Nssh)
        α4_up = zeros(T, Nssh)
        α_dn  = zeros(T, Nssh)
        α2_dn = zeros(T, Nssh)
        α3_dn = zeros(T, Nssh)
        α4_dn = zeros(T, Nssh)

        # allocate mapping arrays
        coupling_to_photon   = zeros(Int, 2, Nssh)
        coupling_to_hopping  = zeros(Int, Nssh)
        hopping_to_couplings = [Int[] for _ in 1:Nhoppings]

        # get all the ssh bonds
        ssh_bonds = [ssh_coupling.bond for ssh_coupling in ssh_couplings_up]

        # construct neighbor table for ssh couplings
        ssh_neighbor_table = build_neighbor_table(ssh_bonds, unit_cell, lattice)

        # iterate over ssh coupling definitions
        ssh_counter = 0 # ssh coupling counter
        for sc in 1:nssh
            # get the ssh coupling definition
            ssh_coupling_up = ssh_couplings_up[sc]
            ssh_coupling_dn = ssh_couplings_dn[sc]
            # get the pair of photon mode definitions assoicated with ssh coupling
            photon_mode_i = ssh_coupling_up.photon_modes[1]
            photon_mode_f = ssh_coupling_up.photon_modes[2]
            # get the bond id associated with the ssh coupling
            ssh_bond_id = ssh_coupling_up.bond_id
            # get range/slice of bare hoppings that need to be iterated over for given bond_id
            hopping_bond_id_index = findfirst(hopping_bond_id -> hopping_bond_id == ssh_bond_id, hopping_bond_ids)
            hopping_bond_slice = hopping_bond_slices[hopping_bond_id_index]
            # iterate over (unit cells) and (bare hopping corresponding to bond ID)
            for (unit_cell_id, hopping_index) in enumerate(hopping_bond_slice)
                # increment ssh coupling counter
                ssh_counter += 1
                # record ssh coupling <==> bare hopping mapping
                coupling_to_hopping[ssh_counter] = hopping_index
                push!(hopping_to_couplings[hopping_index], ssh_counter)
                # record the initial photon
                coupling_to_photon[1,ssh_counter] = Ncells * (photon_mode_i-1) + unit_cell_id
                # get the site the final photon lives on
                site_id_final = ssh_neighbor_table[2, ssh_counter]
                # get the unit cell the final photon lives on from the site it lives on
                unit_cell_id_final = site_to_unitcell(site_id_final, unit_cell)
                # record the final photon
                coupling_to_photon[2,ssh_counter] = Ncells * (photon_mode_f-1) + unit_cell_id_final
                # initialize coupling parameters
                α_up[ssh_counter]  = ssh_coupling_up.α_mean  + ssh_coupling_up.α_std  * randn(rng)
                α2_up[ssh_counter] = ssh_coupling_up.α2_mean + ssh_coupling_up.α2_std * randn(rng)
                α3_up[ssh_counter] = ssh_coupling_up.α3_mean + ssh_coupling_up.α3_std * randn(rng)
                α4_up[ssh_counter] = ssh_coupling_up.α4_mean + ssh_coupling_up.α4_std * randn(rng)
                α_dn[ssh_counter]  = ssh_coupling_dn.α_mean  + ssh_coupling_dn.α_std  * randn(rng)
                α2_dn[ssh_counter] = ssh_coupling_dn.α2_mean + ssh_coupling_dn.α2_std * randn(rng)
                α3_dn[ssh_counter] = ssh_coupling_dn.α3_mean + ssh_coupling_dn.α3_std * randn(rng)
                α4_dn[ssh_counter] = ssh_coupling_dn.α4_mean + ssh_coupling_dn.α4_std * randn(rng)
            end
        end

        # construct photon to coupling maps
        init_photon_to_coupling  = Vector{Int}[]
        final_photon_to_coupling = Vector{Int}[]
        for photon in 1:Nphoton
            coupling_to_init_photon  = @view coupling_to_photon[1,:]
            coupling_to_final_photon = @view coupling_to_photon[2,:]
            push!(init_photon_to_coupling, findall(i -> i==photon, coupling_to_init_photon))
            push!(final_photon_to_coupling, findall(i -> i==photon, coupling_to_final_photon))
        end

        # initialize ssh parameters
        ssh_parameters_up = SSHParameters(
            nssh, Nssh, α_up, α2_up, α3_up, α4_up, ssh_neighbor_table, coupling_to_photon,
            init_photon_to_coupling, final_photon_to_coupling,
            hopping_to_couplings, coupling_to_hopping
        )
        ssh_parameters_dn = SSHParameters(
            nssh, Nssh, α_dn, α2_dn, α3_dn, α4_dn, ssh_neighbor_table, coupling_to_photon,
            init_photon_to_coupling, final_photon_to_coupling,
            hopping_to_couplings, coupling_to_hopping
        )

    else

        # initialize null ssh parameters
        ssh_parameters_up = SSHParameters(electron_photon_model)
        ssh_parameters_dn = SSHParameters(electron_photon_model)
    end

    return ssh_parameters_up, ssh_parameters_dn
end

@doc raw"""
    SSHParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

Initialize and return null (empty) instance of [`SSHParameters`](@ref).
"""
function SSHParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

    return SSHParameters(0, 0, T[], T[], T[], T[], Matrix{Int}(undef,2,0), Matrix{Int}(undef,2,0), Vector{Int}[], Vector{Int}[], Vector{Int}[], Int[])
end

# Update the total hopping energy for each time-slice based on the SSH interaction
# and the photon field configuration `x`, where `sgn = ±1` determines whether the SSH
# contribution to the total hopping energy is either added or subtracted.
function update!(fermion_path_integral::FermionPathIntegral{T,E},
                 ssh_parameters::SSHParameters{T},
                 x::Matrix{E}, sgn::Int) where {T,E}

    (; t, Lτ) = fermion_path_integral
    (; Nssh, α, α2, α3, α4, coupling_to_photon, coupling_to_hopping) = ssh_parameters

    # if ssh coupling present
    if Nssh > 0
        # iterate over imaginary time slice
        @fastmath @inbounds for l in 1:Lτ
            # iterate over ssh couplinges
            for i in 1:Nssh
                # get pair of photons
                p  = coupling_to_photon[1,i]
                p′ = coupling_to_photon[2,i]
                # get the hopping index
                h = coupling_to_hopping[i]
                # calculate the relative photon position
                Δx = x[p′,l] - x[p,l]
                # update total hopping amplitude
                t[h,l] += -sgn * (α[i]*Δx + α2[i]*Δx^2 + α3[i]*Δx^3 + α4[i]*Δx^4)
            end
        end
    end
    # @show t
    return nothing
end