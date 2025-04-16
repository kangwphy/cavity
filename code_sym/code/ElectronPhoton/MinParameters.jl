@doc raw"""
    MinParameters{T<:Number}

Defines the Min coupling parameters in lattice.

# Fields

- `nmin::Int`: Number of types of Min couplings.
- `Nmin::Int`: Number of Min couplings in lattice.
- `α::Vector{T}`: Linear Min coupling.
- `α2::Vector{T}`: Quadratic Min coupling.
- `α3::Vector{T}`: Cubic Min coupling.
- `α4::Vector{T}`: Quartic Min coupling.`
- `neighbor_table::Matrix{Int}`: Neighbor table to Min coupling.
- `coupling_to_photon::Matrix{Int}`: Maps each Min coupling onto that pair of coupled photons.
- `init_photon_to_coupling::Vector{Vector{Int}}`: Maps initial photon mode to corresponding Min coupling(s).
- `final_photon_to_coupling::Vector{Vector{Int}}`: Maps final photon mode to corresponding Min coupling(s).
- `hopping_to_couplings::Vector{Vector{Int}}`: Maps hopping in the tight-binding model onto Min couplings.
- `coupling_to_hopping::Vector{Int}`: Maps each Min coupling onto the corresponding hopping in the tight-binding model.
"""
struct MinParameters{T<:Number}

    # number of types of Min couplings
    nmin::Int

    # number of min couplings in lattice
    Nmin::Int

    # linear coupling
    α::Vector{T}

    # quadratic coupling
    α2::Vector{T}

    # cubic coupling
    α3::Vector{T}

    # quartic coupling
    α4::Vector{T}

    # coupling amplitute
    tc::Vector{T}
    # min neighbor table
    neighbor_table::Matrix{Int}

    # # map min coupling to photon mode
    # coupling_to_photon::Matrix{Int}

    # # initial photon to coupling
    # init_photon_to_coupling::Vector{Vector{Int}}

    # # initial photon to coupling
    # final_photon_to_coupling::Vector{Vector{Int}}

    # # map hopping in bare tight binding model to min coupling
    # hopping_to_couplings::Vector{Vector{Int}}
    
    # map coupling to bare hopping in tight binding model
    coupling_to_hopping::Vector{Int}
end

@doc raw"""
    MinParameters(;
        model_geometry::ModelGeometry{D,E},
        electron_photon_model::ElectronPhotonModel{T,E,D},
        tight_binding_parameters_up::TightBindingParameters{T,E},
        tight_binding_parameters_dn::TightBindingParameters{T,E},
        rng::AbstractRNG
    ) where {T,E,D}

Initialize and return an instance of [`MinParameters`](@ref).
"""
function MinParameters(;
    model_geometry::ModelGeometry{D,E},
    electron_photon_model::ElectronPhotonModel{T,E,D},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E},
    PBCx,PBCy,
    rng::AbstractRNG
) where {T,E,D}

    min_couplings_up = electron_photon_model.min_couplings_up::Vector{MinCoupling{T,E,D}}
    min_couplings_dn = electron_photon_model.min_couplings_dn::Vector{MinCoupling{T,E,D}}
    photon_modes = electron_photon_model.photon_modes::Vector{PhotonMode{E}}
    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}

    # number holstein coupling definitions
    nmin = length(min_couplings_up)
    # @show nmin
    if true

        # get number of types of photon models
        nphoton = length(photon_modes)

        # get the number of unit cells in the lattice
        Ncells = lattice.N

        # get total number of photon modes
        Nphoton = nphoton * 1

        # get the bare tight binding model hopping neighbor table
        hopping_neighbor_table = tight_binding_parameters_up.neighbor_table

        # get the total number of hoppings in lattice
        Nhoppings = size(hopping_neighbor_table,2)

        

        # get bare hopping bond ids
        hopping_bond_ids = tight_binding_parameters_up.bond_ids::Vector{Int}

        # get the slice of hopping neighbor table associated with each bond ID
        hopping_bond_slices = tight_binding_parameters_up.bond_slices::Vector{UnitRange{Int}}


        

        # allocate mapping arrays
        # coupling_to_photon   = zeros(Int, 2, Nmin)
        # coupling_to_hopping  = zeros(Int, Nmin)
        # hopping_to_couplings = [Int[] for _ in 1:Nhoppings]

        # get all the min bonds
        min_bonds = [min_coupling.bonds for min_coupling in min_couplings_up]
        # construct neighbor table for min couplings
        min_neighbor_table = build_neighbor_table(min_bonds, unit_cell, lattice)

     
        # total number of holstein couplings
        Nmin = nmin * size(min_neighbor_table,2)
        # allocate arrays of min coupling parameters
        α_up  = zeros(T, Nmin)
        α2_up = zeros(T, Nmin)
        α3_up = zeros(T, Nmin)
        α4_up = zeros(T, Nmin)
        α_dn  = zeros(T, Nmin)
        α2_dn = zeros(T, Nmin)
        α3_dn = zeros(T, Nmin)
        α4_dn = zeros(T, Nmin)
        tc = zeros(T, Nmin)
        Nmin = 1
        # @show min_neighbor_table, hopping_neighbor_table

    
        # @show min_neighbor_table
        # iterate over min coupling definitions
        # min_counter = 0 # min coupling counter
        coupling_to_hopping = zeros(Int, size(min_neighbor_table,2))
        for sc in 1:nmin
        #     # get the min coupling definition
            min_coupling_up = min_couplings_up[sc]
            min_coupling_dn = min_couplings_dn[sc]
            @show size(min_neighbor_table), size(hopping_neighbor_table)
            # ddd
            min_counter = 0 # ssh coupling counter
            for ith in 1:size(min_neighbor_table,2)
                #### make sure the min bonds are in the hopping bonds set 
                exist = true
                for jth in 1:size(hopping_neighbor_table,2)
                    # @show ith,jth
                    if min_neighbor_table[:,ith] == hopping_neighbor_table[:,jth]
                        # @show jth
                        # push!(coupling_to_hopping, jth)
                        coupling_to_hopping[min_counter+1] = jth
                        exist = true
                        break
                    end
                end
                if !exist
                    error("Not exist in hopping bonds")
                end


                min_counter += 1
                # num = (y)*Lx+x
                # (num - 1) % Lx
                x1 = (min_neighbor_table[1,ith]-1)%lattice.L[1]
                x2 = (min_neighbor_table[2,ith]-1)%lattice.L[1]

                y1 = div(min_neighbor_table[1,ith]-1,lattice.L[1])
                y2 = div(min_neighbor_table[2,ith]-1,lattice.L[1])
                if y1 == y2
                    α_up[min_counter]  = y1*min_coupling_up.α_mean/2  + min_coupling_up.α_std  * randn(rng)
                    α2_up[min_counter] = y1*min_coupling_up.α2_mean/2 + min_coupling_up.α2_std * randn(rng)
                    α3_up[min_counter] = y1*min_coupling_up.α3_mean/2 + min_coupling_up.α3_std * randn(rng)
                    α4_up[min_counter] = y1*min_coupling_up.α4_mean/2 + min_coupling_up.α4_std * randn(rng)
        
                    α_dn[min_counter]  = y1*min_coupling_dn.α_mean/2  + min_coupling_dn.α_std  * randn(rng)
                    α2_dn[min_counter] = y1*min_coupling_dn.α2_mean/2 + min_coupling_dn.α2_std * randn(rng)
                    α3_dn[min_counter] = y1*min_coupling_dn.α3_mean/2 + min_coupling_dn.α3_std * randn(rng)
                    α4_dn[min_counter] = y1*min_coupling_dn.α4_mean/2 + min_coupling_dn.α4_std * randn(rng)
                    tc[min_counter] = 1
                    # @show 0,ith,x1,y1,x2,y2,α_up[min_counter]
                elseif x1 == x2
                    α_up[min_counter]  = -x1*min_coupling_up.α_mean/2  + min_coupling_up.α_std  * randn(rng)
                    α2_up[min_counter] = -x1*min_coupling_up.α2_mean/2 + min_coupling_up.α2_std * randn(rng)
                    α3_up[min_counter] = -x1*min_coupling_up.α3_mean/2 + min_coupling_up.α3_std * randn(rng)
                    α4_up[min_counter] = -x1*min_coupling_up.α4_mean/2 + min_coupling_up.α4_std * randn(rng)
        
                    α_dn[min_counter]  = -x1*min_coupling_dn.α_mean/2  + min_coupling_dn.α_std  * randn(rng)
                    α2_dn[min_counter] = -x1*min_coupling_dn.α2_mean/2 + min_coupling_dn.α2_std * randn(rng)
                    α3_dn[min_counter] = -x1*min_coupling_dn.α3_mean/2 + min_coupling_dn.α3_std * randn(rng)
                    α4_dn[min_counter] = -x1*min_coupling_dn.α4_mean/2 + min_coupling_dn.α4_std * randn(rng)
                    tc[min_counter] = 1
                    # @show 1,ith,x1,y1,x2,y2,α_up[min_counter]
                else
                    error("Wrong bond")
                end


            end
            # @show PBCx
            if PBCx != 1
                for y in 1:Ly
                    n = Lx + (y-1)*Lx
                    tc[n]  = 0
                end
            end
            if PBCy != 1
                for x in 1:Lx
                    n = x + (Ly-1)*Lx
                    tc[n+Lx*Ly]  = 0
                end
            end
            # @show α_up

        end
        # @show min_neighbor_table,coupling_to_photon
        # # initialize min parameters
        min_parameters_up = MinParameters(
            nmin, Nmin, α_up, α2_up, α3_up, α4_up, tc, min_neighbor_table,coupling_to_hopping
            # , coupling_to_photon,
            # init_photon_to_coupling, final_photon_to_coupling,
            # hopping_to_couplings, coupling_to_hopping
        )
        min_parameters_dn = MinParameters(
            nmin, Nmin, α_dn, α2_dn, α3_dn, α4_dn, tc, min_neighbor_table,coupling_to_hopping
            # , coupling_to_photon,
            # init_photon_to_coupling, final_photon_to_coupling,
            # hopping_to_couplings, coupling_to_hopping
        )

    else

        # initialize null min parameters
        min_parameters_up = MinParameters(electron_photon_model)
        min_parameters_dn = MinParameters(electron_photon_model)
    end

    return min_parameters_up, min_parameters_dn
end

@doc raw"""
    MinParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

Initialize and return null (empty) instance of [`MinParameters`](@ref).
"""
function MinParameters(electron_photon_model::ElectronPhotonModel{T,E,D}) where {T,E,D}

    return MinParameters(0, 0, T[], T[], T[], T[], Matrix{Int}(undef,2,0), Matrix{Int}(undef,2,0), Vector{Int}[], Vector{Int}[], Vector{Int}[], Int[])
end

# Update the total hopping energy for each time-slice based on the Min interaction
# and the photon field configuration `x`, where `sgn = ±1` determines whether the Min
# contribution to the total hopping energy is either added or subtracted.
function update!(fermion_path_integral::FermionPathIntegral{T,E},
                 min_parameters::MinParameters{T},
                 x::Matrix{E}, sgn::Int) where {T,E}

    (; t, Lτ) = fermion_path_integral
    (; Nmin, α, α2, α3, α4, tc, coupling_to_hopping) = min_parameters
    # @show "in hre"
    # @show size(t),x[1]
    # if min coupling present
    # @show size(t),x[1],t,sgn
    # @show coupling_to_hopping,size(coupling_to_hopping)
    if Nmin > 0
        # iterate over imaginary time slice
        @fastmath @inbounds for l in 1:Lτ
            for h in coupling_to_hopping
                t[h,l] += tc[h]*sgn*exp(im*α[h]*x[l])
                # @show l,h,-sgn*exp(im*α[h]*x[l]),t[h,l]
            end
            # end
        end
    end
    # @show t[:,1]
    # @show size(t),x[1],t[1],sgn
    # dadasdaf
    # @show "updated t",t
    return nothing
end