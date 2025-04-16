####################################
## MEASURE Min INTERACTION ENERGY ##
####################################

@doc raw"""
    measure_min_energy(
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        Gup::Matrix{T}, Gdn::Matrix{T},
        min_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the return the Min interaction energy
```math
\epsilon_{\rm min} = \sum_\sigma \left\langle [\alpha \hat{X}     + \alpha_2 \hat{X}^2
                                               \alpha_3 \hat{X}^3 + \alpha_4 \hat{X}^4]
                                              (\hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.}) \right\rangle
```
for coupling definition specified by `min_id`.
"""
function measure_min_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                            Gup::Matrix{T}, Gdn::Matrix{T},
                            min_id::Int) where {T<:Number, E<:AbstractFloat}

    x = electron_phonon_parameters.x::Matrix{E}
    min_parameters_up = electron_phonon_parameters.min_parameters_up::MinParameters{T}
    ϵ_min_up = measure_min_energy(min_parameters_up, Gup, x, min_id)
    min_parameters_up = electron_phonon_parameters.min_parameters_dn::MinParameters{T}
    ϵ_min_dn = measure_min_energy(min_parameters_dn, Gdn, x, min_id)
    ϵ_min = ϵ_min_up + ϵ_min_dn

    return ϵ_min, ϵ_min_up, ϵ_min_dn
end

@doc raw"""
    measure_min_energy(
        min_parameters::MinParameters{T},
        G::Matrix{T}, x::Matrix{E}, min_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the return the Min interaction energy
```math
\epsilon_{\rm min} = \left\langle [\alpha \hat{X}     + \alpha_2 \hat{X}^2
                                   \alpha_3 \hat{X}^3 + \alpha_4 \hat{X}^4]
                        (\hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.}) \right\rangle
```
for coupling definition specified by `min_id`.
"""
function measure_min_energy(
    min_parameters::MinParameters{T},
    G::Matrix{T}, x::Matrix{E}, min_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; Nmin, α, α2, α3, α4, tc, neighbor_table) = min_parameters
    # length of imaginary time axis
    Lτ = size(x,2)

    # initialize min energy to zero
    ϵ_min = zero(T)

    if Nmin > 0
        # iterate over imaginary time slice
        # @fastmath @inbounds for l in 1:Lτ
            # iterate over min couplinges
            # for i in 1:Nmin
        #    i=1
            # for h in 1:size(t,1)
            # for i in neighbor_table[1,:]
            #     for j in neighbor_table[2,:]
        for m in size(neighbor_table,2)
            i = neighbor_table[1,m]
            j = neighbor_table[2,m]
            hij = -G[j,i]
            hji = -G[i,j]
            ϵ_min += -tc[m]*exp(im*α[m]*x[Lτ])*hij - tc[m]*exp(-im*α[m]*x[Lτ])*hji
        end
            # end
        # end
        # ϵ_min=0
    end
    return ϵ_min
end


###############################################################
## MEASURE THE FREQUENCY WITH WHICH THE HOPPING CHANGES SIGN ##
###############################################################

@doc raw"""
    measure_min_sgn_switch(
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        tight_binding_parameters::TightBindingParameters{T,E},
        min_id::Int;
        spin::Int = +1
    ) where {T<:Number, E<:AbstractFloat}

Calculate the fraction of the time the sign of the hopping is changed as a result of the
Min coupling associated with `min_id`.
"""
function measure_min_sgn_switch(
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    tight_binding_parameters::TightBindingParameters{T,E},
    min_id::Int;
    spin::Int = +1
) where {T<:Number, E<:AbstractFloat}

    x = electron_phonon_parameters.x::Matrix{E}
    if isone(spin)
        min_parameters = electron_phonon_parameters.min_parameters_up::MinParameters{T}
    else
        min_parameters = electron_phonon_parameters.min_parameters_dn::MinParameters{T}
    end
    sgn = measure_min_sgn_switch(min_parameters, tight_binding_parameters, x, min_id)

    return sgn
end

# function measure_min_sgn_switch(
#     min_parameters::MinParameters{T},
#     tight_binding_parameters::TightBindingParameters{T,E},
#     x::Matrix{E}, min_id::Int
# ) where {T<:Number, E<:AbstractFloat}

#     (; t) = tight_binding_parameters
#     (; nmin, Nmin, α, α2, α3, α4, coupling_to_phonon, coupling_to_hopping) = min_parameters

#     # length of imaginary time axis
#     Lτ = size(x,2)

#     # initialize frequency of hopping sign switch to zero
#     sgn = zero(E)

#     # number of unit cells in lattice
#     Nunitcell = Nmin ÷ nmin

#     # get relevant views into arrays corresponding to min coupling id
#     slice = (min_id-1)*Nunitcell+1:min_id*Nunitcell
#     α′  = @view  α[slice]
#     α2′ = @view α2[slice]
#     α3′ = @view α3[slice]
#     α4′ = @view α4[slice]
#     ctp = @view coupling_to_phonon[:,slice]
#     cth = @view coupling_to_hopping[slice]

#     # iterate over imaginary time slice
#     for l in axes(x, 2)
#         # iterate over unit cells
#         for u in axes(ctp, 2)
#             p  = ctp[1,u]
#             p′ = ctp[2,u]
#             Δx = x[p′,l] - x[p,l]
#             # get the min coupling associated with the current unit cell
#             c = slice[u]
#             # get hopping associated with min coupling
#             h = cth[u]
#             # calculate effective hopping
#             t′ = t[h] - (α′[u]*Δx + α2′[u]*Δx^2 + α3′[u]*Δx^3 + α4′[u]*Δx^4)
#             # check if sign of effective hopping is different from bare hopping
#             sgn += !(sign(t′) ≈ sign(t[h]))
#         end
#     end

#     # normalize measurement
#     sgn /= (Nunitcell * Lτ)

#     return sgn
# end