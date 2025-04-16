@doc raw"""
    density_correlation!(
        DD::AbstractArray{C,D},
        a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
        Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
        Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
        Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
        Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
        sgn::T=one(T)
    ) where {D, C<:Number, T<:Number}

Calculate the unequal-time density-density (charge) correlation function
```math
\begin{align*}
\mathcal{D}_{\mathbf{r}}^{a,b}(\tau)
    & = \frac{1}{N}\sum_{\mathbf{i}} \mathcal{D}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\tau,0)\\
    & = \frac{1}{N}\sum_{\mathbf{i}} \langle \hat{n}_{a,\mathbf{i} + \mathbf{r}}(\tau)\hat{n}_{b,\mathbf{i}}(0) \rangle,
\end{align*}
```
where ``\hat{n}_{b,\mathbf{i}} = (\hat{n}_{\uparrow, b, \mathbf{i}} + \hat{n}_{\downarrow, b, \mathbf{i}})``
and ``\hat{n}_{\sigma, b,\mathbf{i}} = \hat{b}^\dagger_{\sigma, \mathbf{i}} \hat{b}_{\sigma, \mathbf{i}}``
is the number operator for an electron with spin ``\sigma`` on orbital ``b`` in unit cell ``\mathbf{i}``,
with the result being added to the array `DD`.

# Fields

- `DD::AbstractArray{C,D}`: Array the density correlation function ``\mathcal{D}_{\mathbf{r}}^{a,b}(\tau)`` is added to.
- `a::Int`: Index specifying an orbital species in the unit cell.
- `b::Int`: Index specifying an orbital species in the unit cell.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `Gup_τ0::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(\tau,0).``
- `Gup_0τ::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(0,\tau).``
- `Gup_ττ::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(\tau,\tau).``
- `Gup_00::AbstractMatrix{T}`: The matrix ``G_{\uparrow}(0,0).``
- `Gdn_τ0::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(\tau,0).``
- `Gdn_0τ::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(0,\tau).``
- `Gdn_ττ::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(\tau,\tau).``
- `Gdn_00::AbstractMatrix{T}`: The matrix ``G_{\downarrow}(0,0).``
- `sgn::T=one(T)`: The sign of the weight appearing in a DQMC simulation.
"""
function density_correlation!(
    DD::AbstractArray{C,D},
    a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
    Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gup_ττ::AbstractMatrix{T}, Gup_00::AbstractMatrix{T},
    Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    Gdn_ττ::AbstractMatrix{T}, Gdn_00::AbstractMatrix{T},
    sgn::T=one(T),
    translation::Bool=true
) where {D, C<:Number, T<:Number}
    
    # up-up density-density correlation
    density_correlation!(DD, a, b, unit_cell, lattice, Gup_τ0, Gup_0τ, Gup_ττ, Gup_00, +1, +1, sgn, translation)

    # dn-dn density-density correlation
    density_correlation!(DD, a, b, unit_cell, lattice, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn_00, -1, -1, sgn, translation)

    # up-dn density-density correlation
    density_correlation!(DD, a, b, unit_cell, lattice, Gup_τ0, Gup_0τ, Gup_ττ, Gdn_00, +1, -1, sgn, translation)

    # dn-up density-density correlation
    density_correlation!(DD, a, b, unit_cell, lattice, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup_00, -1, +1, sgn, translation)
    
    return nothing
end

@doc raw"""
    density_correlation!(
        DD::AbstractArray{C,D},
        a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
        Gσ_τ0::AbstractMatrix{T}, Gσ_0τ::AbstractMatrix{T},
        Gσ_ττ::AbstractMatrix{T}, Gσ′_00::AbstractMatrix{T},
        σ::Int, σ′::Int, sgn::T=one(T)
    ) where {D, C<:Number, T<:Number}

Calculate the spin-resolved unequal-time density correlation function
```math
\begin{align*}
    \mathcal{D}_{\mathbf{r}}^{(a,\sigma),(b,\sigma')}(\tau)
            = \frac{1}{N}\sum_{\mathbf{i}} & \mathcal{D}_{\mathbf{i}+\mathbf{r},\mathbf{i}}^{(a,\sigma),(b,\sigma')}(\tau,0) \\
            = \frac{1}{N}\sum_{\mathbf{i}} & \langle\hat{n}_{\sigma,a,\mathbf{i}+\mathbf{r}}(\tau)\hat{n}_{\sigma',b,\mathbf{r}}(0)\rangle,
\end{align*}
```
where ``\hat{n}_{\sigma', b,\mathbf{i}} = \hat{b}^\dagger_{\sigma', \mathbf{i}} \hat{b}_{\sigma, \mathbf{i}}``
is the number operator for an electron with spin ``\sigma'`` on orbital ``b`` in unit cell ``\mathbf{i}``,
with the result being added to the array `DD`.

# Fields

- `DD::AbstractArray{C,D}`: The array the spin-resolved density correlation ``\mathcal{D}_{\mathbf{r}}^{(a,\sigma),(b,\sigma')}(\tau)`` is added to.
- `a::Int`: Index specifying an orbital species in the unit cell.
- `b::Int`: Index specifying an orbital species in the unit cell.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `Gσ_τ0::AbstractMatrix{T}`: The matrix ``G_{\sigma}(\tau,0).``
- `Gσ_0τ::AbstractMatrix{T}`: The matrix ``G_{\sigma}(0,\tau).``
- `Gσ_ττ::AbstractMatrix{T}`: The matrix ``G_{\sigma}(\tau,\tau).``
- `Gσ′_00::AbstractMatrix{T}`: The matrix ``G_{\sigma'}(0,0).``
- `σ::Int`: The electron spin appearing in the ``\langle \hat{n} \rangle_{\sigma,a,\mathbf{i}+\mathbf{r}}`` density operator.
- `σ′::Int`: The electron spin appearing in the ``\langle \hat{n} \rangle_{\sigma',b,\mathbf{i}}`` density operator.
- `sgn::T=one(T)`: The sign of the weight appearing in a DQMC simulation.
"""
function density_correlation!(
    DD::AbstractArray{C,D},
    a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
    Gσ_τ0::AbstractMatrix{T}, Gσ_0τ::AbstractMatrix{T},
    Gσ_ττ::AbstractMatrix{T}, Gσ′_00::AbstractMatrix{T},
    σ::Int, σ′::Int, sgn::T=one(T),
    translation::Bool=true
) where {D, C<:Number, T<:Number}

    # define zero unit cell displacement bonds for all combos of a and b orbitals
    z = @SVector zeros(Int,D)
    b_aa = Bond((a,a), z)::Bond{D} # displacement r_a - r_a = 0
    b_bb = Bond((b,b), z)::Bond{D} # displacement r_b - r_b = 
    b_ab = Bond((b,a), z)::Bond{D} # displacement r_a - r_b
    b_ba = Bond((a,b), z)::Bond{D} # displacement r_b - r_a
    
    if translation
        # DD(τ,r) = DD(τ,r) + 1
        @. DD = DD + 1 * sgn 
        # DD(τ,r) = DD(τ,r) - 1/N sum_i Gσ(a,i+r,τ|a,i+r,τ) = DD(τ,r) - 1/N sum_i Gσ(a,i,τ|a,i,τ)
        contract_G00!(DD, Gσ_ττ, a, a, -1, unit_cell, lattice, sgn)

        # DD(τ,r) = DD(τ,r) - 1/N sum_i Gσ'(b,i,0|b,i,0)
        contract_G00!(DD, Gσ′_00, b, b, -1, unit_cell, lattice, sgn)

        # DD(τ,r) = DD(τ,r) + 1/N sum_i Gσ(a,i+r,τ|a,i+r,τ)⋅Gσ'(b,i,0|b,i,0)
        contract_Grr_G00!(DD, Gσ_ττ, Gσ′_00, b_aa, b_bb, 1, unit_cell, lattice, sgn)

        # if spins of two density operators are the same i.e. δ(σ,σ′)
        if σ == σ′

            # DD(τ,r) = DD(τ,r) - 1/N sum_i δ(σ,σ′)⋅Gσ(b,i,0|a,i+r,τ)⋅Gσ(a,i+r,τ|b,i,0)
            contract_G0r_Gr0!(DD, Gσ_0τ, Gσ_τ0, b_ba, b_ab, -1, unit_cell, lattice, sgn)
        end
    else

        ### now the size is N*N 
        for a in 1:size(DD,1)
            for b in 1:size(DD,2)
                DD[a,b] = 1 - Gσ_ττ[a,a] - Gσ′_00[b,b] + Gσ_ττ[a,a]*Gσ′_00[b,b]
                if σ == σ′
                    DD[a,b] -= Gσ_0τ[b,a]*Gσ_τ0[a,b]
                end
            end
        end

    end
    return nothing
end