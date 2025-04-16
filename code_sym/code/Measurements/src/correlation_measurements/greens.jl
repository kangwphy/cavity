@doc raw"""
    greens!(
        G::AbstractArray{C,D},
        a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
        G_τ0::AbstractMatrix{T},
        sgn=one(C)
    ) where {D, C<:Number, T<:Number}

Measure the unequal time Green's function averaged over translation symmetry
```math
G_{\sigma,\mathbf{r}}^{a,b}(\tau)=\frac{1}{N}\sum_{\mathbf{i}}G_{\sigma,\mathbf{i}+\mathbf{r},\mathbf{i}}^{a,b}(\tau,0)
=\frac{1}{N}\sum_{\mathbf{i}}\langle\hat{\mathcal{T}}\hat{a}_{\sigma,\mathbf{i}+\mathbf{r}}^{\phantom{\dagger}}(\tau)\hat{b}_{\sigma,\mathbf{i}}^{\dagger}(0)\rangle,
```
with the result being added to `G`.

# Fields

- `G::AbstractArray{C,D}`: Array the green's function ``G_{\sigma,\mathbf{r}}^{a,b}(\tau)`` is written to.
- `a::Int`: Index specifying an orbital species in the unit cell.
- `b::Int`: Index specifying an orbital species in the unit cell.
- `unit_cell::UnitCell{D}`: Defines unit cell.
- `lattice::Lattice{D}`: Specifies size of finite lattice.
- `G_τ0::AbstractMatrix{T}`: The matrix ``G(\tau,0).``
- `sgn=one(C)`: The sign of the weight appearing in a DQMC simulation.
"""
function greens!(
    G::AbstractArray{C,D},
    a::Int, b::Int, unit_cell::UnitCell{D}, lattice::Lattice{D},
    G_τ0::AbstractMatrix{T},
    sgn=one(C),average::Bool=true
) where {D, C<:Number, T<:Number}

    # construct the relevant bond definition
    z = @SVector zeros(Int, D)
    d = Bond((b,a), z)::Bond{D}
    if average
        # average green's function over translation symmetry
        contract_Gr0!(G, G_τ0, d, 1, unit_cell, lattice, sgn)
    else
        # no average green's function over translation symmetry
        # mycontract_Gr0!(G, G_τ0, d, 1, unit_cell, lattice, sgn)
        # G = G_τ0
        for i in 1:size(G,1)
            for j in 1:size(G,2)
                G[i,j] += sgn*G_τ0[i,j]
            end
        end
    end
    return nothing
end