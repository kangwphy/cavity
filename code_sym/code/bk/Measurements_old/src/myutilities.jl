

function mycontract_Grr_G00!(
    S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
    η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
    r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
    α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
    sgn::T=one(T),idx1::Int=1,idx2::Int=1
) where {D, C<:Complex, T<:Number, E<:AbstractFloat}
    # @show a,b,c,d,r₁,r₂,r₃,r₄,size(S)
    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    # L = size(S)
    L = Tuple(lattice.L)
    # @show Tuple(L)
    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₂
    ipr₂ = sa.circshift(i, tmp) # i + r₂
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1)
        ipr    = sa.circshift(i, tmp)
        @. tmp = -(r.I - 1 + r₄)
        iprpr₄ = sa.circshift(i, tmp) # i + r + r₄
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃

        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r+r₄|b,i+r+r₃)⋅G₁(c,i+r₂|d,i+r₁)
            idxs = [iprpr₄[n], iprpr₃[n], ipr₂[n], ipr₁[n]]
            S[idxs[idx1], idxs[idx2]] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[iprpr₄[n],iprpr₃[n]] * G₁_cd[ipr₂[n],ipr₁[n]]
        end
    end

    return nothing
end

function mycontract_G0r_Gr0!(
    S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
    η₂::AbstractArray{T}, η₁::AbstractArray{T}, a::Int, b::Int, c::Int, d::Int,
    r₄::AbstractVector{Int}, r₃::AbstractVector{Int}, r₂::AbstractVector{Int}, r₁::AbstractVector{Int},
    α::Int, unit_cell::UnitCell{D,E}, lattice::Lattice{D},
    sgn::T=one(T),idx1::Int=1,idx2::Int=1
) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # number of unit cells
    N = lattice.N

    # number of orbitals per unit cell
    n = unit_cell.n

    # dimensions of lattice
    # L = size(S)
    L = Tuple(lattice.L)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α/N

    i = reshape(1:N, L) # i
    @. tmp = -r₄
    ipr₄ = sa.circshift(i, tmp) # i + r₄
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I - 1)
        ipr    = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃
        @. tmp = -(r.I - 1 + r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r₄|b,i+r+r₃)⋅G₁(c,i+r+r₂|d,i+r₁)
            idxs = [ipr₄[n], iprpr₃[n], iprpr₂[n], ipr₁[n]]
            S[idxs[idx1], idxs[idx2]] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[ipr₄[n],iprpr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
        end
    end

    return nothing
end
