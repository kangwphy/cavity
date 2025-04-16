function mycontract_Gr0!(
    S::AbstractArray{C,D}, G::AbstractMatrix{T},
    r′::Bond{D}, α::Int,
    unit_cell::UnitCell{D,E}, lattice::Lattice{D},
    sgn=one(C)
) where {D, C<:Complex, T<:Number, E<:AbstractFloat}

    # get bond definition
    r₁ = r′.displacement
    b, a = r′.orbitals

    # orbitals per unit cell
    n = unit_cell.n

    # number of unit cells
    N  = lattice.N

    # dimensions of lattice
    L = size(S)

    # temporary vector
    tmp = lattice.lvec

    # Get a view into the Greens matrix for a and b orbtials
    Gab = @view G[a:n:end, b:n:end]

    # evaluate S(r) = S(r) + α/N sum_i G(a,i+r+r₁|b,i)
    i = reshape(1:N, L)
    αN⁻¹ = sgn * α/N
    @fastmath @inbounds for r in CartesianIndices(S)
        @. tmp = -(r.I-1+r₁)
        iprpr₁ = sa.circshift(i, tmp)
        for n in CartesianIndices(S)
            S[r] += αN⁻¹ * Gab[iprpr₁[n], i[n]]
        end
    end

    return nothing
end

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
    S0 = zeros(L...)
    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α

    i = reshape(1:N, L) # i
    @. tmp = -r₂
    ipr₂ = sa.circshift(i, tmp) # i + r₂
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements 
    @fastmath @inbounds for r in CartesianIndices(S0)
        @. tmp = -(r.I-1)
        ipr    = sa.circshift(i, tmp)
        @. tmp = -(r.I - 1 + r₄)
        iprpr₄ = sa.circshift(i, tmp) # i + r + r₄
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃

        # average over translation symmetry
        for n in CartesianIndices(S0)
            # @show r,n,idxs[idx1], idxs[idx2]
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r+r₄|b,i+r+r₃)⋅G₁(c,i+r₂|d,i+r₁)
            idxs = [iprpr₄[n], iprpr₃[n], ipr₂[n], ipr₁[n]]
            # if (idxs[idx1]-ipr[n])!=0 || (idxs[idx2]-i[n])!=0
            #     @show "noneq"
            # end
            # S[idxs[idx1], idxs[idx2]] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[iprpr₄[n],iprpr₃[n]] * G₁_cd[ipr₂[n],ipr₁[n]]
            S[ipr[n], i[n]] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[iprpr₄[n],iprpr₃[n]] * G₁_cd[ipr₂[n],ipr₁[n]]
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
    S0 = zeros(L...)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α

    i = reshape(1:N, L) # i
    @. tmp = -r₄
    ipr₄ = sa.circshift(i, tmp) # i + r₄
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S0)
        @. tmp = -(r.I - 1)
        ipr    = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃
        @. tmp = -(r.I - 1 + r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S0)
            # @show r,n,idxs[idx1], idxs[idx2]
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r₄|b,i+r+r₃)⋅G₁(c,i+r+r₂|d,i+r₁)
            idxs = [ipr₄[n], iprpr₃[n], iprpr₂[n], ipr₁[n]]
            # print(idxs[idx1]-ipr[n], idxs[idx2]-i[n])
            # if (idxs[idx1]-ipr[n])!=0 || (idxs[idx2]-i[n])!=0
            #     @show "noneq"
            # end

            # S[idxs[idx1], idxs[idx2]] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[ipr₄[n],iprpr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
            S[ipr[n], i[n]] += αN⁻¹ * η₂[ipr[n]] * η₁[i[n]] * G₂_ab[ipr₄[n],iprpr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
        end
    end

    return nothing
end








function mycontract_Grr_G00!(
    S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
    a::Int, b::Int, c::Int, d::Int,
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
    S0 = zeros(L...)
    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α

    i = reshape(1:N, L) # i
    @. tmp = -r₂
    ipr₂ = sa.circshift(i, tmp) # i + r₂
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements 
    @fastmath @inbounds for r in CartesianIndices(S0)
        @. tmp = -(r.I-1)
        ipr    = sa.circshift(i, tmp)
        @. tmp = -(r.I - 1 + r₄)
        iprpr₄ = sa.circshift(i, tmp) # i + r + r₄
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃

        # average over translation symmetry
        for n in CartesianIndices(S0)
            # @show r,n,idxs[idx1], idxs[idx2]
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r+r₄|b,i+r+r₃)⋅G₁(c,i+r₂|d,i+r₁)
            idxs = [iprpr₄[n], iprpr₃[n], ipr₂[n], ipr₁[n]]
            # if (idxs[idx1]-ipr[n])!=0 || (idxs[idx2]-i[n])!=0
            #     @show "noneq"
            # end
            # S[idxs[idx1], idxs[idx2]] += αN⁻¹ * G₂_ab[iprpr₄[n],iprpr₃[n]] * G₁_cd[ipr₂[n],ipr₁[n]]
            S[ipr[n], i[n]] += αN⁻¹ * G₂_ab[iprpr₄[n],iprpr₃[n]] * G₁_cd[ipr₂[n],ipr₁[n]]
        end
    end

    return nothing
end

function mycontract_G0r_Gr0!(
    S::AbstractArray{C,D}, G₂::AbstractMatrix{T}, G₁::AbstractMatrix{T},
    a::Int, b::Int, c::Int, d::Int,
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
    S0 = zeros(L...)

    # rename vectors
    tmp = lattice.lvec

    # view of Green's matrix for relevant orbitals
    G₂_ab = @view G₂[a:n:end,b:n:end]
    G₁_cd = @view G₁[c:n:end,d:n:end]

    # α/N
    αN⁻¹ = sgn * α

    i = reshape(1:N, L) # i
    @. tmp = -r₄
    ipr₄ = sa.circshift(i, tmp) # i + r₄
    @. tmp = -r₁
    ipr₁ = sa.circshift(i, tmp) # i + r₁
    # iterate over displacements
    @fastmath @inbounds for r in CartesianIndices(S0)
        @. tmp = -(r.I - 1)
        ipr    = sa.circshift(i, tmp) # i + r
        @. tmp = -(r.I - 1 + r₃)
        iprpr₃ = sa.circshift(i, tmp) # i + r + r₃
        @. tmp = -(r.I - 1 + r₂)
        iprpr₂ = sa.circshift(i, tmp) # i + r + r₂
        # average over translation symmetry
        for n in CartesianIndices(S0)
            # S(r) = S(r) + α/N sum_i η₂(i+r)⋅η₁(i)⋅G₂(a,i+r₄|b,i+r+r₃)⋅G₁(c,i+r+r₂|d,i+r₁)
            idxs = [ipr₄[n], iprpr₃[n], iprpr₂[n], ipr₁[n]]
            # if (idxs[idx1]-ipr[n])!=0 || (idxs[idx2]-i[n])!=0
            #     @show "noneq"
            # end
            # S[idxs[idx1], idxs[idx2]] += αN⁻¹ * G₂_ab[ipr₄[n],iprpr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
            S[ipr[n], i[n]] += αN⁻¹ * G₂_ab[ipr₄[n],iprpr₃[n]] * G₁_cd[iprpr₂[n],ipr₁[n]]
        end
    end

    return nothing
end
