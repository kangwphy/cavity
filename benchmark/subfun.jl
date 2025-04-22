using LinearAlgebra

function mapto1D(x::Int64,y::Int64)
    return (y-1)*Lx+x
end
function current(G,G0t,i,ipa,j,jpa)
    a = 4 * (- G[ipa,i]*G[jpa,j] - G[i,ipa]*G[j,jpa] + G[ipa,i]*G[j,jpa] + G[i,ipa]*G[jpa,j]) + 
            2 * (G[ipa,j]*G0t[jpa,i] + G[i,jpa]*G0t[j,ipa] - G[ipa,jpa]*G0t[j,i] - G[i,j]*G0t[jpa,ipa])
    return a       
end

function getH(phi,Lx,Ly,PBCx,PBCy)
    N = Lx*Ly
    H = zeros(N,N) + 0im* zeros(N,N)
    t=1
    for x in 1:Lx
        for y in 1:Ly
            H[mapto1D(x,y),mapto1D(rem(x,Lx)+1,y)] = -t*exp(1im*phi*(y-1))
            H[mapto1D(x,y),mapto1D(x,rem(y,Ly)+1)] = -t
            if x==Lx && !PBCx
                H[mapto1D(x,y),mapto1D(rem(x,Lx)+1,y)] = 0
            end
            if y==Ly && !PBCy
                H[mapto1D(x,y),mapto1D(x,rem(y,Ly)+1)] = 0
            end
        end
    end
    H = H + transpose(conj(H))
    return H
end

function calculate_ldos(H::Matrix, site_index::Int; num_points::Int=1000, gamma::Float64=0.05)
    """
    计算局域态密度（LDOS）
    :param H: 哈密顿量矩阵
    :param site_index: 目标格点的一维索引
    :param num_points: 能量点的数量
    :param sigma: 高斯展宽
    :return: 能量数组和局域态密度数组
    """
    # 对角化哈密顿量
    eigenvalues, eigenvectors = eigen(H)
    eigenvalues = real(eigenvalues)
    # 能量范围
    emin, emax = minimum(eigenvalues), maximum(eigenvalues)
    energy = range(emin, stop=emax, length=num_points)
    
    # 计算 LDOS
    ldos = zeros(Float64, num_points)
    for n in 1:length(eigenvalues)
        psi_n = eigenvectors[:, n]  # 第 n 个本征矢
        weight = abs(psi_n[site_index])^2  # |ψ_n(r)|^2
        ldos .+= weight .* (gamma ./ (pi * ((energy .- eigenvalues[n]).^2 .+ gamma^2)))
    end
    
    return energy, ldos
end

function calFS(H,ω;η=0.05)

    A = (inv(-H+(ω + im*η)*I(size(H)[1])))
    Nx = Lx
    Ny = Ly
    Ak =zeros(ComplexF64,Nx,Ny)
    for kx in 1:Nx
        for ky in 1:Ny
            Kx = kx*2*pi/Nx
            Ky = ky*2*pi/Ny
            for x1 in 1:Nx
                for y1 in 1:Ny
                    r1 = mapto1D(x1,y1)
                    for x2 in 1:Nx
                        for y2 in 1:Ny
                            r2 = mapto1D(x2,y2)
                            Ak[kx,ky] += exp(im*Kx*(x1-x2)+im*Ky*(y1-y2))*A[r1,r2]/(N)
                        end
                    end
                end
            end
        end
    end
    return Ak
    end

function fftvector(kx,ky,Lx,Ly;pbc = false)
    V = zeros(ComplexF64,Lx*Ly)
    for x = 1:Lx
        for y = 1:Ly
            # V(num(x,y,Lx))= exp(i*(kx*(x)+ky*(y)))/sqrt(Lx*Ly);
            print(pbc)
            if pbc
                V[mapto1D(x,y)] = exp(im*(kx*x+ky*y))/(sqrt(Lx*Ly))
            else
                # print(pbc)
                V[mapto1D(x,y)] = sqrt(2/(Lx+1))*sqrt(2/(Ly+1))*sin(kx*x)*sin(ky*y)
            end
        end
    end
    return V
end
# function calG(H,w,eta)
#     Ens, evs = eigen(H)
#     # A = 1/((w+1i*eta)*I(size(Ens,1))-Ens)
#     A = Diagonal(1 ./((w+1im*eta).-Ens))

#     G = evs*(A.*evs');
#     return G
# end
