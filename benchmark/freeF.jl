using MeshGrid
function main()
Lx = 6
Ly = 6
N = Lx*Ly
x = range(1, stop=Lx, length=Lx)*pi/(Lx+1)
y = range(1, stop=Ly, length=Ly)*pi/(Ly+1)
# y = range(1, stop=Ly, length=Ly)*pi/(Ly+1)


X, Y = meshgrid(x, y)
E = -2*(cos.(X)+cos.(Y))
@show E
idxs = findall(x -> x < 0, E)
E0 = sum(E[idxs])
# @show E0

beta = 6
A = 0
B = 0
@show size(E)
# for a in E
#     @show a
# end
# sum.(exp.(-beta*E).*E) / sum.(exp.(-beta*E))
# for i
E = reshape(E,(N,1))
# @show sum(exp.(-beta*E).*E)/sum(exp.(-beta*E))
ET = sum((ones(N,1)-inv.(ones(N,1) .+exp.(-beta*E))) .* E)
@show E0/(Lx*Ly),ET/(Lx*Ly)
# exp(-beta*E[i]) /exp(-beta*E[i]) 

# for i in idxs
#     A += E[i]*exp(-beta*E[i]) 
#     B += exp(-beta*E[i]) 
# end
# @show A/B
end
main()









# @show sum(E[idxs] .* exp.(-beta*E[idxs]))/sum(exp.(-beta*E[idxs]))
# sum( exp.(-10*E[findall(x -> x < 0, E)]))
# @show sum( exp.(-10*E[findall(x -> x < 0, E)]))
# @show sum(E[findall(x -> x < 0, E)] .* exp.(-10*E[findall(x -> x < 0, E)]))/sum( exp.(-10*E[findall(x -> x < 0, E)]))
# @show x,y
# X = [repeat([x], outer=(Ly,1)) for x in x]
# Y = [repeat(y, inner=1, outer=Lx) for _ in y]

# Flatten the arrays to get the final meshgrid
# X = vec(reshape(X, 10000, 1))
# Y = vec(reshape(Y, 10000, 1))

# @show size(X),size(Y)