using JLD2

path = "photon_minicoup_square_U0.00_w1.00_g10.00_mu0.00_Lx15_Ly15_BC00_b15.00_eqavg0_tdavg0_init3-1/local/"

# pID = 


# a = 0.0
for bin in 1:12  # 或者 range(1, 12)
for pID in 0:29
    file_path = path * "bin-$(bin)_pID-$(pID).jld2"
    data = load(file_path)
    # a += real(data["photon_kin_energy"][1])
    # println("Bin $bin pID $pID photon energy: ", data["photon_kin_energy"])
    print(real(data["photon_kin_energy"][1])," ")
end
# print(a/12,"\n")
print("\n")
end
