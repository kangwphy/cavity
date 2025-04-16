##### this is to compute the obs with MC steps ####
using JLD2
using Glob
using Printf
#using Plots
function checkconverge(folder,pID,N)
    # pID = 0
    # folder = "./data/Lx4_Ly4_BC01_b4.00/photon_minicoup_square_U0.00_w1.00_g1.00_mu0.00_Lx4_Ly4_BC01_b4.00_eqavg0_tdavg0-1/"

    ending = @sprintf("*_pID-%d.jld2", pID)
    directory = joinpath(folder, "local")
    files = glob(ending, directory)
    Nfiles = length(files)
    # get_bin_intervals(folder, N_bins, 0)
    Etot = zeros(ComplexF64,Nfiles)
    Eel = zeros(ComplexF64,Nfiles)
    Ekin = zeros(ComplexF64,Nfiles)
    Epot = zeros(ComplexF64,Nfiles)
    for i in 1:Nfiles
        local_measurements = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jld2", directory, i, pID))
        Etot[i] = sum(local_measurements["hopping_energy"]) + (sum(local_measurements["photon_pot_energy"])+sum(local_measurements["photon_kin_energy"]))/(N)
        Eel[i] = sum(local_measurements["hopping_energy"])
        Epot[i] = sum(local_measurements["photon_pot_energy"])
        Ekin[i] = sum(local_measurements["photon_kin_energy"])
    end

    # JLD2.jldsave(@sprintf("%s/Etot.jld2", folder); Etot)
    open(joinpath(folder,"Etot_pID-$(pID).csv"), "w") do fout
        # _write_global_measurements(fout, global_measurements_avg, global_measurements_std)
        for i in 1:Nfiles
            @printf(fout,"%.8f\n", real(Etot[i]))
        end
    end

    open(joinpath(folder,"Ehop_pID-$(pID).csv"), "w") do fout
        # _write_global_measurements(fout, global_measurements_avg, global_measurements_std)
        for i in 1:Nfiles
            @printf(fout,"%.8f\n", real(Eel[i]))
        end
    end

    open(joinpath(folder,"Ekin_pID-$(pID).csv"), "w") do fout
        # _write_global_measurements(fout, global_measurements_avg, global_measurements_std)
        for i in 1:Nfiles
            @printf(fout,"%.8f\n", real(Ekin[i]))
        end
    end

    open(joinpath(folder,"Epot_pID-$(pID).csv"), "w") do fout
        # _write_global_measurements(fout, global_measurements_avg, global_measurements_std)
        for i in 1:Nfiles
            @printf(fout,"%.8f\n", real(Epot[i]))
        end
    end
end

## Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    ## Read in the command line arguments.
    sID = parse(Int, ARGS[1]) # simulation ID
    U = parse(Float64, ARGS[2])
    Ω = parse(Float64, ARGS[3])
    g = parse(Float64, ARGS[4])
    μ = parse(Float64, ARGS[5])
    β = parse(Float64, ARGS[6])
    Lx = parse(Int, ARGS[7])
    Ly = parse(Int, ARGS[8])
    PBCx = parse(Int, ARGS[9])
    PBCy = parse(Int, ARGS[10])
    N_burnin = parse(Int, ARGS[11])
    N_updates = parse(Int, ARGS[12])
    N_bins = parse(Int, ARGS[13])
    eqt_average = parse(Bool,ARGS[14])
    tdp_average = parse(Bool,ARGS[15])
    prefix = ARGS[16]
    ## Run the simulation.
    filepath = "./data/"*prefix
    dir = @sprintf "%s/Lx%d_Ly%d_BC%d%d_b%.2f" filepath Lx Ly PBCx PBCy β
    datafolder_prefix =  @sprintf "%s/photon_minicoup_square_U%.2f_w%.2f_g%.2f_mu%.2f_Lx%d_Ly%d_BC%d%d_b%.2f_eqavg%d_tdavg%d-%d" dir U Ω g μ Lx Ly PBCx PBCy β eqt_average tdp_average sID
    N = Lx*Ly
    checkconverge(datafolder_prefix,0,N)
end
