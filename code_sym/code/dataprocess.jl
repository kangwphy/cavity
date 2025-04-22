using LinearAlgebra
using Random
using Printf
using JLD2
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
using LatticeUtilities
using MPI
###########################
## ELECTRON-photon MODEL ##
###########################
# Define electron-photon model agnostic to lattice size

# Define various electron-photon parameter i.e. given a electron-photon model,
# # define all the parameters in the model given a specific finite lattice size

# include("mydataprocess/mydataprocess.jl")
# using .mydataprocess
include("mydataprocess/process_measurements.jl")


## Define top-level function for running DQMC simulation
function dataprocess(datafolder, N_bins, N_start = 1)
    start_time = time()
    process_measurements(datafolder, N_bins, time_displaced=true, N_start = N_start)
    elapsed_time = time()-start_time
    @show elapsed_time

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
    eqt_average = parse(Bool,ARGS[11])
    tdp_average = parse(Bool,ARGS[12])
    prefix = ARGS[13]
    init = parse(Int, ARGS[16])
    ## Run the simulation.
    filepath = "./data/"*prefix
    dir = @sprintf "%s/Lx%d_Ly%d_BC%d%d_b%.2f" filepath Lx Ly PBCx PBCy β

    datafolder_prefix =  @sprintf "%s/photon_minicoup_square_U%.2f_w%.2f_g%.2f_mu%.2f_Lx%d_Ly%d_BC%d%d_b%.2f_eqavg%d_tdavg%d_init%d-%d" dir U Ω g μ Lx Ly PBCx PBCy β eqt_average tdp_average init sID
    # datafolder_prefix = "data/Lx4_Ly4_BC00_b4.00/photon_minicoup_square_U0.00_w1.00_g1.00_mu0.00_Lx4_Ly4_BC00_b4.00_eqavg0_tdavg0-1"
    N_start = parse(Int, ARGS[14])
    N_bins = parse(Int, ARGS[15])
    @show Lx,Ly,N_start,N_bins
    dataprocess(datafolder_prefix, N_bins, N_start)
end
