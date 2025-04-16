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
# define all the parameters in the model given a specific finite lattice size
# include("ElectronPhoton/ElectronPhotonModel.jl")
# include("ElectronPhoton/ElectronPhotonParameters.jl")
# include("Measurements/initialize_measurements.jl")
# include("Measurements/make_measurements.jl")
# include("Measurements/write_measurements.jl")
# include("Measurements/process_measurements.jl")

# include("ElectronPhoton/EFAHMCUpdater.jl")
# # include("ElectronPhoton/HMCUpdater.jl")
# include("ElectronPhoton/reflection_update.jl")
# include("Measurements/initialize_measurements.jl")
include("mydataprocess/mydataprocess.jl")
# using
using .mydataprocess
include("mydataprocess/process_measurements.jl")

Nbin = 2
Nstart =3 
process_measurements("data/Lx4_Ly4_BC00_b4.00/photon_minicoup_square_U0.00_w1.00_g1.00_mu0.00_Lx4_Ly4_BC00_b4.00_eqavg0_tdavg0-1/", Nbin, Nstart, time_displaced=true)