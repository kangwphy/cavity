module ElectronPhoton
using SmoQyDQMC
using Printf
using Random
# Define electron-photon model agnostic to lattice size
include("ElectronPhotonModel.jl")
export ElectronPhotonModel, PhotonMode, HolsteinCoupling, SSHCoupling, MinCoupling, PhotonDispersion
export add_photon_mode!, add_holstein_coupling!, add_ssh_coupling!,add_min_coupling!, add_photon_dispersion!

# Define various electron-photon parameter i.e. given a electron-photon model,
# define all the parameters in the model given a specific finite lattice size
include("PhotonParameters.jl")
include("HolsteinParameters.jl")
include("SSHParameters.jl")
include("MinParameters.jl")
include("DispersionParameters.jl")
include("ElectronPhotonParameters.jl")
export ElectronPhotonParameters, update!

# methods for evaluating the bosonic action Sb and its derivative with respect to photon fields ∂Sb/∂x
include("bosonic_action.jl")

# methods for evaluating the derivative of the fermionic action with respect to photon fields ∂Sf/∂x
include("fermionic_action_derivative.jl")

# implements fourier mass matrix to use in HMC/Langevin updates, which gives us fourier acceleration
include("FourierMassMatrix.jl")

# low-level (private) hybrid/hamiltonian monte carlo (HMC) update method
include("hmc_update.jl")

# defines HMC udpater struct and public API for perform HMC updates to photon fields
include("HMCUpdater.jl")
export HMCUpdater, hmc_update!

# implement exact fourier acceleration integration of
# equation of motion
include("ExactFourierAccelerator.jl")

# defines Exact Fourier Acceleration HMC update method
include("EFAHMCUpdater.jl")
export EFAHMCUpdater

# impelment reflection, swap and radial updates for photon fields
include("reflection_update.jl")
# include("swap_update.jl")
# include("radial_update.jl")
export reflection_update!
# , swap_update!, radial_update!

end