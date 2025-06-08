module MyMeasurements
using Printf
using SmoQyDQMC
using JDQMCFramework
using Random
using LatticeUtilities
# include("../ElectronPhoton/ElectronPhoton.jl")
# using .ElectronPhoton
using ..ElectronPhoton: HolsteinCoupling,SSHCoupling,MinCoupling,PhotonParameters,MinParameters,ElectronPhotonModel,ElectronPhotonParameters,PhotonMode,PhotonDispersion



# measurements associated with bare photon modes
include("../ElectronPhoton/photon_measurements.jl")
export measure_photon_kinetic_energy, measure_photon_potential_energy, measure_photon_position_moment

# # measurements for holstein interaction
# include("../ElectronPhoton/holstein_measurements.jl")
# export measure_holstein_energy

# # measurements for ssh interaction
# include("../ElectronPhoton/ssh_measurements.jl")
# export measure_ssh_energy

# # measurements for photon dispersion
# include("../ElectronPhoton/dispersion_measurements.jl")
# export measure_dispersion_energy

# defines dictionaries as global variables that contain the names of all
# local measurements and correlation measurements that can be made, and the
# type ID type they are reported in terms of
include("measurement_names.jl")
export GLOBAL_MEASUREMENTS
export LOCAL_MEASUREMENTS
export CORRELATION_FUNCTIONS


# Define CorrelationContainer struct to store correlation measurements in.
include("CorrelationContainer.jl")

# Define CompositeCorrelationContainer struct to store composite correlation measurements
include("CompositeCorrelationContainer.jl")

# initialize measurement container
include("initialize_measurements.jl")
export initialize_measurement_container
export initialize_measurements!
export initialize_correlation_measurement!, initialize_correlation_measurements!
export initialize_composite_correlation_measurement!
export initialize_measurement_directories

# make measurements
include("make_measurements.jl")
export make_equal_measurements!,make_measurements!

# write measurements to file.
# additionally, the two following things are done here:
# 1. fourier transform position space correlation to momentum space
# 2. perform integration over imaginary time of correlation function to calculate susceptibilies
# include("write_measurements.jl")
# export write_measurements!

end
