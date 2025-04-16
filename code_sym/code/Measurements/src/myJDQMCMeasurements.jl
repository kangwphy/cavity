module myJDQMCMeasurements

using LinearAlgebra
using Statistics
using FFTW
using StaticArrays
using LatticeUtilities
import ShiftedArrays as sa
import OffsetArrays as oa

# implement methods no exported by the package
include("myutilities.jl")
export mycontract_Grr_G00!, mycontract_G0r_Gr0!
include("utilities.jl")
export contract_G00!,contract_Gr0!,contract_Grr_G00!,contract_Gr0_Gr0!,contract_G0r_Gr0!

# calculate the fourier transform for correlation data
include("fourier_transform.jl")
export fourier_transform!

# calculate susceptibility from correlation function
include("susceptibility.jl")
export susceptibility, susceptibility!

# implement scalar measurements
include("scalar_measurements.jl")
export measure_N, measure_n, measure_Nsqrd, measure_double_occ

# implement Green's function measurement
include("correlation_measurements/greens.jl")
export greens!

# implement density-density correlation measurement
include("correlation_measurements/density_correlation.jl")
export density_correlation!

# implement Sx-Sx spin correlation function
include("correlation_measurements/spin_x_correlation.jl")
export spin_x_correlation!

# implement Sz-Sz spin correlation function
include("correlation_measurements/spin_z_correlation.jl")
export spin_z_correlation!

# implement pair correlation function
include("correlation_measurements/pair_correlation.jl")
export pair_correlation!

# implement bond correlation function
include("correlation_measurements/bond_correlation.jl")
export bond_correlation!

# implement current correlation function
include("correlation_measurements/current_correlation.jl")
export current_correlation!

# implements the jackknife algorithm
include("jackknife.jl")
export jackknife

end