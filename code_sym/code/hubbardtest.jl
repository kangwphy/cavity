using LinearAlgebra
using Random
using Printf
using JLD2
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
# import SmoQyDQMC.JDQMCMeasurements as dqmcm
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
# include("Measurements/initialize_measurements.jl")
# include("Measurements/make_measurements.jl")
# include("Measurements/write_measurements.jl")
# include("Measurements/process_measurements.jl")

# include("ElectronPhoton/EFAHMCUpdater.jl")
# # include("ElectronPhoton/HMCUpdater.jl")
# include("ElectronPhoton/reflection_update.jl")

# include("mydataprocess/mydataprocess.jl")
# using .mydataprocess
# include("mydataprocess/process_measurements.jl")
include("checkconverge.jl")

include("mydataprocess/process_measurements.jl")
# include("Measurements/process_correlation_measurements.jl")
# initialize MPI
MPI.Init()
## Define top-level function for running DQMC simulation
function run_photon_minicoup_square_simulation(sID, U, Ω, g, μ, β, Lx, Ly, PBCx, PBCy, N_burnin, N_updates, N_bins, eqt_average, tdp_average; filepath = ".",Nt = 10, at = 1, init=0)
    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD
    start_time = time()
    ## Construct the foldername the data will be written to.
    if !isdir(filepath)
        mkdir(filepath)
    end
    dir = @sprintf "%s/Lx%d_Ly%d_BC%d%d_b%.2f" filepath Lx Ly PBCx PBCy β
    if !isdir(dir)
        mkdir(dir)
    end
    
    # folder = @sprintf "data/photon_minicoup_square_U%.2f_w%.2f_g%.2f_mu%.2f_L%d_b%.2f" U Ω g μ L β
    datafolder_prefix =  @sprintf "%s/photon_minicoup_square_U%.2f_w%.2f_g%.2f_mu%.2f_Lx%d_Ly%d_BC%d%d_b%.2f_eqavg%d_tdavg%d_init%d" dir U Ω g μ Lx Ly PBCx PBCy β eqt_average tdp_average init
    # datafolder_prefix = folder*datafolder_prefix

    # Get the MPI comm rank, which fixes the process ID (pID).
    pID = MPI.Comm_rank(comm)
    ## Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = ".",                     
        datafolder_prefix = datafolder_prefix,
        sID = sID,
        pID = pID
    )
    #@show simulation_info.resuming
    # simulation_info.resuming = true
    
    # Synchronize all the MPI processes.
    # Here we need to make sure the data folder is initialized before letting
    # all the various processes move beyond this point.
    # MPI.Barrier(comm)


    # Define checkpoint filename.
    # We implement three checkpoint files, an old, current and new one,
    # that get cycled through to ensure a checkpoint file always exists in the off
    # chance that the simulation is killed while a checkpoint is getting written to file.
    # Additionally, each simulation that is running in parallel with MPI will have their own
    # checkpoints written to file.
    datafolder = simulation_info.datafolder
    sID        = simulation_info.sID
    pID        = simulation_info.pID
    checkpoint_name_old          = @sprintf "checkpoint_sID%d_pID%d_old.jld2" sID pID
    checkpoint_filename_old      = joinpath(datafolder, checkpoint_name_old)
    checkpoint_name_current      = @sprintf "checkpoint_sID%d_pID%d_current.jld2" sID pID
    checkpoint_filename_current  = joinpath(datafolder, checkpoint_name_current)
    checkpoint_name_new          = @sprintf "checkpoint_sID%d_pID%d_new.jld2" sID pID
    checkpoint_filename_new      = joinpath(datafolder, checkpoint_name_new)

   # @show checkpoint_filename_new

    ######################################################
    ### DEFINE SOME RELEVANT DQMC SIMULATION PARAMETERS ##
    ######################################################

    ## Set the discretization in imaginary time for the DQMC simulation.
    Δτ = 0.1

    ## Calculate the length of the imaginary time axis, Lτ = β/Δτ.
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    ## This flag indicates whether or not to use the checkboard approximation to
    ## represent the exponentiated hopping matrix exp(-Δτ⋅K)
    checkerboard = false

    ## Whether the propagator matrices should be represented using the
    ## symmetric form B = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)
    ## or the asymetric form B = exp(-Δτ⋅V)⋅exp(-Δτ⋅K)
    symmetric = false

    ## Set the initial period in imaginary time slices with which the Green's function matrices
    ## will be recomputed using a numerically stable procedure.
    n_stab = 10

    ## Specify the maximum allowed error in any element of the Green's function matrix that is
    ## corrected by performing numerical stabiliziation.
    δG_max = 1e-6

    δG = 0.0
    δθ = 0.0


    #######################
    ### MC Parameters ##
    #######################
    ## Calculate the bins size.
    bin_size = div(N_updates, N_bins)
    N_bins_burnin = div(N_burnin, bin_size)



    #######################
    ### DEFINE THE MODEL ##
    #######################

    ## Initialize an instance of the type UnitCell.
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0, 0.0],
                                            [0.0, 1.0]],
                            basis_vecs   = [[0.0, 0.0]])

    ## Initialize an instance of the type Lattice.
    lattice = lu.Lattice(
        L = [Lx, Ly],
        periodic = [true, true]
    )

    ## Initialize an instance of the ModelGeometry type.
    model_geometry = ModelGeometry(unit_cell, lattice)

    ## Get the number of orbitals in the lattice.
    N = lu.nsites(unit_cell, lattice)

    ## Define the nearest-neighbor bond in the +x direction.
    bond_px = lu.Bond(orbitals = (1,1), displacement = [1,0])

    ## Add nearest-neighbor bond in the +x direction.
    bond_px_id = add_bond!(model_geometry, bond_px)

    ## Define the nearest-neighbor bond in the +y direction.
    bond_py = lu.Bond(orbitals = (1,1), displacement = [0,1])

    ## Add the nearest-neighbor bond in the +y direction.
    bond_py_id = add_bond!(model_geometry, bond_py)

#md     ## Here we define bonds to points in the negative x and y directions respectively.
#md     ## We do this in order to be able to measure all the pairing channels we need
#md     ## in order to reconstruct the extended s-wave and d-wave pair susceptibilities.

    ## Define the nearest-neighbor bond in the -x direction.
    bond_nx = lu.Bond(orbitals = (1,1), displacement = [-1,0])

    ## Add nearest-neighbor bond in the -x direction.
    bond_nx_id = add_bond!(model_geometry, bond_nx)

    ## Define the nearest-neighbor bond in the -y direction.
    bond_ny = lu.Bond(orbitals = (1,1), displacement = [0,-1])

    ## Add the nearest-neighbor bond in the -y direction.
    bond_ny_id = add_bond!(model_geometry, bond_ny)

    # bond_3d_2px_px_id = add_bond!(model_geometry, bond_3d_2px_px)
    ## Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0+0.0im
    # @show model_geometry

    ## Define the tight-binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py], # defines hopping
        t_mean = [0.0+0.0im, 0.0+0.0im],            # defines corresponding hopping amplitude
        μ = μ,                      # set chemical potential
        ϵ_mean = [0.]               # set the (mean) on-site energy
    )

    # @show tight_binding_model.t_bonds, tight_binding_model.t_bond_ids
    # Initialize the Hubbard interaction in the model.

    hubbard_model = HubbardModel(
    shifted = false,
    U_orbital = [1],
    U_mean = [U],
    )

    

    # ## Initialize a null electron-photon model.
    # electron_photon_model = ElectronPhotonModel(
    #     model_geometry = model_geometry,
    #     tight_binding_model = tight_binding_model
    #     )

    #######################################################
    ### BRANCHING BEHAVIOR BASED ON WHETHER STARTING NEW ##
    ### SIMULAIOTN OR RESUMING PREVIOUS SIMULATION.      ##
    #######################################################

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # If starting a new simulation.
    if !simulation_info.resuming
        ## Initialize the directory the data will be written to.
        initialize_datafolder(simulation_info)

        ## Initialize a random number generator that will be used throughout the simulation.
        seed = abs(rand(Int))
        rng = Xoshiro(seed)

        ## Number of fermionic time-steps in HMC update.
        # Nt = 10

        ## Fermionic time-step used in HMC update.
        # Δt = π/(2*Ω)/Nt
        Δt = at*π/(2*Ω)/Nt
 
        ## Initialize a dictionary to store additional information about the simulation.
        additional_info = Dict(
            "dG_max" => δG_max,
            "N_burnin" => N_burnin,
            "N_updates" => N_updates,
            "N_bins" => N_bins,
            "bin_size" => bin_size,
            "local_acceptance_rate" => 0.0,
            "hmc_acceptance_rate" => 0.0,
            "reflection_acceptance_rate" => 0.0,
            "n_stab_init" => n_stab,
            "symmetric" => symmetric,
            "checkerboard" => checkerboard,
            "seed" => seed,
            "Nt" => Nt,
            "dt" => Δt,
        )

        ## Define a dispersionless electron-photon mode to live on each site in the lattice.
        # photon = PhotonMode(orbital = 1, Ω_mean = Ω)

        ## Add the photon mode definition to the electron-photon model.
        # photon_id = add_photon_mode!(
        #     electron_photon_model = electron_photon_model,
        #     photon_mode = photon
        # )

        α = sqrt(2)*g/sqrt(Lx*Ly)

        if iszero(simulation_info.pID)
            @show dir
            @show Nt,Δt,Nt*Δt,α
        end


        # ## Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        if PBCx!=1
            for y in 1:Ly
                n = Lx+(y-1)*Lx
                tight_binding_parameters.t[n]=0
            end
        end
        if PBCy!=1
            for x in 1:Lx
                n = x + (Ly-1)*Lx
                tight_binding_parameters.t[N+n]=0
            end
        end

        # ###### minimal couppling parameters in Ele-photon models ###### 
        # min_coupling = MinCoupling(
        # model_geometry = model_geometry,
        # tight_binding_model = tight_binding_model,
        # photon_modes = (photon_id, photon_id),
        # # bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
        # bonds = [bond_px,bond_py],
        # α_mean = α
        # )
        
        # # add_min_coupling!
        # min_coupling_id = add_min_coupling!(
        #     electron_photon_model = electron_photon_model,
        #     tight_binding_model = tight_binding_model,
        #     min_coupling = min_coupling
        #     # model_geometry = model_geometry
        # )
        
        # # ## Initialize electron-photon parameters.
        # electron_photon_parameters = ElectronPhotonParameters(;
        #     β = β, Δτ = Δτ,
        #     electron_photon_model = electron_photon_model,
        #     tight_binding_parameters = tight_binding_parameters,
        #     model_geometry = model_geometry,
        #     PBCx = PBCx, PBCy = PBCy, init = init,
        #     rng = rng
        #     )

        # Initialize Hubbard interaction parameters.
        hubbard_parameters = HubbardParameters(
            model_geometry = model_geometry,
            hubbard_model = hubbard_model,
            rng = rng
        )

        # Apply Ising Hubbard-Stranonvich (HS) transformation, and initialize
        # corresponding HS fields that will be sampled in DQMC simulation.
        hubbard_ising_parameters = HubbardIsingHSParameters(
            β = β, Δτ = Δτ,
            hubbard_parameters = hubbard_parameters,
            rng = rng
        )
        
        ## Write the model summary to file.
        # if U != 0
         
        # end
        if iszero(simulation_info.pID)
            # @show electron_photon_parameters
            @show tight_binding_parameters
            if U!=0
                @show hubbard_parameters
                @show hubbard_ising_parameters
            end
        end
        if U!=0
            model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            # interactions = ( hubbard_model, electron_photon_model,)
            interactions = ( hubbard_model, )
        )
        else
            model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = ( electron_photon_model,)
        )
        end

        # #########################################
        # ### INITIALIZE FINITE MODEL PARAMETERS ##
        # #########################################

        ## Calculating weight
        ## Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)
        # @show measurement_container
        ## Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        ## Initialize the Hubbard interaction related measurements.
        if U != 0
            initialize_measurements!(measurement_container, hubbard_model)
        end
        ## Initialize the electron-photon interaction related measurements.
        # initialize_measurements!(measurement_container, electron_photon_model)
        
        ## Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens",
            time_displaced = true,
            # tdp_average = tdp_average,
	    pairs = [(1, 1)]
        )

        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "holegreens",
        #     time_displaced = true,
        #     # tdp_average = tdp_average,
        #     pairs = [(1, 1)]
        # )

        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "greens_onsite",
        #     time_displaced = true,
        #     # tdp_average = tdp_average,
        #     pairs = [(1, 1)]
        # )

        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "holegreens_onsite",
        #     time_displaced = true,
        #     # tdp_average = tdp_average,
        #     pairs = [(1, 1)]
        # )


        ## measure equal-times green's function for all τ
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens_tautau",
            time_displaced = false,
            integrated = true,
            pairs = [(1, 1)]
        )

        # # Initialize the photon Green's function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "photon_greens",
        #     time_displaced = true,
        #     pairs = [(1, 1)]
        # )

        ## Initialize density correlation function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "density",
        #     time_displaced = false,
        #     integrated = true,
        #     eqt_average = eqt_average,
        #     tdp_average = tdp_average,
        #     pairs = [(1, 1)]
        # )
        # # Initialize the spin-z correlation function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "spin_z",
        #     time_displaced = false,
        #     integrated = true,
        #     pairs = [(1, 1)]
        # )
        # # Measure all possible combinations of bond pairing channels
        # # for the bonds we have defined. We will need each of these
        # # pairs channels measured in order to reconstruct the extended
        # # s-wave and d-wave pair susceptibilities.
        # # Initialize the pair correlation function measurement.
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "pair",
        #     time_displaced = false,
        #     integrated = true,
        #     pairs = [(1, 1),
        #             (bond_px_id, bond_px_id), (bond_px_id, bond_nx_id),
        #             (bond_nx_id, bond_px_id), (bond_nx_id, bond_nx_id),
        #             (bond_py_id, bond_py_id), (bond_py_id, bond_ny_id),
        #             (bond_ny_id, bond_py_id), (bond_ny_id, bond_ny_id),
        #             (bond_px_id, bond_py_id), (bond_px_id, bond_ny_id),
        #             (bond_nx_id, bond_py_id), (bond_nx_id, bond_ny_id),
        #             (bond_py_id, bond_px_id), (bond_py_id, bond_nx_id),
        #             (bond_ny_id, bond_px_id), (bond_ny_id, bond_nx_id)]
        # )

        # # Initialize the current correlation function measurement
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "current",
        #     time_displaced = false,
        #     integrated = false,
        #     eqt_average = eqt_average,
        #     tdp_average = tdp_average,
        #     pairs = [(1, 1), # hopping ID pair for x-direction hopping
        #             (2, 2)] # hopping ID pair for y-direction hopping
        # )
 
        # # Initialize the current correlation function measurement
        # initialize_correlation_measurements!(
        #     measurement_container = measurement_container,
        #     model_geometry = model_geometry,
        #     correlation = "bond",
        #     time_displaced = false,
        #     integrated = false,
        #     eqt_average = eqt_average,
        #     tdp_average = tdp_average,
        #     pairs = [(1, 1), # hopping ID pair for x-direction hopping
        #             (2, 2)] # hopping ID pair for y-direction hopping
        # )

        # initialize_measurement_directories(
        #     simulation_info = simulation_info,
        #     measurement_container = measurement_container
        # )


        
        # Initialize variable to keep track of the current burnin bin.
        n_bin_burnin = 1

        # Initialize variable to keep track of the current bin.
        n_bin = 1
        # Write an initial checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_current;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin, n_bin_burnin, n_bin,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            # electron_photon_parameters,
            hubbard_parameters,
            hubbard_ising_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab
            # ,photon_id=photon_id
        )



    # If resuming simulation from previous checkpoint.
    else
        @show simulation_info.pID," continue from ",checkpoint_filename_new
        # Initialize checkpoint to nothing before it is loaded.
        checkpoint = nothing

        # Try loading in the new checkpoint.
        if isfile(checkpoint_filename_new)
            try
                # Load the new checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_new)
            catch
                nothing
            end
        end

        # Try loading in the current checkpoint.
        if isfile(checkpoint_filename_current) && isnothing(checkpoint)
            try
                # Load the current checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_current)
            catch
                nothing
            end
        end

        # Try loading in the current checkpoint.
        if isfile(checkpoint_filename_old) && isnothing(checkpoint)
            try
                # Load the old checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_old)
                print()
            catch
                nothing
            end
        end

        # Throw an error if no checkpoint was succesfully loaded.
        if isnothing(checkpoint)
            error("Failed to load checkpoint successfully!")
        end

        # Unpack the contents of the checkpoint.
        rng                      = checkpoint["rng"]
        additional_info          = checkpoint["additional_info"]
        # N_burnin                 = checkpoint["N_burnin"]
        # N_updates                = checkpoint["N_updates"]
        # N_bins                   = checkpoint["N_bins"]
        # bin_size                 = checkpoint["bin_size"]
        # N_bins_burnin            = checkpoint["N_bins_burnin"]
        n_bin_burnin             = checkpoint["n_bin_burnin"]
        n_bin                    = checkpoint["n_bin"]
        model_geometry           = checkpoint["model_geometry"]
        measurement_container    = checkpoint["measurement_container"]
        tight_binding_parameters = checkpoint["tight_binding_parameters"]
        # electron_photon_parameters= checkpoint["electron_photon_parameters"]
        hubbard_parameters       = checkpoint["hubbard_parameters"]
        hubbard_ising_parameters = checkpoint["hubbard_ising_parameters"]
        δG                       = checkpoint["dG"]
        δθ                       = checkpoint["dtheta"]
        n_stab                   = checkpoint["n_stab"]
        photon_id                = checkpoint["photon_id"]
        Nt = additional_info["Nt"]
        Δt = additional_info["dt"]
        @show n_bin,n_bin_burnin
    end
    # Synchronize all the MPI processes.
    # We need to ensure the sub-directories the measurements will be written are created
    # prior to letting any of the processes move beyond this point.
    MPI.Barrier(comm)

       #############################
    ### SET-UP DQMC SIMULATION ##
    #############################

    # ## Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    # @show fermion_path_integral_up
    if U != 0
        ## Initialize the FermionPathIntegral type for both the spin-up and spin-down electrons.
        initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)
        initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_ising_parameters)
        @show fermion_path_integral_up
        @show hubbard_parameters
        @show "obkj"
    end
    ## Initialize the fermion path integral type with respect to electron-photon interaction.
    # initialize!(fermion_path_integral_up, fermion_path_integral_dn, electron_photon_parameters)

    ## Initialize the imaginary-time propagators for each imaginary-time slice for both the
    ## spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    ## Initialize FermionGreensCalculator for the spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    ## Initialize alternate fermion greens calculator required for performing various global updates.
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)
    
    ## Allcoate matrices for spin-up and spin-down electron Green's function matrices.
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    ## Initialize the spin-up and spin-down electron Green's function matrices, also
    ## calculating their respective determinants as the same time.
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)


    ########## Benchmark with free configuration here ########
    # δG = zero(typeof(logdetGup))
    # δθ = zero(Float64)
    # Gup_ττ = similar(Gup) # G↑(τ,τ)
    # Gup_τ0 = similar(Gup) # G↑(τ,0)
    # Gup_0τ = similar(Gup) # G↑(0,τ)
    # Gdn_ττ = similar(Gdn) # G↓(τ,τ)
    # Gdn_τ0 = similar(Gdn) # G↓(τ,0)
    # Gdn_0τ = similar(Gdn) # G↓(0,τ)
    # # @show measurement_container
    # (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
    #             measurement_container,
    #             logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
    #             logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
    #             fermion_path_integral_up = fermion_path_integral_up,
    #             fermion_path_integral_dn = fermion_path_integral_dn,
    #             fermion_greens_calculator_up = fermion_greens_calculator_up,
    #             fermion_greens_calculator_dn = fermion_greens_calculator_dn,
    #             Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
    #             model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
    #             coupling_parameters = (
    #                 electron_photon_parameters,
    #             )
    #         )
    # @show measurement_container,logdetGup
    # @show logdetGup
    # measurement_container.local
    # dasdaf
    #  ## Write the average measurements for the current bin to file.
    # write_measurements!(
    #     measurement_container = measurement_container,
    #     simulation_info = simulation_info,
    #     model_geometry = model_geometry,
    #     bin = 1,
    #     bin_size = 1,
    #     Δτ = Δτ
    # )
    # process_measurements(simulation_info.datafolder, 1)
    ########## Benchmark with free configuration here ########


    # ## Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    # hmc_updater = EFAHMCUpdater(
    #     electron_photon_parameters = electron_photon_parameters,
    #     G = Gup, Nt = Nt, Δt = Δt, reg = 0.0
    # )
    # hmc_updater = HMCUpdater(
    #     electron_photon_parameters = electron_photon_parameters,
    #     G = Gup, Nt = Nt, nt = Nt, Δt = Δt, reg = 0.0
    # )


    ## Allocate matrices for various time-displaced Green's function matrices.
    Gup_ττ = similar(Gup) # G↑(τ,τ)
    Gup_τ0 = similar(Gup) # G↑(τ,0)
    Gup_0τ = similar(Gup) # G↑(0,τ)
    Gdn_ττ = similar(Gdn) # G↓(τ,τ)
    Gdn_τ0 = similar(Gdn) # G↓(τ,0)
    Gdn_0τ = similar(Gdn) # G↓(0,τ)

    ## Initialize variables to keep track of the largest numerical error in the
    ## Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetGup))
    δθ = zero(Float64)
    # @show typeof(sgndetGup)
    # δθ = 0.0
    if iszero(simulation_info.pID)
        elapsed_time = time()-start_time
        @show "Initialization finished!", elapsed_time
    end
    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization bins.
    for bin in n_bin_burnin:N_bins_burnin
        # Iterate over updates in current bin.
        for n in 1:bin_size

            ## Perform a reflection update.
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                electron_photon_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, rng = rng, photon_types = (photon_id,)
            )

            ## Record whether the reflection update was accepted or rejected.
            additional_info["reflection_acceptance_rate"] += accepted

            ## Perform an HMC update.
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = hmc_update!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                electron_photon_parameters,
                hmc_updater,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn,
                δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            ## Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted
            
            if U != 0
                ## Perform a sweep through the lattice, attemping an update to each Ising HS field.
                (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                    Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                    hubbard_ising_parameters,
                    fermion_path_integral_up = fermion_path_integral_up,
                    fermion_path_integral_dn = fermion_path_integral_dn,
                    fermion_greens_calculator_up = fermion_greens_calculator_up,
                    fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                    Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
                )

                # Record the acceptance rate for the attempted local updates to the HS fields.
                additional_info["local_acceptance_rate"] += acceptance_rate
            end

            # if iszero(simulation_info.pID)
            #     # # @show additional_info["hmc_acceptance_rate"]/n
            #     # hmcacpt_ratio = additional_info["hmc_acceptance_rate"]/n
            #     # @show n,hmcacpt_ratio
            #     # @show bin,
            # end
            # if n%50 == 0 && iszero(simulation_info.pID)
            #     elapsed_time = time()-start_time
            #     @show "THERMALIZATION STEPS ", n, elapsed_time
            # end
        end
        if iszero(simulation_info.pID)
            elapsed_time = time()-start_time
            @show "THERMALIZATION Bin ", bin, elapsed_time
        end


        # Write the new checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin,
            n_bin_burnin = bin + 1,
            n_bin = 1,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            electron_photon_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab,photon_id=photon_id
        )
        # Make the current checkpoint the old checkpoint.
        mv(checkpoint_filename_current, checkpoint_filename_old, force = true)
        # Make the new checkpoint the current checkpoint.
        mv(checkpoint_filename_new, checkpoint_filename_current, force = true)
    end
    

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    ## Re-initialize variables to keep track of the largest numerical error in the
    ## Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetGup))
    δθ = zero(Float64)
    # δθ = zero(typeof(sgndetGup))

    ## Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in n_bin:N_bins
        ## Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            ## Perform a reflection update.
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                electron_photon_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, rng = rng, photon_types = (photon_id,)
            )

            ## Record whether the reflection update was accepted or rejected.
            additional_info["reflection_acceptance_rate"] += accepted

            ## Perform an HMC update.
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = hmc_update!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                electron_photon_parameters,
                hmc_updater,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn,
                δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            ## Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted
            if U != 0
                ## Perform a sweep through the lattice, attemping an update to each Ising HS field.
                (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                    Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                    hubbard_ising_parameters,
                    fermion_path_integral_up = fermion_path_integral_up,
                    fermion_path_integral_dn = fermion_path_integral_dn,
                    fermion_greens_calculator_up = fermion_greens_calculator_up,
                    fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                    Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
                )

                ## Record the acceptance rate for the attempted local updates to the HS fields.
                additional_info["local_acceptance_rate"] += acceptance_rate
            end
            # @show size(Gup),size(Gup_ττ),size(Gup_τ0)
            ## Make measurements, with the results being added to the measurement container.
            (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
                measurement_container,
                logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
                logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (
                    electron_photon_parameters,
                )
            )
        end
        

        if iszero(simulation_info.pID)
            # @show measurement_container.local_measurements
            Et = sum(measurement_container.local_measurements["hopping_energy"]) + sum(measurement_container.local_measurements["photon_pot_energy"] + measurement_container.local_measurements["photon_kin_energy"])/N
            Et/=bin_size
            elapsed_time = time()-start_time
            accept_ratio = additional_info["hmc_acceptance_rate"]/(N_burnin + bin*bin_size)
            @show bin, bin_size, elapsed_time, accept_ratio, Et
        end

        ## Write the average measurements for the current bin to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
        
         # Write the new checkpoint to file.
         JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin,
            n_bin_burnin = N_bins_burnin+1,
            n_bin = bin + 1,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            electron_photon_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab, photon_id=photon_id
        )
        # Make the current checkpoint the old checkpoint.
        mv(checkpoint_filename_current, checkpoint_filename_old, force = true)
        # Make the new checkpoint the current checkpoint.
        mv(checkpoint_filename_new, checkpoint_filename_current, force = true)
    end



    ## Calculate acceptance rates.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["reflection_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["local_acceptance_rate"] /= (N_updates + N_burnin)

    ## Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator_up.n_stab

    ## Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

#     #################################
#     ### PROCESS SIMULATION RESULTS ##
#     #################################

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

   # Synchronize all the MPI processes.
    # Before we prcoess the binned data to get the final averages and error bars
    # we need to make sure all the simulations running in parallel have run to
    # completion.
    MPI.Barrier(comm)
    

    #### compute the energy at each MC step to determine the Nburnin ####
    if iszero(simulation_info.pID)
        checkconverge(simulation_info.datafolder,0,N)
    end

    # # # Have the primary MPI process calculate the final error bars for all measurements,
    # # # writing final statisitics to CSV files.
    if iszero(simulation_info.pID)
        process_measurements(simulation_info.datafolder, 50, time_displaced=true, N_start = 11)
    end


    return nothing
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
    Nt = parse(Int, ARGS[17])
    at = parse(Float64, ARGS[18])
    init = parse(Int, ARGS[19])
    ## Run the simulation.
    run_photon_minicoup_square_simulation(sID, U, Ω, g, μ, β, Lx, Ly, PBCx, PBCy, N_burnin, N_updates, N_bins, eqt_average, tdp_average; filepath = "./data/"*prefix,Nt = Nt,at = at, init = init)
end
