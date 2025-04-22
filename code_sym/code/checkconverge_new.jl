##### this is to compute the obs with MC steps #####
using JLD2
using Glob
using Printf
#using Plots
# function checkconverge(folder,pID,N)
function checkconverge(folder, pIDs, N::Int)
    directory = joinpath(folder, "local")
    if isempty(pIDs)
        all_files = glob("*.jld2", directory)
        pIDs = unique([parse(Int, match(r"pID-(\d+)\.jld2", basename(f)).captures[1]) for f in all_files if occursin(r"pID-\d+\.jld2", basename(f))])
        sort!(pIDs)
    end

    # 找出最大的 Nfiles 以统一 bin 长度
    maxN = maximum([length(glob(@sprintf("*_pID-%d.jld2", pid), directory)) for pid in pIDs])

    Etot_mat = fill(NaN, maxN, length(pIDs))
    Eel_mat  = fill(NaN, maxN, length(pIDs))
    Ekin_mat = fill(NaN, maxN, length(pIDs))
    Epot_mat = fill(NaN, maxN, length(pIDs))

    for (j, pID) in enumerate(pIDs)
        println("Processing pID = $pID ...")
        ending = @sprintf("*_pID-%d.jld2", pID)
        files = glob(ending, directory)
        Nfiles = length(files)

        for i in 1:Nfiles
            filepath = @sprintf("%s/bin-%d_pID-%d.jld2", directory, i, pID)
            local_measurements = JLD2.load(filepath)
            Etot_mat[i, j] = real(sum(local_measurements["hopping_energy"]) +
                                  (sum(local_measurements["photon_pot_energy"]) +
                                   sum(local_measurements["photon_kin_energy"])) / N)
            Eel_mat[i, j] = real(sum(local_measurements["hopping_energy"]))
            Epot_mat[i, j] = real(sum(local_measurements["photon_pot_energy"]))
            Ekin_mat[i, j] = real(sum(local_measurements["photon_kin_energy"]))
        end
    end

    function write_matrix_to_csv(matrix, filename)
        open(joinpath(folder, filename), "w") do io
             # 写列名
            header = join(["pID$(pid)" for pid in pIDs], ",")
            println(io, header)
            for i in 1:size(matrix, 1)
                @printf(io, "%s\n", join(map(x -> isnan(x) ? "" : @sprintf("%.8f", x), matrix[i, :]), ","))
            end
        end
    end

    write_matrix_to_csv(Etot_mat, "Etot_pID-all.csv")
    write_matrix_to_csv(Eel_mat,  "Ehop_pID-all.csv")
    write_matrix_to_csv(Ekin_mat, "Ekin_pID-all.csv")
    write_matrix_to_csv(Epot_mat, "Epot_pID-all.csv")
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
    # print(prefix)
    init = parse(Int, ARGS[14])
    
    # prefix = ARGS[16]
    ## Run the simulation.
    filepath = "./data/"*prefix
    dir = @sprintf "%s/Lx%d_Ly%d_BC%d%d_b%.2f" filepath Lx Ly PBCx PBCy β
    datafolder_prefix =  @sprintf "%s/photon_minicoup_square_U%.2f_w%.2f_g%.2f_mu%.2f_Lx%d_Ly%d_BC%d%d_b%.2f_eqavg%d_tdavg%d_init%d-%d" dir U Ω g μ Lx Ly PBCx PBCy β eqt_average tdp_average init sID
    N = Lx*Ly
    checkconverge(datafolder_prefix,[],N)
end
