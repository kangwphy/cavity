#!/bin/bash
#SBATCH -p normal
##SBATCH -N 1
#SBATCH -J converge
##SBATCH --gres=dcu:4
##SBATCH --nodes=1:4        # 最小节点数
#SBATCH --ntasks=1            # 总的MPI进程数

##SBATCH --mem=64G
##SBATCH --exclude=g02r4n06,g02r4n07,g04r1n01,g06r4n15,g04r4n17,g06r4n16,g05r2n07,g07r4n15,g02r3n16,g08r2n03
#SBATCH --exclude=g03r2n10,g03r4n15,g03r4n11,g06r1n09,g04r4n00,g04r4n01
##SBATCH -MaxNodes=5
##SBATCH --exclusive 
# module purge
# module load compiler/devtoolset/7.3.1
# module load mpi/hpcx/2.11.0/gcc-7.3.1
# module load compiler/rocm/dtk/23.10
# source activate kangw





source ./scripts/param.sh
g=2.0
init=0
folder=./logs/${prefix}/Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy}/g${g}w${w}U${U}mu${mu}Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy}avg_${eqt_average}${tdp_average}init${init}
if [ ! -d $folder ];then
    mkdir -p $folder
fi

# ntask=80
# julia code/checkconverge_new.jl $sID $U $w $g $mu $bt $Lx $Ly $PBCx $PBCy $N_burnin $N_updates $N_bins $eqt_average $tdp_average $prefix > ${folder}/converge  2>&1 &

julia  code/checkconverge_new.jl $sID $U $w $g $mu $bt $Lx $Ly $PBCx $PBCy $eqt_average $tdp_average $prefix $init > ${folder}/converge  2>&1 &

wait
