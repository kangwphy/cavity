#!/bin/bash
#SBATCH -p fat6326
#SBATCH -N 1
#SBATCH -J L16_0_1.8
##SBATCH --gres=dcu:4
##SBATCH --nodes=1:4        # 最小节点数
#SBATCH --ntasks=30             # 总的MPI进程数

##SBATCH --mem=64G
##SBATCH --exclude=g02r4n06,g02r4n07,g04r1n01,g06r4n15,g04r4n17,g06r4n16,g05r2n07,g07r4n15,g02r3n16,g08r2n03
##SBATCH --exclude=g03r2n10,g03r4n15,g03r4n11,g06r1n09,g04r4n00,g04r4n01
##SBATCH -MaxNodes=5
#SBATCH --exclusive 
# module purge
# module load compiler/devtoolset/7.3.1
# module load mpi/hpcx/2.11.0/gcc-7.3.1
# module load compiler/rocm/dtk/23.10
# source activate kangw

source ./scripts/param.sh
ntask=30
Nt=8
at=0.45
sID=1
g=1.8
init=0
folder=./logs/${prefix}/Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy}/g${g}w${w}U${U}mu${mu}Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy}avg_${eqt_average}${tdp_average}init${init}
if [ ! -d $folder ];then
    mkdir -p $folder
fi

mpiexec -n $ntask julia code/mainMPI_spinless.jl $sID $U $w $g $mu $bt $Lx $Ly $PBCx $PBCy $N_burnin $N_updates $N_bins $eqt_average $tdp_average $prefix $Nt $at $init >> ${folder}/${sID}_${Nt}_${at}  2>&1 &
# julia mainMPI.jl 1 0 1 1 0 1 2 2 0 0 10 10 1 false false . > phy  2>&1 &
# julia mainMPI.jl 1 0 1 0 0 4 4 4 0 0 10 10 1 false false . > phy  2>&1 &

wait
