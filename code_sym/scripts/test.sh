#!/bin/bash
#SBATCH -p fat6326
#SBATCH -N 1
#SBATCH -J L16_0_1.8
##SBATCH --gres=dcu:4
##SBATCH --nodes=1:4        # 最小节点数
#SBATCH --ntasks=30             # 总的MPI进程数


sID=1
L=4
Lx=$L
Ly=$L
bt=$L
U=1
w=1
mu=0
N_burnin=0
N_updates=5
N_bins=1

PBCx=0
PBCy=0
prefix=test
eqt_average=false
tdp_average=false

if [ ! -d ./data/Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy} ];then
    mkdir -p $folder
fi


Nt=8
at=1
sID=0
g=2.0
init=0
folder=./logs/${prefix}/Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy}/g${g}w${w}U${U}mu${mu}Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy}avg_${eqt_average}${tdp_average}init${init}
if [ ! -d $folder ];then
    mkdir -p $folder
fi


# julia code/hubbardtest_lessmeas.jl $sID $U $w $g $mu $bt $Lx $Ly $PBCx $PBCy $N_burnin $N_updates $N_bins $eqt_average $tdp_average $prefix $Nt $at $init > ${folder}/${sID}_${Nt}_${at}   &
julia code/mainMPI_spinless.jl $sID $U $w $g $mu $bt $Lx $Ly $PBCx $PBCy $N_burnin $N_updates $N_bins $eqt_average $tdp_average $prefix $Nt $at $init > ${folder}/${sID}_${Nt}_${at}   &


# sID, U, Ω, α, μ, β, L, N_burnin, N_updates, N_bins
# julia code/hoshubbard.jl $sID $U $w $g $mu $bt $Lx $N_burnin $N_updates $N_bins > ${folder}/${sID}_${Nt}_${at}   &
# $Ly $PBCx $PBCy $N_burnin $N_updates $N_bins $eqt_average $tdp_average $prefix $Nt $at $init > ${folder}/${sID}_${Nt}_${at}   &

wait
