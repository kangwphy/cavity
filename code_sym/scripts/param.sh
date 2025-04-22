#!/bin/bash
sID=1
g=2.0
L=4
Lx=$L
Ly=$L
bt=$L
U=1
w=1
mu=0
N_burnin=0
N_updates=1200
N_bins=12

PBCx=1
PBCy=1
prefix=symGauge
eqt_average=false
tdp_average=false

if [ ! -d ./data/Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy} ];then
    mkdir -p $folder
fi


