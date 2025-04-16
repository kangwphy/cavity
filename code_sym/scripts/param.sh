#!/bin/bash
sID=1
g=2.0
L=16
Lx=$L
Ly=$L
bt=$L
U=0
w=1
mu=0
N_burnin=0
N_updates=3000
N_bins=60

PBCx=0
PBCy=0
prefix=symGauge____0
eqt_average=false
tdp_average=false

if [ ! -d ./data/Lx${Lx}Ly${Ly}bt${bt}BC${PBCx}${PBCy} ];then
    mkdir -p $folder
fi


