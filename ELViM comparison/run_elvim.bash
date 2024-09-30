#!/bin/bash
pdb="/dartfs-hpc/rc/lab/R/RobustelliP/Kaushik/Paper_Data/PaaA2/PaaA2.pdb"
python3 elvim.py -f /dartfs-hpc/rc/lab/R/RobustelliP/Kaushik/Paper_Data/PaaA2/a99SBdisp/PaaA2-a99SBdisp-Traj.dcd -t $pdb -o /dartfs-hpc/rc/lab/R/RobustelliP/Tommy/kaush/elvim/elvim.paa2.a99
python3 elvim.py -f /dartfs-hpc/rc/lab/R/RobustelliP/Kaushik/Paper_Data/PaaA2/Charmm36m/PaaA2-Charmm36m-Traj.dcd -t $pdb -o /dartfs-hpc/rc/lab/R/RobustelliP/Tommy/kaush/elvim/elvim.paa2.c36
