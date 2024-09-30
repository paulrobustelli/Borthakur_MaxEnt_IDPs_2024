#!/bin/bash
pdb="/dartfs-hpc/rc/lab/R/RobustelliP/Kaushik/Paper_Data/PaaA2/PaaA2.pdb"
dcd="/dartfs-hpc/rc/lab/R/RobustelliP/Kaushik/Paper_Data/PaaA2/a99SBdisp/PaaA2-a99SBdisp-Traj.dcd"
output_file = "/dartfs-hpc/rc/lab/R/RobustelliP/Tommy/kaush/elvim/elvim.paa2.a99"
python3 elvim.py -f "$dcd" -t "$pdb" -o "$output_file"
