#!/bin/bash

# this is an example script -- first 4 variables should be replaced with your own files

# first we need to concatenate the trajecory files we wish to analyze together with ELViM
dcds="path/to/numpy/array/containing/paths_to_protein_simulation_files.npy"
pdb="path/to/protein.pdb"
dscr="string_with_protein_name" # the output concatenated simulation file will be named "{dscr}_cat.dcd"
stride="int" # stride for simulation data

./cat_trajs.py "$dcds" "$pdb" "$dscr" "$stride"

# now we compute ELViM
concatenated_traj = ""$dscr"_cat.dcd" # this will be generated from cat_trajs.npy and does not need to be modified

elvim_output_file = ""$dscr"_elvim_cat.txt"

python3 elvim.py -f "$concatenated_traj" -t "$pdb" -o "$elvim_output_file"
