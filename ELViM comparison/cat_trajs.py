#!/usr/bin/env python
import mdtraj as md
import numpy as np
import sys

def concatenate(traj_files: np.ndarray,
                pdb_file: str,
                dscr: str, 
                stride: int=2):
    load = lambda x : md.load(x, top=pdb_file, stride=stride)
    cat_trj = md.join(list(map(load, traj_files)))
    cat_trj.save_dcd(f"{dscr}_cat.dcd")
    return cat_trj

if __name__ == "__main__":
    traj_files = np.load(sys.argv[1])
    pdb_file, dscr, stride = sys.argv[2], sys.argv[3], int(sys.argv[4])
    concatenate(traj_files, pdb_file, dscr, stride)
