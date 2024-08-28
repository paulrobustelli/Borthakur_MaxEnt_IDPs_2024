import os
import shutil
import numpy as np
import argparse
from Bio.PDB import PDBParser, PDBIO
from Bio.SeqUtils import seq1
from multiprocessing import Pool

pepsi_saxs_dir = os.path.dirname(os.path.realpath(__file__))
os.environ['PATH'] += os.pathsep + pepsi_saxs_dir

def save_saxs(filename, saxs_data, drho, r0):
    mode = "a" if os.path.isfile(filename) else "w"
    with open(filename, mode) as f:
        f.write("# drho = {} ; r0 = {}\n".format(drho, r0))
        for el in saxs_data:
            f.write("{:.8f}".format(el[0]))
            for ell in el[1:]:
                f.write(",{:.8f}".format(ell))
            f.write("\n")
    return 0

def save_ave_saxs(filename, saxs_data, drho, r0):
    mode = "a" if os.path.isfile(filename) else "w"
    with open(filename, mode) as f:
        f.write("# drho = {} ; r0 = {}\n".format(drho, r0))
        for el in saxs_data:
            f.write("{:.8f} {:.8f}\n".format(el[0], el[1]))

def process_frame(args):
    frame_idx, drho, r0, pdb_file, datapath, saxsdir = args

    # Output SAXS file name:
    temp_saxs_file = f"{saxsdir}/temp_saxs_{frame_idx}.out"
    temp_saxs_log = f"{saxsdir}/temp_saxs_{frame_idx}.log"

    # Run Pepsi-SAXS on the pdb file:
    #ret_code = os.system(f"Pepsi-SAXS {pdb_file} {datapath} --cstFactor 0.0 --dro {drho} --r0_min_factor {r0} --r0_max_factor {r0} --r0_N 1 -o {temp_saxs_file} > {temp_saxs_log} 2>&1")
    
    ret_code = os.system(f"Pepsi-SAXS {pdb_file} {datapath} --dro {drho} --r0_min_factor {r0} --r0_max_factor {r0} --r0_N 1 -o {temp_saxs_file} > {temp_saxs_log} 2>&1")

    if ret_code != 0:
        raise RuntimeError(f"Pepsi-SAXS failed for frame {frame_idx} with return code {ret_code}. Check log: {temp_saxs_log}")

    # Read SAXS data
    d = np.loadtxt(temp_saxs_file)
    q = d[:, 0]
    I = d[:, 3]

    return q, I

def calc_saxs(pdb_dir, workdir, datapath, pdb_range=None, drho=3.34, r0=1.68, nproc=1, nchunk=1):
    saxsdir = os.path.join(workdir, "saxs")
    os.makedirs(saxsdir, exist_ok=True)

    if pdb_range is None:
        pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")], key=lambda x: int(x.split('.')[0]))
        start_frame, end_frame = 1, len(pdb_files)
    else:
        start_frame, end_frame = pdb_range
        pdb_files = [f"{t}.pdb" for t in range(start_frame, end_frame + 1)]

    pdb_files = [os.path.join(pdb_dir, pdb_file) for pdb_file in pdb_files]

    calc_frames = []
    args_list = [(t, drho, r0, pdb_files[t - start_frame], datapath, saxsdir) for t in range(start_frame, end_frame + 1)]

    with Pool(processes=nproc) as pool:
        results = pool.map(process_frame, args_list, nchunk)

    for t, (q, I) in enumerate(results):
        if t == 0:
            calc_frames.append(q)
        calc_frames.append(I)

    calc_array = np.asarray(calc_frames)
    I_calc = np.average(calc_array, axis=0)

    save_saxs(f"{saxsdir}/SAXS_drho_{drho}_r0_{r0}.csv", calc_array.T, drho, r0)
    save_ave_saxs(f"{saxsdir}/SAXS_drho_{drho}_r0_{r0}.ave.dat", np.column_stack((q, I_calc)), drho, r0)

    return 0

def main():
    parser = argparse.ArgumentParser(description="Run Pepsi-SAXS on a range of PDB files with multiprocessing.")
    parser.add_argument('pdb_dir', type=str, help="Directory containing PDB files.")
    parser.add_argument('datapath', type=str, help="Path to the experimental data file.")
    parser.add_argument('workdir', type=str, help="Working directory.")
    parser.add_argument('--start_frame', type=int, help="Start frame number.")
    parser.add_argument('--end_frame', type=int, help="End frame number.")
    parser.add_argument('--nproc', type=int, default=1, help="Number of processors to use.")
    parser.add_argument('--nchunk', type=int, default=1, help="Chunk size for multiprocessing.")

    args = parser.parse_args()

    pdb_range = (args.start_frame, args.end_frame) if args.start_frame is not None and args.end_frame is not None else None

    calc_saxs(args.pdb_dir, args.workdir, args.datapath, pdb_range, drho=3.34, r0=1.68, nproc=args.nproc, nchunk=args.nchunk)

if __name__ == "__main__":
    main()

