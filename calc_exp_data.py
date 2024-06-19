import sys
import mdtraj as md
import os
import shutil
import math
import numpy as np
import textwrap
from Bio.PDB import *
from Bio.SeqUtils import seq1
import argparse

# check python version
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

# create parser
parser = argparse.ArgumentParser(prog='python calc_exp_data.py', \
         formatter_class=argparse.RawDescriptionHelpFormatter, \
         epilog=textwrap.dedent('''\
Required software/libraries:
- Python 3.x: https://www.python.org
- catdcd: https://www.ks.uiuc.edu/Development/MDTools/catdcd
- SPARTA+: https://spin.niddk.nih.gov/bax/software/SPARTA+
- PALES: https://spin.niddk.nih.gov/bax/software/PALES
- mdtraj: http://mdtraj.org
- numpy: https://numpy.org
- pandas: https://pandas.pydata.org
- Biopython: https://biopython.org
 '''))
# define arguments
parser.add_argument('top',     metavar='structure.xxx',  nargs=1,   type=str, help='topology/structure file [pdb,gro,mae,...]')
parser.add_argument('traj',    metavar='traj.xxx',       nargs='*', type=str, help='list of trajectory files [xtc,trr,dcd,...]')
parser.add_argument('--itask', metavar='I', default=[0], nargs=1,   type=int, help='task id (for splitting trajectory)')
parser.add_argument('--ntask', metavar='N', default=[1], nargs=1,   type=int, help='number of tasks (for splitting trajectory)')
# define which experimental data will be calculated
parser.add_argument('--cs',        action='store_true', default=False, help='calculate chemical shifts')
parser.add_argument('--rdc',       action='store_true', default=False, help='calculate RDCs')
parser.add_argument('--Jcoupling', action='store_true', default=False, help='calculate 3J scalar couplings')
parser.add_argument('--pre',       metavar="resID", nargs='*', type=int, default=[],    help='list of residue numbers for PRE spin label')
parser.add_argument('--debug',     action='store_true', default=False, help='keep temporary directories')
# parse arguments
args = parser.parse_args()

# name of the topology file
TOP_=vars(args)['top'][0]
# get PREFIX and TYPE
PREFIX_ = (os.path.splitext(TOP_)[0]).split("/")[-1]
TYPE_   = (os.path.splitext(TOP_)[1]).split(".")[1]
# define pdb name
PDB_=PREFIX_+".pdb"
# list of trajectory files
TRAJ_=vars(args)['traj']
# tasks info
ITASK_=vars(args)['itask'][0]
NTASK_=vars(args)['ntask'][0]

# create a working directory labelled by ITASK_
wdir="task-"+str(ITASK_)
os.mkdir(wdir)

# convert to pdb
# This requires catdcd installed:
# https://www.ks.uiuc.edu/Development/MDTools/catdcd/
if(TYPE_!="pdb"):
 os.system("catdcd -o "+wdir+"/"+PDB_+" -otype pdb -stype "+TYPE_+" -s "+TOP_+" -"+TYPE_+" "+TOP_)
else:
 os.system("cp "+TOP_+" "+wdir+"/"+PDB_)

# read trajectory files and topology (from pdb)
# this requires mdtraj
# http://mdtraj.org/1.9.3/
trj = md.load(TRAJ_, top=wdir+"/"+PDB_)
# remove the pdb for all tasks except 0
if(ITASK_!=0): os.remove(wdir+"/"+PDB_)

# slice the trajectory based on ITASK_ and NTASK_ 
n_frames=int(math.floor(float(trj.n_frames)/float(NTASK_)))
# set initial and final frame
first_frame = ITASK_ * n_frames
last_frame  = first_frame + n_frames
# adjust last task
if(ITASK_==NTASK_-1): last_frame = trj.n_frames
# do the actual slicing
trj=trj.slice(range(first_frame,last_frame), copy=False)

# calculate number of residues
nres=[]
for res in trj.topology.residues: nres.append(res.resSeq)

# Print information about the system to log file
log = open(wdir+"/log", "w")
log.write("** SYSTEM INFO **\n")
log.write("Structure filename: %s\n" % PDB_)
log.write("Trajectory filenames: %s\n" % str(TRAJ_))
log.write("Number of atoms: %d\n" % trj.n_atoms)
log.write("Number of residues: %d\n" % len(set(nres)))
log.write("Number of frames: %d\n" % trj.n_frames)
log.write("Starting frame: %d\n" % first_frame)
log.write("Last frame: %d\n" % last_frame)
log.write("Task id: %d\n" % ITASK_)
log.write("Number of tasks: %d\n" % NTASK_)

# Define format for output
fmt0='%d,'; fmt1=''
for i in range(0, trj.n_frames-1): fmt1+='%.4lf,'
fmt1+='%.4lf'

# Calculate chemical shifts
# This requires SPARTA+ installed and pandas
# https://spin.niddk.nih.gov/bax/software/SPARTA+/
# https://pandas.pydata.org
if(vars(args)['cs']):
  log.write("- Calculating chemical shifts\n")
  cs=md.chemical_shifts_spartaplus(trj, rename_HN=True)
  # print the panda DataFrame to csv
  cs.to_csv(wdir+"/chemical_shifts.csv", float_format="%.3lf")

# Calculate RDCs
# This requires PALES installed
# https://spin.niddk.nih.gov/bax/software/PALES/ 
if(vars(args)['rdc']):
 log.write("- Calculating RDC\n")
 for t in range(0,trj.n_frames):
    # create a temporary directory
    tmpdir=wdir+"/tmp-"+str(t)
    os.mkdir(tmpdir)
    # save pdb file
    ipdb = tmpdir+"/out.pdb"
    trj[t].save_pdb(ipdb)
    # clean it - this requires Bio.PDB
    # https://biopython.org/wiki/Download
    structure = PDBParser().get_structure('PDB', ipdb)
    # get sequence info (residue name and number)
    # WARNING: assuming pdb with 1 model and 1 chain
    resname=[]; resnum=[] 
    for i in structure[0].get_chains():
     for j in i.get_residues(): 
         resname.append(j.get_resname())
         resnum.append(j.get_id()[1])
    # get sequence (one letter code)
    seq=seq1("".join(resname))
    # number of residues
    nres=len(resname)
    # sanity check of sequence length
    if(nres!=len(seq)):
      "Check length of the protein failed!"
      exit()
    # save clean pdb
    io = PDBIO()
    io.set_structure(structure)
    opdb = tmpdir+"/out-clean.pdb"
    io.save(opdb)
    # create PALES input file
    ifile = tmpdir+"/PALES_input.dat"
    f = open(ifile,"w")
    f.write("DATA SEQUENCE ")
    for i in range(0, nres):
        f.write("%s" % seq[i])
        # add a space every 10 residues, but not for last one
        if((i+1)%10==0 and i!=(nres-1)): f.write(" ")
    f.write("\n\n")
    f.write("VARS   RESID_I RESNAME_I ATOMNAME_I RESID_J RESNAME_J ATOMNAME_J D      DD    W\n")
    f.write("FORMAT %5d     %6s       %6s        %5d     %6s       %6s    %9.3f   %9.3f %.2f\n")
    f.write("\n")
    for i in range(0, nres):
        f.write("%d %3s H %d %3s N 0 1.00 1.00\n" % (resnum[i],resname[i],resnum[i],resname[i])) 
    f.close() 
    # run PALES on clean pdb
    rdc_frame=[]
    # cycle on residues, except first and last
    for ires in range(1, nres-1):
        # determine window
        l=7; h=7
        if(ires<7):      l=ires
        if(ires>nres-8): h=nres-1-ires
        # window is the minimum between l and h
        w=min(l,h)
        # output file
        ofile = tmpdir+"/"+str(resnum[ires])+".dat"
        # run PALES
        os.system("/dartfs-hpc/rc/home/k/f0044gk/storage/pales/linux/pales -inD "+ifile+" -pdb "+opdb+" -r1 "+str(resnum[ires-w])+" -rN "+str(resnum[ires+w])+" -outD "+ofile)
        # parse the output file and add to list of RDCs 
        for lines in open(ofile, "r").readlines():
            riga=lines.strip().split()
            if(len(riga)==12 and riga[0].isdigit()):
              if(float(riga[0])==resnum[ires]): rdc_frame.append(float(riga[8]))
    # delete the temporary directory
    if(vars(args)['debug']==False): shutil.rmtree(tmpdir)
    # create/add to global numpy array (n_data, n_frames)
    if(t==0):
      rdc=np.array(rdc_frame)
    else:
      rdc=np.column_stack((rdc,np.array(rdc_frame)))
 # save RDCs to file
 label=np.array(resnum[1:nres-1])
 np.savetxt(wdir+"/RDC.csv", np.column_stack((label,rdc)), fmt=fmt0+fmt1, header="resSeq,frame")

# Calculate phi on the entire trajectory
# indices : np.ndarray, shape=(n_phi, 4), C(i-1),N(i),Ca(i),C(i)
# The indices of the atoms involved in each of the phi dihedral angles
# phis : np.ndarray, shape=(n_frames, n_phi)
# The value of the dihedral angle phi for each of the residue in each of the frames.
if(vars(args)['Jcoupling']):
  log.write("- Calculating 3J scalar couplings\n")
  indices, phis = md.compute_phi(trj)
  # create list of labels (residue index)
  phi_label=[]
  # cycle on the number of phi dihedrals (n_phi)
  for i_phi in range(0, indices.shape[0]):
      resindex=trj.topology.atom(indices[i_phi][3]).residue.resSeq
      phi_label.append(resindex)
  # convert to numpy array
  phi_label=np.array(phi_label)
  # calculate 3J scalar couplings according to
  # Karplus Coefficient For 3J_HNHA From Vuister Bax JACS 93
  A0 = [6.51, -1.76, 1.60, math.radians(-60)]
  # Karplus Coefficient For 3J_HNHA From Vogeli Bax JACS 07
  A1 = [7.97, -1.26, 0.63, math.radians(-60)]
  # Karplus Coefficient For 3J_CC
  B  = [1.61, -0.93, 0.66]
  # calculate 3Js on the entire trajectory
  J_HNHA0 = ( A0[0] * np.power(np.cos(phis+A0[3]),2) + A0[1] * np.cos(phis+A0[3]) + A0[2] ).transpose()
  J_HNHA1 = ( A1[0] * np.power(np.cos(phis+A1[3]),2) + A1[1] * np.cos(phis+A1[3]) + A1[2] ).transpose()
  J_CC    = (  B[0] * np.power(np.cos(phis),2)       +  B[1] * np.cos(phis)       +  B[2] ).transpose()
  # print 3Js to file
  np.savetxt(wdir+"/J_HNHA0.csv",  np.column_stack((phi_label,J_HNHA0)), fmt=fmt0+fmt1, header="resSeq,frame")
  np.savetxt(wdir+"/J_HNHA1.csv",  np.column_stack((phi_label,J_HNHA1)), fmt=fmt0+fmt1, header="resSeq,frame")
  np.savetxt(wdir+"/J_CC.csv",     np.column_stack((phi_label,J_CC)),    fmt=fmt0+fmt1, header="resSeq,frame")

# list of PRE labels
label_list=vars(args)['pre']
if(len(label_list)>0):
  log.write("- Calculating PRE\n") 
  # Calculte PREs for specified spin label positions
  # first residue
  res0 = trj.topology.atom(0).residue.resSeq
  for label in label_list:
   # select the CA atom of the spin labeled residue
   label_index=trj.topology.select("name CA and residue "+str(label))
   H_indices=[]
   # select all backbone H atoms - except first residue
   H_indices=trj.topology.select("name H and not residue "+str(res0))
   # determine the residue number for selected atoms
   res_label=[]
   for i in H_indices: 
     resindex=trj.topology.atom(i).residue.resSeq
     res_label.append(resindex)
   # convert to numpy array
   res_label=np.array(res_label)
   # make a list of atom pairs between spin labelled CA and all H atoms
   atom_pairs=np.zeros(shape=(len(H_indices),2))
   atom_pairs[:,0]=np.asarray(H_indices)
   atom_pairs[:,1]=label_index 
   # Compute distances in nanometers
   distances=md.compute_distances(trj, atom_pairs)
   # define magnetic field strength and other PRE constants
   # TODO: read them from command line
   b=18.8; k=1.23e-32; tdelay=0.010 
   R2exp=4.0; b=18.8; tc=4.0e-9
   wH=2.6752e8*b*2.0*np.pi
   # calculate PRE
   PRE = k*np.power(distances,-6)*(4.0*tc+3.0*tc/(1.0+(wH*tc)**2))*1.0e42
   # calculate Iratios
   Iratio = R2exp*np.exp(-tdelay*PRE)/(PRE+R2exp)
   # save to file
   np.savetxt(wdir+"/Iratio-label-"+str(label)+".csv", np.column_stack((res_label,Iratio.T)), fmt=fmt0+fmt1, header="resSeq,frame")

# get mass list
log.write("- Calculating radius of gyration\n")
mass=[]
for at in trj.topology.atoms:
    mass.append(at.element.mass)
# calculate mass-weighted radius of gyration on all atoms
rg_all=md.compute_rg(trj, masses=np.array(mass))
# write to file
np.savetxt(wdir+"/Rgyr-all.csv", np.array([rg_all]), header="Rgyr-all", fmt=fmt1)
# prepare an empty mass list
mass_CA = len(mass)*[0.0]
# put the CA entries equal to 1.0
for i in trj.topology.select("name CA"): mass_CA[i]=1.0
# calculate CA radius of gyration
rg_CA=md.compute_rg(trj, masses=np.array(mass_CA))
# write to file
np.savetxt(wdir+"/Rgyr-CA.csv", np.array([rg_CA]), header="Rgyr-CA", fmt=fmt1)

# closing log file
log.write("ALL DONE!\n")
log.close()
