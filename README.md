
# Determining accurate conformational ensembles of intrinsically disordered proteins at atomic resolution

This is the github repository for the manuscript "Determining accurate conformational ensembles of intrinsically disordered proteins at atomic resolution" by Kaushik Borthakur, Thomas R. Sisk, Francesco P. Panei , Massimiliano Bonomi, and Paul Robustelli.

https://doi.org/10.1101/2024.10.04.616700

This repository includes notebooks and scripts to calculate experimental data from MD trajectories and perform maximum entropy reweighting of MD ensembles using experimental data as restraints. 


## PUBLICATION

To learn more about our Maximum Entropy Reweighting Method please check out our preprint:
https://doi.org/10.1101/2024.10.04.616700

## DOCUMENTATION

1. Download the Max_ent_env.yml file and activate the environment by "conda env create -f Max_ent_env.yml"

2. Run the calc_exp.py script as follows:
  - ./calc_exp.py <pdb> <xtc/dcd> 
  - You can add flags for whatever experimental data you need calculated eg --cs --rdc --jcoupling --pre
  - Add the residue numbers for which you want to calculate the PRES for (eg for ACTR ./calc_exp.py --pre 3 21 41 61)
  - The script requires SPARTA+ and PALES installed in your environment


3. Run the Pepsi-SAXS calculation script as follows:
 *  -./calc_saxs.py <pdb_dir> <exp_file> <work_dir> --nproc --nchunk
(the script requires the user to have the Pepsi-SAXS executable in the working directory or for the user to point the script towards the executable by editing the script. The script uses multiprocessing, and you can customize the number of processors and chunk size
you want to use, with --nproc for number of processors and --nchunk for chunksize on each processor.

* If Pepsi-SAXS is not automatically installed when activating the environment, it can be downloaded from https://team.inria.fr/nano-d/software/pepsi-saxs/

4. Ensemble computed data for the a99SBdisp, c22star and Charmm36m trajectories for PaaA2, ACTR, drkN, alpha-synuclein and Ab40 proteins will be uploaded shortly

5. For any protein system run the {protein}_{forcefieldname}_reweight.ipynb notebook. This notebook does the actual reweighting. After all the rounds of reweighting
have been completed, the notebook automatically saves all the relevant information and weights in their own separate directories.

6. After the reweighting notebook, for the same protein system run the {protein}_{forcefield}_analysis.ipynb notebook
This notebook loads the relevant dictionaries that were generated in step 5, and plots all the necessary tables and figures.

7. Run the {protein}_{forcefield}_ensembles.ipynb notebook. Requirements for the notebook are:
- a structure file such as a pdb file for your protein system
- an xtc or dcd file for your simulation
- a perfectly ideal helix structure for alpha-helical order paramter calculations (can be generated by the make_helix.py script)
- weights from your reweighting run (We use the weights from the combined data-type as restraint run, but any weights would do)
This notebook loads the relevant required files and calculates the before and after reweighting ensemble properties such as secondary structure propensities,
inter-residue contact maps, Radius of gyration distributions etc.


 ## CONTACT INFORMATION
 If you have any queries about the paper and/or the code please reach out to:
kaushik.borthakur.gr@dartmouth.edu,
paul.j.robustelli@dartmouth.edu



   
