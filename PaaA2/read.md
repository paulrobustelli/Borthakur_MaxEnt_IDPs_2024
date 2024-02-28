The notebooks are examples of the reweighting, plotting all the figures+tables and doing all the ensemble comparison plots for one trajectory (PaaA2 - Charmm36m / TIP3P)


calc_exp_data.py is the script to calculate NMR data from forward models (SPARTA+, PALES) by doing:
python calc_exp_data.py <pdb_name>.pdb <trajectory_name>.dcd --cs --rdc --Jcoupling
