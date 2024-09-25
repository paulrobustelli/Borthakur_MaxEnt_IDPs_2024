import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
import sys
import random
import scipy as sp
from scipy import optimize
from scipy.optimize import least_squares, leastsq
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from os.path import join, exists
import matplotlib.patheffects as pe
from configparser import ConfigParser

def Weighted_Independent_Blocks(x, w, block_dim, verbose = False):

    #  returns a reshaped version of a random variable x with its weights w
    #  by considering as independent not all the elements of x, but blocks of a dimension dim (decided a priori)

    dim = len(x)
    n_block = int(dim/block_dim)

    if n_block == 0: n_block = 1

    X = np.zeros(n_block)
    W = np.zeros(n_block)

    for ii in range(n_block):

        i0 = ii*block_dim
        i1 = (ii+1)*block_dim

        if i1> dim:
            X[ii] = np.average(x[i0:], weights=w[i0:])
            W[ii] = np.sum(w[i0:])
            break

        else:
            X[ii] = np.average(x[i0:i1], weights=w[i0:i1])
            W[ii] = np.sum(w[i0:i1])

    if verbose: print('Completed.\nInitial dimension = {}\nBlock dimension = {}\nn_block = {}'.format(dim, block_dim, n_block))

    return X, W


def Get_Sigma_from_Bootstrapping(X, W, n_boot, n_block, ):

    """
    Returns the std deviation by bootstrapping analysis for the population X with weights W
    (OSS: in our case for now X, W come out from block analysis performed with Weighted_Independent_Blocks() )
    """

    ave_boot = np.zeros(n_boot)
    for ii_boot in range(n_boot):
        blocks = random.choices(range(n_block), k = n_block)
        ave_boot[ii_boot] = np.average(X[blocks], weights=W[blocks])

    # return the sigma
    return np.std(ave_boot)


def Weighted_Sigma_of_Mean(x, w,):

    # return the best estimator of the variance of the mean of a random variable x with weights w

    return np.sqrt((np.sum(w**2 * (x - np.average(x, weights = w))**2))/(np.sum(w)**2))


def read_cs_md(data):
    confd = {'HA': {},
             'CA': {},
             'CB': {},
             'C': {},
             'H': {},
             'HA2': {},
             'HA3': {},
             'N': {}}
    for lines in open(data, "r").readlines():
        riga = lines.strip().split(",")
        key = riga[0]
        key2 = riga[1]
        confd[key2][key] = []
        timeseries = np.asarray(riga[2:-1]).astype(float)
        confd[key2][key] = timeseries
    return (confd)

def read_exp_cs(data):
    confd = {'HA': {},
             'CA': {},
             'CB': {},
             'C': {},
             'H': {},
             'HA2': {},
             'HA3': {},
             'N': {}}

    for lines in open(data, "r").readlines():
        riga = lines.strip().split(" ")
        key = riga[0]
        key2 = riga[1]
        confd[key2][key] = []
        confd[key2][key] += map(float, [float(riga[2])])
    return (confd)


def add_md_data(data, dict, key, ):
    dict[key] = {}
    for lines in open(data, "r").readlines():
        riga = lines.strip().split(",")
        key2 = riga[0]
        timeseries = np.asarray(riga[1:-1]).astype(float)
        dict[key][key2] = []
        dict[key][key2] = (timeseries)
    return (dict)


def add_exp_data(data, dict, key):
    dict[key] = {}
    for lines in open(data, "r").readlines():
        # riga=lines.strip().split(",")
        riga = lines.strip().split(" ")
        key2 = riga[0]
        dict[key][key2] = []
        dict[key][key2] += map(float, [float(riga[1])])
    return (dict)


def add_md_scalar(data, dict, key):
    dict[key] = {}
    dict[key][0] = np.loadtxt(data)
    return (dict)


def add_md_scalar_skip0(data, dict, key):
    dict[key] = {}
    dict[key][0] = np.loadtxt(data)[0:-1]
    return (dict)


def add_exp_scalar(data, dict, key):
    dict[key] = {}
    dict[key][0] = [data]
    return (dict)


def add_md_scalar_skip0(data, dict, key):
    dict[key] = {}
    dict[key][0] = np.loadtxt(data)[1:]
    return (dict)


def add_exp_scalar(data, dict, key):
    dict[key] = {}
    dict[key][0] = [data]
    return (dict)

def scale_rdc_Q_fit(exp, calc):
    def residuals(p, y, x):
        c = p
        err = ((c * y) - x)
        return err

    p0 = 0.0001
    exp = np.trim_zeros(exp)
    calc = np.trim_zeros(calc)
    Q_i = (np.sum(np.square(exp - calc)) / (np.sum(np.square(exp)))) ** .5
    c, d = leastsq(residuals, p0, args=(calc, exp))
    fit = c * calc
    Q_f = (np.sum(np.square(exp - fit)) / (np.sum(np.square(exp)))) ** .5
    rmsd_i = (sum(np.square(calc - exp)) / len(exp)) ** 0.5
    rmsd_f = (sum(np.square(fit - exp)) / len(exp)) ** 0.5
    return Q_i, rmsd_i, Q_f, rmsd_f, fit


def scale_rdc_Q(exp, calc):
    exp = np.trim_zeros(exp)
    calc = np.trim_zeros(calc)
    Q_i = (np.sum(np.square(exp - calc)) / (np.sum(np.square(exp)))) ** .5
    c = np.linalg.norm(np.dot(exp, calc)) / (np.dot(calc, calc))
    fit = c * calc
    Q_f = (np.sum(np.square(exp - fit)) / (np.sum(np.square(exp)))) ** .5
    rmsd_f = (sum(np.square(fit - exp)) / len(exp)) ** 0.5
    rmsd_i = (sum(np.square(calc - exp)) / len(exp)) ** 0.5
    return Q_i, rmsd_i, Q_f, rmsd_f, fit, c


def Separate_Validation_Reweighting_Data(CS, true_exp_labels):
    if ((type(CS) == list) & (len(CS) >= 2)):

        Reweight_Data = []  # true_exp_labels.copy()
        for cs in CS:
            Reweight_Data.append(cs)

        Validation_Data = true_exp_labels.copy()
        for cs in CS:
            Validation_Data.remove(cs)

        # create a label string for more-than-one reweighting data
        string = CS[0]
        for cs in CS[1:]:
            string += '-{}'.format(cs)

        CS = string


    else:

        if type(CS) == list:
            CS = CS[0]

        Validation_Data = true_exp_labels.copy()
        Validation_Data.remove(CS)

        Reweight_Data = [CS]

        CS = str(CS)

    return CS, Reweight_Data, Validation_Data


def Align_Comp_Exp_Data(compdata, expdata):
    for data1, data1_t in zip([compdata, expdata], ['comp', 'exp']):
        data2 = expdata if data1_t == 'comp' else compdata
        to_align = {}
        for key1 in data1.keys():
            to_align[key1] = []
            for key2 in data1[key1].keys():
                if ((key2 not in data2[key1].keys()) | (data1[key1][key2][0] == 0.)):
                    to_align[key1].append(key2)

        for key1 in to_align.keys():
            for key2 in to_align[key1]:
                if key2 in data1[key1].keys():
                    data1[key1].pop(key2)
                if key2 in data2[key1].keys():
                    data2[key1].pop(key2)

    return compdata, expdata


def Print_Number_of_Data(nframes, nobs_r, nobs_v, data_r_type, data_v_type):
    print('# of Frames:', nframes)

    print('Reweight Data Points:', nobs_r)
    for key in data_r_type:
        print(str(key) + ':', sum(data_r_type[key]))

    print('Validation Data Points:', nobs_v)
    for key in data_v_type:
        print(str(key) + ':', sum(data_v_type[key]))



def Process_Data_Into_Arrays(data, expdata, compdata, err_d, compare_dict):
    ### OBS: this process takes into account that for the chemical shift we DO NOT HAVE experimental data on border residues --> the data_type boolean mask count zero for those residues even if MD provide such data

    ### data represent either Reweight_Data either Validation_Data
    ### same operationas

    exp = []
    traj = []
    err = []
    data_t = []
    data_type = {}

    # Process Reweight data into arrays
    for key in data:
        for key2 in expdata[key]:
            # print(key,key2,expdata[key][key2][0])
            # Check if Experimental Value is 0, if not retrieve the MD timeseries
            if expdata[key][key2][0] != 0:
                if key2 in compdata[key]:
                    if compdata[key][key2][0] != 0:
                        compare_dict[key][key2] = ([expdata[key][key2][0], compdata[key][key2]])
                        traj.append(compdata[key][key2])
                        exp.append(expdata[key][key2][0])
                        err.append(err_d[key])
                        data_t.append(str(key))

    for key in data:
        data_id = np.zeros(len(data_t))
        for i, id in enumerate(data_t):
            if id == str(key):
                data_id[i] = float(1)
        data_type[key] = data_id

    return np.asarray(exp), np.asarray(traj), np.asarray(err), data_t, data_type, compare_dict


def Process_Data_Into_Arrays_with_sigma(data, expdata, compdata, err_d, sigma_dict, compare_dict):
    ### OBS: this process takes into account that for the chemical shift we DO NOT HAVE experimental data on border residues --> the data_type boolean mask count zero for those residues even if MD provide such data

    ### data represent either Reweight_Data either Validation_Data
    ### same operationas

    exp = []
    traj = []
    err = []
    data_t = []
    sigmas = []
    data_type = {}

    # Process Reweight data into arrays
    for key in data:
        for key2 in expdata[key]:
            # Check if Experimental Value is 0, if not retrieve the MD timeseries
            if expdata[key][key2][0] != 0:
                if key2 in compdata[key]:
                    if compdata[key][key2][0] != 0:
                        compare_dict[key][key2] = ([expdata[key][key2][0], compdata[key][key2]])
                        traj.append(compdata[key][key2])
                        exp.append(expdata[key][key2][0])
                        sigmas.append(sigma_dict[key][key2][-1])
                        err.append(err_d[key])
                        data_t.append(str(key))

    for key in data:
        data_id = np.zeros(len(data_t))
        for i, id in enumerate(data_t):
            if id == str(key):
                data_id[i] = float(1)
        data_type[key] = data_id

    return np.asarray(exp), np.asarray(traj), np.asarray(err), np.asarray(sigmas), data_t, data_type, compare_dict


def Normalize_Weights(weights, ):
    ## OBS the argument is not really a weight, is the log...

    weights -= np.max(weights)
    weights = np.exp(weights)
    weights /= np.sum(weights)

    return weights


def Print_RMSE(data_type, obs, obs_exp, ):
    print(" * Total :     %6.3lf" % np.sqrt(np.mean((obs - obs_exp) ** 2)))
    for t in data_type:

        print(" *    %2s :" % t, end='')
        print("     %6.3lf" % np.sqrt(np.sum((obs - obs_exp) ** 2 * data_type[t]) / np.sum(data_type[t])))
        if str(t) == 'RDC':
            qi, rms_i, qf, rms_f, rdc_scale = scale_rdc_Q(obs * data_type['RDC'], obs_exp * data_type['RDC'])
            print(" *    RDC scaled Q:", end='')
            print(" %6.3lf" % qf)
    if 'RDC' in data_type:
        return qi, rms_i, qf, rms_f, rdc_scale
    else:
        return [None] * 5


def RMSE(theor, exp):
    if len(theor) != len(exp): raise ValueError(
        'Incopatible lenght between theor ({}) and exp ({})'.format(len(theor), len(exp)))
    theor = np.array(theor)
    exp = np.array(exp)
    return np.sqrt(np.sum(((theor - exp) ** 2) / (len(exp))))


def Calculate_RMSE(data_type, obs, obs_exp, ):
    RMSEs = {}
    RMSEs['Tot'] = RMSE(obs, obs_exp)
    for t in data_type.keys():
        RMSEs[t] = RMSE(np.trim_zeros(data_type[t] * obs), np.trim_zeros(data_type[t] * obs_exp))
    return RMSEs


def Get_Sigma_from_Independent_Blocks(traj, weights, key, start, stop, step, idx_check=3, min_dim=1000, bootstrap=False,
                                      n_boot=None, verbose=None, fig=False, spec_res=False):
    # "wrapper" for the function Weighted_Independent_Block to be utilized for all data of reweighting project
    # It works with dict-organized trajectory data: the keys are residue number, their value is the quantity over simulation

    nres = len(traj)
    nframes = len(traj[list(traj.keys())[0]])

    if verbose >= 1:
        print(
            '\n******************\nAnalyzing statistical errors for {} data\nNframes = {}\tNdata(res) = {}'.format(key,
                                                                                                                   nframes,
                                                                                                                   nres))

    ### ITERATION OVER BLOCK DIMENSION

    tot_start = time()

    block_dim = np.arange(start, stop + 1, step)

    df_block = pd.DataFrame(columns=('Saturation_Value', 'Block_Dim', 'Last_Derivative', 'Fit_Success', 'Elapsed_Time'),
                            index=traj.keys())

    if verbose >= 2:
        print('Beginning block analysis')

    for key2 in traj.keys():

        # to perform only on one residue
        if ((spec_res != False) & (key2 != spec_res)):
            if verbose >= 1: print('Skippato {}: spec_res = {}'.format(key2, spec_res))
            pass

        else:

            averages = []
            sigmas = []

            a = traj[key2]
            w = weights

            if verbose >= 2:
                print('\n{} Data\tResidue {}'.format(key, key2))
                print('N frames = {}\nWeighted average: {}\nInitial weighted std variation: {}'.format(nframes,
                                                                                                       np.average(a,
                                                                                                                  weights=weights),
                                                                                                       Weighted_Sigma_of_Mean(
                                                                                                           a, w)))

            start_block = time()

            for dim, ll in zip(block_dim, range(len(block_dim))):

                n_block = int(nframes / dim)

                A, W = Weighted_Independent_Blocks(a, w, dim, verbose=False)
                ave = np.average(A, weights=W)
                sigma = Weighted_Sigma_of_Mean(A, W) if not bootstrap else Get_Sigma_from_Bootstrapping(A, W, n_boot,
                                                                                                        n_block)

                # saturation check
                if ((ll > idx_check) & (dim > min_dim)):
                    if (sigma <= sigmas[idx_check]):
                        stopped = True
                        if verbose >= 3:
                            print('We are in noise regime: I stop block analysis for {}'.format(key + key2))
                        break
                else:
                    stopped = False

                averages.append(ave)
                sigmas.append(sigma)

            # end of block analysis
            stop_idx = ll if stopped else ll - 1
            end_block = time()
            block_time = end_block - start_block

            if verbose >= 1:
                print('Elapsed time for {}{} block analysis: {:3.2f} s'.format(key, key2, block_time))

                ########### FIT and estimation of saturation values

            # print(Saturation_Residuals(p0,block_dim[:stop_idx], sigmas[:stop_idx] ))

            try:
                p0 = [-.1, .01, .1, .1]
                fit = least_squares(Saturation_Residuals, x0=p0, args=[block_dim[:stop_idx], sigmas[:stop_idx]],
                                    bounds=((-np.inf, -np.inf, 0, 0,), (np.inf, np.inf, np.inf, np.inf)),
                                    verbose=verbose - 1 if verbose != 0 else verbose)
                success = fit.success
                x = block_dim[:stop_idx]
                y = saturation_func(x, fit.x)
                sat_dim = x[-1]
                last_ddx = saturation_derivative(x[-1], fit.x)

                first_value = y[0]
                sat_value = y[-1]

                if fig:
                    f, ax = plt.subplots()
                    ax.hlines(sat_value, x[0], x[-1], ls='dotted', color='grey', label='sat value')
                    ax.plot(x, y, c='firebrick', ls='dashed', alpha=0.6, label='fit')
                    ax.plot(block_dim[:stop_idx], np.array(sigmas[:stop_idx]), marker='o', ms=2.6, markerfacecolor='k',
                            ls='dashed', color='goldenrod', label='{}: res {}'.format(key, key2))
                    ax.set_title('Block analysis for MD statistical error', fontweight='bold')
                    ax.set_xlabel('Dimension of blocks')
                    ax.set_ylabel('Sigma of average')
                    ax.legend(loc='lower right')
                    plt.show(block=False)
                    plt.pause(0.0003)
                    plt.close()


            except ValueError:

                print('Skipped fit')
                sat_value = max(sigmas[:stop_idx])
                sat_dim = dim
                last_ddx = np.nan
                first_value = sigmas[0]
                success = False

            df_block.at[key2, 'Average'] = np.average(A, weights=W)
            df_block.at[key2, 'Saturation_Value'] = sat_value
            df_block.at[key2, 'Block_Dim'] = sat_dim
            df_block.at[key2, 'Fit_Success'] = success
            df_block.at[key2, 'Elapsed_Time'] = block_time
            df_block.at[key2, 'Last_Derivative'] = last_ddx
            df_block.at[key2, 'First_Value'] = first_value

    # FINAL PERFORMANCES
    tot_end = time()
    if verbose >= 1:
        print('\n\nElapsed time for all the analysis: {:3.2f} s'.format(tot_end - tot_start))
        print('Converged fit: {}/{}'.format(np.sum(df_block.Fit_Success.values), len(df_block)))

    if df_block.shape[0] != nres:
        raise ValueError('Porcamadonna: {} ha cambiato le cose'.format(key))
    return df_block


def saturation_func(x, p):
    return (p[0] / (x + p[1]) ** p[2]) + p[3]


def saturation_derivative(x, p):
    return -p[0] * p[2] * (x + p[1]) ** -(p[2] + 1)


def Saturation_Residuals(p, x, y):
    return saturation_func(x, p) - y


def Print_Separator(char='#', how_many=70, file=sys.stdout):
    print('\n\n{}\n'.format('{}'.format(char) * int(how_many)), file=file)


def func_sigma_reg_sigma_md(l, traj_r, obs_exp_r, weight_bias, sigma_reg, sigma_md):
        l = np.array(l)  # ensure array
        weight_all = +weight_bias  # copy
        weight_all -= np.dot(l, traj_r)  # maxent correction
        shift_all = np.max(weight_all)  # shift to avoid overflow
        weight_all = np.exp(weight_all - shift_all)  # compute weights
        weight_0 = +weight_bias  # copy
        shift_0 = np.max(weight_0)  # shift to avoid overflow
        weight_0 = np.exp(weight_0 - shift_0)  # compute weights

        # Gamma function in maxent:
        # Shifts to avoid overflows
        f = np.log(np.sum(weight_all) / np.sum(weight_0)) + shift_all - shift_0 + np.dot(l, obs_exp_r)

        # derivative of Gamma function:
        der = obs_exp_r - np.dot(traj_r, weight_all) / np.sum(weight_all)  # derivative with respect to l

        f += 0.5 * np.sum((sigma_reg * sigma_reg) * l ** 2 + sigma_md * sigma_md * l ** 2)
        der += sigma_reg * sigma_reg * l + sigma_md * sigma_md * l
        return (f, der)


def initialize_reweight(To_Scan):
    for CS in To_Scan:
            CS, Reweight_Data, Validation_Data = Separate_Validation_Reweighting_Data(CS, true_exp_labels)
            print('Reweight Data = {}'.format(CS))
            print('Validation Data = {}'.format(Validation_Data))

            ### debugging variables
            compare = []
            compare_dict_r = {key: {} for key in compdata.keys()}
            compare_dict_v = {key: {} for key in compdata.keys()}

            ####### B) The reweighting procedure

            # i) Process Reweight data into arrays "à la Paul"
            obs_exp_r, traj_r, err_r, data_r_t, data_r_type, compare_dict_r = Process_Data_Into_Arrays(Reweight_Data,
                                                                                                    expdata, compdata,
                                                                                                    theta_0,
                                                                                                    compare_dict_r)

            # ii) Process Validation data into arrays "à la Paul"
            obs_exp_v, traj_v, err_v, data_v_t, data_v_type, compare_dict_v = Process_Data_Into_Arrays(Validation_Data,
                                                                                                    expdata, compdata,
                                                                                                    theta_0,
                                                                                                    compare_dict_v)

            # final_weights

            for key in data_v_type:
                print(key)
                res = []
                md_ave = []
                sigmas = []
                exp = []
                sd[key] = {}
                for key2 in compare_dict_v[key]:
                    timeseries = compare_dict_v[key][key2][1]
                    uniform_weights = np.ones(len(timeseries))
                    dim = 5000
                    A, W = Weighted_Independent_Blocks(timeseries, uniform_weights, dim, verbose=False)
                    ave = np.average(A, weights=W)
                    sigma = Weighted_Sigma_of_Mean(A, W)
                    # sd[key][key2] = np.array([key2,float(compare_dict_v[key][key2][0]),float(ave),float(sigma)])
                    sd[key][key2] = np.array([float(key2), float(compare_dict_v[key][key2][0]), float(ave), float(sigma)])

            for key in data_r_type:
                print(key)
                res = []
                md_ave = []
                sigmas = []
                exp = []
                sd[key] = {}
                for key2 in compare_dict_r[key]:
                    timeseries = compare_dict_r[key][key2][1]
                    uniform_weights = np.ones(len(timeseries))
                    dim = 5000
                    A, W = Weighted_Independent_Blocks(timeseries, uniform_weights, dim, verbose=False)
                    ave = np.average(A, weights=W)
                    sigma = Weighted_Sigma_of_Mean(A, W)
                    # sd[key][key2] = np.array([key2,float(compare_dict_r[key][key2][0]),float(ave),float(sigma)])
                    sd[key][key2] = np.array([float(key2), float(compare_dict_r[key][key2][0]), float(ave), float(sigma)])



def reweight_function(To_Scan):
    for CS in To_Scan:
        CS, Reweight_Data, Validation_Data = Separate_Validation_Reweighting_Data(CS, true_exp_labels)
        KishScan_one_data[CS] = {}
        print('Reweight Data = {}'.format(CS))
        RMSE_dict[CS] = {}

        ### debugging variables
        compare = []
        compare_dict_r = {key: {} for key in compdata.keys()}
        compare_dict_v = {key: {} for key in compdata.keys()}

        ####### B) The reweighting procedure

        obs_exp_r, traj_r, err_r, sigma_md_r, data_r_t, data_r_type, compare_dict_r = Process_Data_Into_Arrays_with_sigma(
            Reweight_Data, expdata, compdata, theta_0, sd, compare_dict_r)
        obs_exp_v, traj_v, err_v, sigma_md_v, data_v_t, data_v_type, compare_dict_v = Process_Data_Into_Arrays_with_sigma(
            Validation_Data, expdata, compdata, theta_0, sd, compare_dict_v)

        # THETA CYCLE
        theta_list = []
        kish_list = []
        rmsd_f_list = []
        for theta_m in np.flip(thetas):

            s = '{:.2f}'.format(theta_m)
            # print(s)
            sigma_reg = theta_m * err_r
            sigma_md = sigma_md_r
            RMSE_dict[CS][s] = {}

            # iv) Print dimensions
            nobs_r = len(obs_exp_r)
            nobs_v = len(obs_exp_v)
            # Print_Number_of_Data(nframes, nobs_r, nobs_v, data_r_type, data_v_type)

            # v) perform minimization
            # OBS: res.X = lagrange multiplier
            weight_bias = np.ones(nframes)

            if 'RDC' in Reweight_Data:
                initial_weights = weight_bias
                initial_weights /= np.sum(initial_weights)
                initial_obs_r = np.dot(traj_r, initial_weights)
                initial_obs_r
                exp_rdc = np.trim_zeros(obs_exp_r * data_r_type['RDC'])
                calc_rdc = np.trim_zeros(initial_obs_r * data_r_type['RDC'])
                qi_pos, rms_i_pos, qf_pos, rms_f_pos, rdc_scale_pos, c_pos = scale_rdc_Q(exp_rdc, calc_rdc)
                qi_neg, rms_i_neg, qf_neg, rms_f_neg, rdc_scale_neg, c_neg = scale_rdc_Q(-exp_rdc, calc_rdc)

                if (qf_neg < qf_pos):
                    c = -c_neg
                else:
                    c = c_pos

                RDC_rows = np.where(data_r_type['RDC'] == 1)
                traj_r[RDC_rows] = traj_r[RDC_rows] * c

            if 'RDC' in Validation_Data:
                initial_weights = weight_bias
                initial_weights /= np.sum(initial_weights)
                initial_obs_v = np.dot(traj_v, initial_weights)
                exp_rdc = np.trim_zeros(obs_exp_v * data_v_type['RDC'])
                calc_rdc = np.trim_zeros(initial_obs_v * data_v_type['RDC'])
                qi_pos, rms_i_pos, qf_pos, rms_f_pos, rdc_scale_pos, c_pos = scale_rdc_Q(exp_rdc, calc_rdc)
                qi_neg, rms_i_neg, qf_neg, rms_f_neg, rdc_scale_neg, c_neg = scale_rdc_Q(-exp_rdc, calc_rdc)

                if (qf_neg < qf_pos):
                    c = -c_neg
                else:
                    c = c_pos

                RDC_rows = np.where(data_v_type['RDC'] == 1)
                traj_v[RDC_rows] = traj_v[RDC_rows] * c

            res = sp.optimize.minimize(func_sigma_reg_sigma_md,
                                       args=(traj_r, obs_exp_r, weight_bias, sigma_reg, sigma_md),
                                       x0=np.zeros((nobs_r,)), method='L-BFGS-B', jac=True)
            initial_weights = Normalize_Weights(weight_bias)
            initial_obs_r = np.dot(traj_r, initial_weights)
            initial_obs_v = np.dot(traj_v, initial_weights)

            final_weights = Normalize_Weights(weight_bias - np.dot(res.x, traj_r))
            final_obs_r = np.dot(traj_r, final_weights)
            final_obs_v = np.dot(traj_v, final_weights)
            # g) calculating Kish effective size
            Ks_b = np.average(initial_weights) ** 2 / np.average(initial_weights ** 2)
            Ks_a = np.average(final_weights) ** 2 / np.average(final_weights ** 2)

            Kish_ratio = (Ks_a / Ks_b) * 100
            RMSE_initial = np.sqrt(np.mean((initial_obs_r - obs_exp_r) ** 2))
            RMSE_reweight = np.sqrt(np.mean((final_obs_r - obs_exp_r) ** 2))

            theta_list.append(theta_m)
            kish_list.append(Kish_ratio)
            rmsd_f_list.append(RMSE_reweight)
            RMSE_r_i = {}
            RMSE_v_i = {}
            RMSE_r_f = {}
            RMSE_v_f = {}
            RMSE_r_i['Tot'] = np.sqrt(np.mean((initial_obs_r - obs_exp_r) ** 2))

            for t in data_r_type:
                RMSE_r_i[t] = np.sqrt(
                    np.sum((initial_obs_r - obs_exp_r) ** 2 * data_r_type[t]) / np.sum(data_r_type[t]))
                if str(t) == 'RDC':
                    qi, rms_i, qf, rms_f, rdc_scale_i, c = scale_rdc_Q(initial_obs_r * data_r_type['RDC'],
                                                                       obs_exp_r * data_r_type['RDC'])
                    RMSE_r_i['RDC'] = qf

            RMSE_v_i['Tot'] = np.sqrt(
                np.sum((initial_obs_r - obs_exp_r) ** 2 * data_r_type[t]) / np.sum(data_r_type[t]))
            for t in data_v_type:
                RMSE_v_i[t] = np.sqrt(
                    np.sum((initial_obs_v - obs_exp_v) ** 2 * data_v_type[t]) / np.sum(data_v_type[t]))
                if str(t) == 'RDC':
                    qi, rms_i, qf, rms_f, rdc_scale_i, c = scale_rdc_Q(initial_obs_v * data_v_type['RDC'],
                                                                       obs_exp_v * data_v_type['RDC'])
                    RMSE_v_i['RDC'] = qf

            RMSE_r_f['Tot'] = np.sqrt(np.mean((final_obs_r - obs_exp_r) ** 2))

            for t in data_r_type:
                RMSE_r_f[t] = np.sqrt(np.sum((final_obs_r - obs_exp_r) ** 2 * data_r_type[t]) / np.sum(data_r_type[t]))
                if str(t) == 'RDC':
                    qi, rms_i, qf, rms_f, rdc_scale_i, c = scale_rdc_Q(final_obs_r * data_r_type['RDC'],
                                                                       obs_exp_r * data_r_type['RDC'])
                    RMSE_r_f['RDC'] = qf

            for t in data_v_type:
                RMSE_v_f[t] = np.sqrt(np.sum((final_obs_v - obs_exp_v) ** 2 * data_v_type[t]) / np.sum(data_v_type[t]))
                if str(t) == 'RDC':
                    qi, rms_i, qf, rms_f, rdc_scale_i, c = scale_rdc_Q(final_obs_v * data_v_type['RDC'],
                                                                       obs_exp_v * data_v_type['RDC'])
                    RMSE_v_f['RDC'] = qf

            RMSE_dict[CS][s]['Kish'] = Kish_ratio
            RMSE_dict[CS][s]['r_i'] = RMSE_r_i
            RMSE_dict[CS][s]['r_f'] = RMSE_r_f
            RMSE_dict[CS][s]['v_i'] = RMSE_v_i
            RMSE_dict[CS][s]['v_f'] = RMSE_v_f

        KishScan_one_data[CS]['kish'] = np.column_stack((theta_list, kish_list))
        KishScan_one_data[CS]['rmsd'] = np.column_stack((theta_list, rmsd_f_list))



def print_results():
    print(CS, "Theta:", theta_m, "Kish: %9.6lf" % (Kish_ratio), "RMSD initail: %0.4f" % RMSE_initial,
            "RMSD final: %0.4f" % RMSE_reweight)

    print("Initial RMSE reweight data ")
    print(" * Total :     %6.3lf" % np.sqrt(np.mean((initial_obs_r - obs_exp_r) ** 2)))

    for t in data_r_type:
        print(" *    %2s :" % t, end='')
        print("     %6.3lf" % np.sqrt(
            np.sum((initial_obs_r - obs_exp_r) ** 2 * data_r_type[t]) / np.sum(data_r_type[t])))

        if str(t) == 'RDC':
            qi, rms_i, qf, rms_f, rdc_scale_i, c = scale_rdc_Q(initial_obs_r * data_r_type['RDC'],
                                                                obs_exp_r * data_r_type['RDC'])

            print(" *    RDC scaled Q:", end='')
            print(" %6.3lf" % qf)

    print("Initial RMSE validation data")
    print(" * Total :     %6.3lf" % np.sqrt(np.mean((initial_obs_v - obs_exp_v) ** 2)))

    for t in data_v_type:
        print(" *    %2s :" % t, end='')
  
        print("     %6.3lf" % np.sqrt(
            np.sum((initial_obs_v - obs_exp_v) ** 2 * data_v_type[t]) / np.sum(data_v_type[t])))
        if str(t) == 'RDC':
            qi, rms_i, qf, rms_f, rdc_scale_i, c = scale_rdc_Q(initial_obs_v * data_v_type['RDC'],
                                                                obs_exp_v * data_v_type['RDC'])
 
            print(" *    RDC scaled Q:", end='')
            print(" %6.3lf" % qf)

    print("Final RMSE reweight data")
    print(" * Total :     %6.3lf" % np.sqrt(np.mean((final_obs_r - obs_exp_r) ** 2)))

    for t in data_r_type:
        print(" *    %2s :" % t, end='')
        print("     %6.3lf" % np.sqrt(
            np.sum((final_obs_r - obs_exp_r) ** 2 * data_r_type[t]) / np.sum(data_r_type[t])))
        if str(t) == 'RDC':
            qi, rms_i, qf, rms_f, rdcs_scale_f, c = scale_rdc_Q(final_obs_r * data_r_type['RDC'],
                                                                obs_exp_r * data_r_type['RDC'])

            print(" *    RDC Q_scaled:", end='')
            print(" %6.3lf" % qf)

    print("Final RMSE validation data")
    print(" * Total :     %6.3lf" % np.sqrt(np.mean((final_obs_v - obs_exp_v) ** 2)))

    for t in data_v_type:
        print(" *    %2s :" % t, end='')
        print("     %6.3lf" % np.sqrt(
            np.sum((final_obs_v - obs_exp_v) ** 2 * data_v_type[t]) / np.sum(data_v_type[t])))
        if str(t) == 'RDC':
            qi, rms_i, qf, rms_f, rdc_scale_f, c = scale_rdc_Q(final_obs_v * data_v_type['RDC'],
                                                                obs_exp_v * data_v_type['RDC'])
           
            print(" *    RDC Q_scaled:", end='')
            print(" %6.3lf" % qf)



def print_comb_results():
    print(CS, "Theta:", theta_m, "Kish: %9.6lf" % (Kish_ratio), "RMSD initail: %0.4f" % RMSE_initial,
          "RMSD final: %0.4f" % RMSE_reweight)

    # Iniitial RMSE and Sigma
    print("Initial RMSE reweight data ")
    print(" * Total :     %6.3lf" % np.sqrt(np.mean((initial_obs_r - obs_exp_r) ** 2)))

    for t in data_r_type:
        print(" *    %2s :" % t, end='')
        print(
            "     %6.3lf" % np.sqrt(np.sum((initial_obs_r - obs_exp_r) ** 2 * data_r_type[t]) / np.sum(data_r_type[t])))

        if str(t) == 'RDC':
            qi, rms_i, qf, rms_f, rdc_scale_i, c = scale_rdc_Q(initial_obs_r * data_r_type['RDC'],
                                                               obs_exp_r * data_r_type['RDC'])
            # print(" *    RDC scaled RMSD:", end = '')
            # print(" %6.3lf" % rms_f)
            print(" *    RDC scaled Q:", end='')
            print(" %6.3lf" % qf)

    print("Final RMSE reweight data")
    print(" * Total :     %6.3lf" % np.sqrt(np.mean((final_obs_r - obs_exp_r) ** 2)))

    for t in data_r_type:
        print(" *    %2s :" % t, end='')
        print("     %6.3lf" % np.sqrt(np.sum((final_obs_r - obs_exp_r) ** 2 * data_r_type[t]) / np.sum(data_r_type[t])))
        if str(t) == 'RDC':
            qi, rms_i, qf, rms_f, rdcs_scale_f, c = scale_rdc_Q(final_obs_r * data_r_type['RDC'],
                                                                obs_exp_r * data_r_type['RDC'])
            # print(" *    RDC RMSE_scale:", end = '')
            # print(" %6.3lf" % rms_f)
            print(" *    RDC Q_scaled:", end='')
            print(" %6.3lf" % qf)


