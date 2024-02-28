import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
import sys
import random
import scipy as sp
from scipy import optimize
from scipy.optimize import least_squares
from scipy.special import erf
from time import time
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from os.path import join, exists
from platform import uname
from datetime import datetime
import matplotlib.patheffects as pe
from configparser import ConfigParser
import pickle

main_path = '/Users/kaushikb/Desktop/PaaA2/'
forcefields = ['Charmm36m']
for ff in forcefields:
    outdir = os.path.join(main_path, ff +'/analyses')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    dic_dir = os.path.join(main_path, ff +'/dictionaries')

    with open('%s/KishScan_single_data.pkl' % dic_dir, 'rb') as fp:
        KishScan_single = pickle.load(fp)

    with open('%s/KishScan_leave_one.pkl' % dic_dir, 'rb') as fp:
        Kish_leave = pickle.load(fp)

    with open('%s/KishScan_combined.pkl' % dic_dir, 'rb') as fp:
        Kish_all_data = pickle.load(fp)

    with open('%s/RMSE_single.pkl' % dic_dir, 'rb') as fp:
        RMSE_dict = pickle.load(fp)

    with open('%s/RMSE_leave_one.pkl' % dic_dir, 'rb') as fp:
        RMSE_leave_one_dict = pickle.load(fp)

    with open('%s/RMSE_combined.pkl' % dic_dir, 'rb') as fp:
        RMSE_dict_combined = pickle.load(fp)

    with open('%s/theta_dict.pkl' % dic_dir, 'rb') as fp:
        theta_dict = pickle.load(fp)

    with open('%s/theta_dict_combined.pkl' % dic_dir, 'rb') as fp:
        theta_dict_combined = pickle.load(fp)

    with open('%s/theta_dict_leave_one.pkl' % dic_dir, 'rb') as fp:
        theta_dict_leave_one = pickle.load(fp)

    with open('%s/colors.pkl' % dic_dir, 'rb') as fp:
        colors = pickle.load(fp)

    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1.5


    for key in KishScan_single:
        plt.plot(KishScan_single[key]['kish'][:, 0], KishScan_single[key]['kish'][:, 1], label='%s' % key,
                 color=colors[key])
    plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', fontsize=15)
    plt.xlabel(r'$\sigma_{reg}$', size=30)
    plt.ylabel('Kish Score (%)', size=30)
    xticks=np.linspace(0,20,6)
    plt.xticks(ticks=xticks ,size=20)
    plt.yticks(size=20)
    plt.axhline(y=10, color='black', linestyle='-')
    plt.savefig('%s/Kishscan.pdf'%outdir, bbox_inches='tight')
    plt.clf()

    for key in Kish_all_data:
        plt.plot(Kish_all_data[key]['kish'][:, 0], Kish_all_data[key]['kish'][:, 1], label='all restraints')
    plt.legend( loc='upper left', fontsize=15)
    plt.xlabel(r'$\sigma_{global}$', size=30)
    plt.ylabel('Kish Score (%)', size=30)
    xticks = np.linspace(0,20,6)
    plt.xticks(ticks=xticks, size=20)
    plt.yticks(size=20)
    plt.axhline(y=10, color='black', linestyle='-')
    plt.savefig('%s/Kishscan_combined.pdf'%outdir, bbox_inches='tight')
    plt.clf()

    combined_key_string = 'CA-CB-HA-H-N-C-RDC-SAXS'
    keys = set(combined_key_string.split("-"))

    for key in Kish_leave:
        k = keys - set(key.split("-"))  # complement of the set of all being restrained
        plt.plot(Kish_leave[key]['kish'][:, 0], Kish_leave[key]['kish'][:, 1], label=next(iter(k)), color=colors[next(iter(k))])
    plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left', fontsize=15)
    plt.xlabel(r'$\sigma_{reg}$', size=30)
    plt.ylabel('Kish Score (%)', size=30)
    xticks = np.linspace(0,20,6)
    plt.xticks(ticks=xticks, size=20)
    plt.yticks(size=20)
    plt.axhline(y=10, color='black', linestyle='-')
    plt.savefig('%s/Kishscan_leave_one.pdf'%outdir, bbox_inches='tight')
    plt.clf()

    for key in RMSE_dict:
        print(key)
        kish = []
        theta = []
        rmse_r = []
        rmse_i = []
        for i in RMSE_dict[key].keys():
            kish.append(RMSE_dict[key][i]['Kish'])
            theta.append(i)
            rmse_r.append(RMSE_dict[key][i]['r_f'][key])
        rmse_i = float(RMSE_dict[key][i]['r_i'][key])
        theta_rev = np.asarray(theta, dtype=float)[::-1]
        rmse_r_rev = np.asarray(rmse_r, dtype=float)[::-1]
        if (key == 'SAXS'):
            rmse_r_rev = rmse_r_rev / rmse_i
        kish_rev = np.asarray(kish, dtype=float)[::-1]
        f, ax = plt.subplots()
        ax.plot(theta_rev, rmse_r_rev, label=key, c='black', linestyle='dashed')
        # keyslist=RMSE_dict[key][i]['v_f'].keys()
        # colors = [f"C{j}" for j in range(len(keys_list))]
        for key2 in RMSE_dict[key][i]['v_f'].keys():
            rmse_v = []
            for k in RMSE_dict[key].keys():
                rmse_v.append(RMSE_dict[key][k]['v_f'][key2])
            rmse_v_i = float(RMSE_dict[key][k]['v_i'][key2])
            rmse_v_rev = np.asarray(rmse_v, dtype=float)[::-1]
            if (key2 == 'SAXS'):
                rmse_v_rev = rmse_v_rev / rmse_v_i
            ax.plot(theta_rev, rmse_v_rev, label=key2, color=colors[key2])

        ax.tick_params(labelsize=20)
        ax.legend(bbox_to_anchor=(1.2, 0.75), loc='upper left', fontsize=15)
        ax.set_title(f"Restraint : {key}", size=20)
        ax.set_xlabel(r'$\sigma_{reg}$', size=30)
        ax.set_ylabel("RMSE", size=30)
        ax.set_xlim(0.1, 5)

        ax2 = ax.twinx()
        ax2.plot(theta_rev, kish_rev, color='grey', ls='dotted', label='Kish Score')
        ax2.set_ylabel("Kish Score (%)", size=30)
        ax2.set_ylim(0, 100)
        ax2.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left', fontsize=12)
        ax2.tick_params(labelsize=20)
        plt.savefig('%s/%s.rmse_v_kish_v_sigma.pdf'%(outdir,key), bbox_inches="tight")
        plt.clf()

    def plot_Kish_RMSE_normalized(rmse_dict):
        for key in rmse_dict:
            rmse = []
            kish = []
            for i in rmse_dict[key].keys():
                kish.append(rmse_dict[key][i]['Kish'])
            for i in rmse_dict[key].keys():
                for key2 in rmse_dict[key][i]['v_f'].keys():
                    rmse.append(rmse_dict[key][i]['v_f'][key2])
            plt.plot(kish, rmse / rmse[0], label=key2, color=colors[key2])
            plt.tick_params(labelsize=20)
            plt.legend(bbox_to_anchor=(1.0, 0.9), loc='upper left', fontsize=15)
            plt.xlabel("Kish Score (%)", size=30)
            plt.ylabel("RMSE", size=30)
            plt.ylim(0.4,1.4)
            plt.savefig('%s/Kish_vs_RMSE_normalized.pdf' % outdir, bbox_inches='tight')


    plot_Kish_RMSE_normalized(RMSE_leave_one_dict)

    kish = []
    theta = []
    rmse_r = []
    for i in RMSE_dict_combined[combined_key_string].keys():
        kish.append(RMSE_dict_combined[combined_key_string][i]['Kish'])
        theta.append(i)
    theta_rev = np.asarray(theta, dtype=float)[::-1]
    kish_rev = np.asarray(kish, dtype=float)[::-1]
    f, ax = plt.subplots()
    for key2 in RMSE_dict_combined[combined_key_string][i]['r_f'].keys():
        rmse_r = []
        for k in RMSE_dict_combined[combined_key_string].keys():
            rmse_r.append(RMSE_dict_combined[combined_key_string][k]['r_f'][key2])
        rmse_r_rev = np.asarray(rmse_r, dtype=float)[::-1]
        ax.plot(theta_rev, rmse_r_rev, label=key2, color=colors[key2], linestyle='dashed')

    ax.tick_params(labelsize=20)
    ax.legend(bbox_to_anchor=(1.2, 0.75), loc='upper left', fontsize=15)
    ax.set_title("All Restraints", size=20)
    ax.set_xlabel(r'$\sigma_{reg}$', size=30)
    ax.set_ylabel("RMSE", size=30)

    ax2 = ax.twinx()
    ax2.plot(theta_rev, kish_rev, color='grey', ls='dotted', label='Kish Score')
    # ax2.plot(theta_rev,kish_rev,'.', color = 'grey', ls = 'dotted', label = 'Kish Score', lw = 0.9, alpha = 0.8)
    ax2.set_ylabel("Kish Score (%)", size=30)
    ax2.set_ylim(0, 100)
    ax2.legend(bbox_to_anchor=(1.2, 1.0), loc='upper left', fontsize=15)
    ax2.tick_params(labelsize=20)
    plt.axvline(x=2.1, linestyle='--', color='black')
    plt.savefig('%s/rmse_v_kish_v_sigma.pdf'%(outdir), bbox_inches='tight')

    reweighting_keys = ['CA', 'CB', 'HA', 'H', 'N', 'C', 'RDC', 'SAXS']
    keys = list(theta_dict.keys())
    keys2 = list(np.round(np.array(list(theta_dict.values())), 2))
    keys2 = ["{:.2f}".format(value) for value in keys2]
    dfs = []
    columns = "initial,final".split(',')
    for k, k2 in zip(keys, keys2):
        if k2 == "0.0": k2 += "0"
        data = []
        index = []
        index.append(k)
        data.append([RMSE_dict[k][k2][i][k] for i in "r_i,r_f".split(",")])
        for k3 in keys:
            if k3 != k:
                data.append([RMSE_dict[k][k2][i][k3] for i in "v_i,v_f".split(",")])
                index.append(k3)
        l = np.array(data)
        index = np.array(index)
        df = pd.DataFrame(data=l, index=index, columns=columns)
        dfs.append(df)
    df_dict = dict(zip(keys, dfs))

    keys = list(theta_dict_combined.keys())
    keys2 = list(np.round(np.array(list(theta_dict_combined.values())), 2))
    keys2 = ["{:.2f}".format(value) for value in keys2]
    keys3 = list(theta_dict.keys())
    dfs = []
    columns = "initial,final".split(',')
    for k, k2, k3 in zip(keys, keys2, keys3):
        if k2 == "0.0": k2 += "0"
        data = []
        index = []
        for k3 in keys3:
            data.append([RMSE_dict_combined[k][k2][i][k3] for i in "r_i,r_f".split(",")])
            index.append(k3)

        l = np.array(data)
        index = np.array(index)
        df = pd.DataFrame(data=l, index=index, columns=columns)
        dfs.append(df)
    df_dict_combined = dict(zip(keys, dfs))


    def subframe(df, key, key2):
        df_key = df_dict[key].assign(Unbiased_MD=df_dict[key][key2].round(3))
        df_key = df_key.drop(columns=['initial', 'final'])
        df_key = df_key.T[reweighting_keys].round(3)
        return df_key


    def subframe_combined(df, key, key2):
        df_key = df_dict_combined[key].assign(all_data=df_dict_combined[key][key2].round(3))
        df_key = df_key.drop(columns=['initial', 'final'])
        df_key = df_key.T[reweighting_keys].round(3)
        return df_key


    subframes_only_initial = subframe(df_dict['CA'], 'CA', 'initial')
    subframes_combined = subframe_combined(df_dict_combined[combined_key_string], combined_key_string, 'final')

    subframes_final = pd.concat([subframe(df_dict, i, 'final') for i in reweighting_keys])
    subframes_final.index = subframes_final.columns

    table = pd.concat([subframes_only_initial, subframes_combined, subframes_final])

    keys = list(theta_dict_leave_one.keys())
    keys2 = list(np.round(np.array(list(theta_dict_leave_one.values())), 2))
    keys2 = ["{:.2f}".format(value) for value in keys2]
    keys3 = list(theta_dict.keys())
    dfs = []
    columns = "initial,final".split(',')
    data_f = []
    data_i = []
    for k, k2 in zip(keys, keys2):
        if k2 == "0.0": k2 += "0"
        d_f = []
        d_i = []
        klist = k.split("-")
        for k3 in keys3:
            if k3 in set(klist):
                d_f.append(RMSE_leave_one_dict[k][k2]["r_f"][k3])
                d_i.append(RMSE_leave_one_dict[k][k2]["r_i"][k3])
            else:
                d_f.append(RMSE_leave_one_dict[k][k2]["v_f"][k3])
                d_i.append(RMSE_leave_one_dict[k][k2]["v_i"][k3])
        data_f.append(np.array(d_f))
        data_i.append(np.array(d_i))

    data_mat_f = np.stack(data_f, axis=0)
    data_mat_i = np.stack(data_i, axis=0)
    left_out = [list(set(keys3) - set(k.split("-")))[0] for k in keys]

    table_left_out = pd.DataFrame(data=data_f, index=left_out, columns=left_out)
    cross_validation = np.diagonal(table_left_out)

    table_leave_one_out = pd.concat([subframes_only_initial, subframes_combined, table_left_out])
    table_new = table.rename(columns={'CA': r'C$\alpha$', 'CB': r'C$\beta$', 'HA': r'H$\alpha$'},
                             index={'Unbiased_MD': 'Unbiased MD', 'all_data': 'All restraints',
                                    'CA': r'C$\alpha$', 'CB': r'C$\beta$', 'HA': r'H$\alpha$'})

    table_new['Average'] = np.average((table_new / (table_new.iloc[0])), axis=1).round(3)

    import matplotlib
    import seaborn as sns


    def plot_mat(df: "pandas dataframe", title, unit, cbarlabel, textsize, textcolor, cmap, filename,
                 fig_dims: tuple = (40, 40),
                 epsilon=0):
        """mat = square matrix
        unit = string specifying the units"""
        # ratio = emat/mat
        mat = df.to_numpy()
        ratio = mat / mat[0, :]
        divnorm = matplotlib.colors.TwoSlopeNorm(vmin=ratio.min(), vcenter=1., vmax=ratio.max() + epsilon)
        fig, ax = plt.subplots(1, figsize=fig_dims)
        s = sns.heatmap(pd.DataFrame(ratio, index=df.index, columns=df.columns), linewidths=1,
                        linecolor='black', cmap=cmap, norm=divnorm, ax=ax, cbar_kws={'label': cbarlabel, 'aspect': 15})
        s.figure.axes[-1].set_ylabel(cbarlabel, size=60)
        for i in range(len(mat)):
            for j in range(mat.shape[1]):
                c = mat[i, j]
                ax.text(j + .5, i + .5, f"{np.round(c, 3)}{unit}",
                        va='center', ha='center', color=textcolor, size=textsize, weight="bold")
        # ax.figure.axes[-1].yaxis.label.set_size( 40)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=70)
        ax.tick_params(labelsize=70, top=True)
        ax.tick_params(axis="y", rotation=0)
        ax.tick_params(axis="x", rotation=30)
        # ax.figure.axes[-1].yaxis.label.set_size(25)
        # ax.set_title(title, size = 25)
        ax.xaxis.tick_top()
        # ax.grid(b=True,which='minor',color='black', linestyle='-', linewidth=1, alpha=0.2)
        # plt.minorticks_on()
        figpath = os.path.join(outdir, filename)
        plt.savefig(figpath, bbox_inches="tight")
        return


    filename1 = "RMSE_table.pdf"
    plot_mat(table_new, "RMSEs", "", "fractional improvement", 45, "black", "bwr", filename1, epsilon=0.2)

    diagonal_array = np.diagonal(subframes_final)
    cross_validation = pd.DataFrame(data=cross_validation, columns=['Cross Validation'], index=left_out).T
    single_restraints = pd.DataFrame(data=diagonal_array, columns=['Single Restraints'], index=left_out).T
    table_left_out = pd.concat([subframes_only_initial, subframes_combined, single_restraints, cross_validation]).round(3)

    table_new_left_one = table_left_out.rename(columns={'CA': r'C$\alpha$', 'CB': r'C$\beta$', 'HA': r'H$\alpha$'},
                                               index={'Unbiased_MD': 'Unbiased MD', 'all_data': 'All restraints'})

    filename2 = 'RMSE_left_one.pdf'

    plot_mat(table_new_left_one, " ", "", "final / initial", 45, "black", "bwr", filename2, fig_dims=(40, 10), epsilon=0.6)












