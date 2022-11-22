"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last modified: Jan 2022

Description: This script generates figures for experiments p1 vs p2 vs p3

Usage

    $ python exp_p1_p2_p3_figures.py -name Karate_temporal -k 2 -dose_response linear

    $ python exp_p1_p2_p3_figures.py -name UIHC_HCP_patient_room_withinHCPxPx_2011_sampled -k 6 -dose_response linear

Saved in 

    `../plots/exp_p1_p2_p3_figures/`
"""

import argparse
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_box(df, title, ylabel, outpath, filename):
    ax = df.plot.box(title=title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(outpath+filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='greedy source detection, missing infection')
    parser.add_argument('-name', '--name', type=str, default="Karate_temporal",
            help= 'Karate_temporal | UIHC_HCP_patient_room_withinHCPxPx_2011_sampled')
    parser.add_argument('-k', '--k', type=int, default=2,
            help= 'GT seed count. 2 | 6')
    parser.add_argument('-dose_response', '--dose_response', type=str, default="linear",
            help= 'linear | exponential')
    args = parser.parse_args()

    path = "../tables/exp_p1_p2_p3/"
    if args.dose_response == "linear":
        infile = "linear_{}_k{}.csv".format(args.name, args.k)
    elif args.dose_response == "exponential":
        infile = "{}_k{}.csv".format(args.name, args.k)

    if args.name=="Karate_temporal":
        outname = "Karate"
    elif args.name=="UIHC_HCP_patient_room_withinHCPxPx_2011_sampled":
        outname = "UIHC_S"
    
    df_result = pd.read_csv(path+infile)
    n_exp = df_result.shape[0]

    print("df_result.columns: {}".format(df_result.columns))

    columns_F1 = ['GT_F1', 'BR_F1', 'lazy_G1_F1', 'lazy_ISCK_F1', 'lazy_ISCK_MU_F1', 'GR_F1', 'GR_ghit50_F1']
    columns_MCC = ['GT_MCC', 'BR_MCC', 'lazy_G1_MCC', 'lazy_ISCK_MCC', 'lazy_ISCK_MU_MCC', 'GR_MCC', 'GR_ghit50_MCC']
    columns_TP = ['GT_TP', 'BR_TP', 'lazy_G1_TP', 'lazy_ISCK_TP', 'lazy_ISCK_MU_TP', 'GR_TP', 'GR_ghit50_TP']

    columns = ['GT', 'BR', 'lazy_G1', 'lazy_ISCK', 'lazy_ISCK_MU', 'GR', 'GR_ghit50']

    outpath = "../plots/exp_p1_p2_p3_figures/"

    df_F1 = df_result[columns_F1]
    df_F1 = df_F1.rename(
            columns=dict([(col_prev, col) for col_prev, col in zip(columns_F1, columns)])
            )

    df_MCC = df_result[columns_MCC]
    df_MCC = df_MCC.rename(
            columns=dict([(col_prev, col) for col_prev, col in zip(columns_MCC, columns)])
            )

    df_TP = df_result[columns_TP]
    df_TP = df_TP.rename(
            columns=dict([(col_prev, col) for col_prev, col in zip(columns_TP, columns)])
            )

    # F1
    filename = "exp_p1_p2_p3_{}_k{}_{}_F1.png".format(outname, args.k, args.dose_response)
    title = "Comparison of algorithms (avg F1 score of 100 simulation runs)\nGT seed count: {}, {} dose response ({} rep)".format(args.k, args.dose_response, n_exp)
    ylabel = "F1 score"
    plot_box(df_F1, title, ylabel, outpath, filename)

    # MCC
    filename = "exp_p1_p2_p3_{}_k{}_{}_MCC.png".format(outname, args.k, args.dose_response)
    title = "Comparison of algorithms (avg MCC score of 100 simulation runs)\nGT seed count: {}, {} dose response ({} rep)".format(args.k, args.dose_response, n_exp)
    ylabel = "MCC score"
    plot_box(df_MCC, title, ylabel, outpath, filename)

    # TP
    filename = "exp_p1_p2_p3_{}_k{}_{}_TP.png".format(outname, args.k, args.dose_response)
    title = "Comparison of algorithms (avg TP score of 100 simulation runs)\nGT seed count: {}, {} dose response ({} rep)".format(args.k, args.dose_response, n_exp)
    ylabel = "TP score"
    plot_box(df_TP, title, ylabel, outpath, filename)

