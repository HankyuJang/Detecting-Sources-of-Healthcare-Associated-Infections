"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last modified: Mar 2021

Description: This script generates bar plots

Usage:
    $ python barplots.py --name Karate_temporal --dose_response linear
    $ python barplots.py --name Karate_temporal --dose_response exponential -GT_quality median
    $ python barplots.py --name Karate_temporal --dose_response exponential -GT_quality best

    $ python barplots.py --name UIHC_HCP_patient_room --year 2011 --sampled True --dose_response linear
    $ python barplots.py --name UIHC_HCP_patient_room --year 2011 --sampled True --dose_response exponential -GT_quality median
    $ python barplots.py --name UIHC_HCP_patient_room --year 2011 --sampled True --dose_response exponential -GT_quality best

    $ python barplots.py --name UIHC_HCP_patient_room --year 2011 --dose_response linear
    $ python barplots.py --name UIHC_HCP_patient_room --year 2011 --dose_response exponential -GT_quality median
    $ python barplots.py --name UIHC_HCP_patient_room --year 2011 --dose_response exponential -GT_quality best
"""

import argparse
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def read_dataframes():
    df_GT = pd.read_csv("../tables/{}/{}/{}/{}/GT.csv".format(folder, name, dose_response, GT_quality))
    df_B_random = pd.read_csv("../tables/{}/{}/{}/{}/B_random.csv".format(folder, name, dose_response, GT_quality))
    df_B_degree = pd.read_csv("../tables/{}/{}/{}/{}/B_degree.csv".format(folder, name, dose_response, GT_quality))
    df_exp1 = pd.read_csv("../tables/{}/{}/{}/{}/exp1.csv".format(folder, name, dose_response, GT_quality))
    df_exp2 = pd.read_csv("../tables/{}/{}/{}/{}/exp2.csv".format(folder, name, dose_response, GT_quality))
    df_ISCK = pd.read_csv("../tables/{}/{}/{}/{}/ISCK.csv".format(folder, name, dose_response, GT_quality))
    df_ISCK_diff = pd.read_csv("../tables/{}/{}/{}/{}/ISCK_diff.csv".format(folder, name, dose_response, GT_quality))
    return df_GT, df_B_random, df_B_degree, df_exp1, df_exp2, df_ISCK, df_ISCK_diff

def barchart(k, em, em_str, label_list, y, y_lim, title, dose_response, GT_quality):
    width, gap = 0.9, 0.5

    # x_GT = np.arange(1)
    x_B_random = np.arange(1) + 0*(gap + 1)
    x_B_degree = np.arange(1) + 1*(gap + 1)
    x_Greedy1 = np.arange(1) + 2*(gap + 1)
    x_Greedy2 = np.arange(1) + 3*(gap + 1)
    x_ISCK = np.arange(1) + 4*(gap + 1)
    x_ISCK_diff = np.arange(1) + 5*(gap + 1)
    # ind = np.concatenate((x_B_random, x_B_degree, x_Greedy1, x_Greedy2, x_ISCK))
    ind = np.concatenate((x_B_random, x_B_degree, x_Greedy1, x_Greedy2, x_ISCK, x_ISCK_diff))

    fig, ax = plt.subplots()
    fig.suptitle(title)
    # rects_GT = ax.bar(x_GT, y[0], width, color="tab:gray", label="GT")
    rects_B_random = ax.bar(ind[0], y[1], width, color="tab:olive", label=label_list[1])
    rects_B_degree = ax.bar(ind[1], y[2], width, color="tab:pink", label=label_list[2])
    rects_Greedy1 = ax.bar(ind[2], y[3], width, color="tab:cyan", label=label_list[3])
    rects_Greedy2 = ax.bar(ind[3], y[4], width, color="tab:green", label=label_list[4])
    rects_ISCK = ax.bar(ind[4], y[5], width, color="tab:blue", label=label_list[5])
    rects_ISCK_diff = ax.bar(ind[5], y[6], width, color="tab:red", label=label_list[6])
    ax.set_ylabel("{}".format(em_str))
    ax.axhline(y=y[0], linestyle="dashed", color="black")
    ax.legend(loc="upper left")
    ax.set_ylim(y_lim)
    ax.set_xticks(ind)
    ax.set_xticklabels(["" for lab in label_list[1:]], rotation = 45)
    # plt.tight_layout()
    # plt.title(em_str)
    plt.savefig("../plots/bar/{}/{}/{}/k{}_{}.png".format(name, dose_response, GT_quality, k, em), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bar plot')
    parser.add_argument('-name', '--name', type=str, default="Karate_temporal",
                        help= 'network to use. Karate_temporal | UIHC_Jan2010_patient_room_temporal | UIHC_HCP_patient_room')
    parser.add_argument('-year', '--year', type=int, default=2011,
                        help= '2007 | 2011')
    parser.add_argument('-sampled', '--sampled', type=bool, default=False,
                        help= 'set it True to use sampled data.')
    parser.add_argument('-dose_response', '--dose_response', type=str, default="exponential",
                        help= 'dose-response function. linear | exponential')
    parser.add_argument('-GT_quality', '--GT_quality', type=str, default="median",
                        help= 'Quality of the ground truth simulation. best | median')
    args = parser.parse_args()
    name = args.name
    year = args.year
    sampled = args.sampled
    dose_response = args.dose_response
    GT_quality = args.GT_quality

    if name == "UIHC_HCP_patient_room":
        if sampled == True:
            name = "{}_{}_sampled".format(name, year)
        else:
            name = "{}_{}".format(name, year)

    folder = "ISCK_temporal"
    df_GT, df_B_random, df_B_degree, df_Greedy1, df_Greedy2, df_ISCK, df_ISCK_diff = read_dataframes()

    # Fig1. MCC
    label_list = ["Ground truth", "Random", "Degree", "Greedy1", "Greedy2", "ISCK", "ISCK_diff"]
    evaluation_metric_list = ["MCC", "F1", "precision", "recall"]
    for evaluation_metric in evaluation_metric_list:
        for idx in range(3):
            # idx=0
            k=df_GT.loc[idx]['k']
            len_P=df_GT.loc[idx]['len(P)']
            len_N=df_GT.loc[idx]['len(N)']
            em = "E_{}".format(evaluation_metric) #em: evaluation metric
            em_str = "Expected {} score".format(evaluation_metric)
            value_list = [df_GT.loc[idx,em], df_B_random.loc[idx,em], df_B_degree.loc[idx,em], df_Greedy1.loc[idx,em], df_Greedy2.loc[idx,em], df_ISCK.loc[idx,em], df_ISCK_diff.loc[idx,em]]
            title = "{}. Karate. Num seeds: {}\n|P|: {}, |N|: {}".format(em_str, k, len_P, len_N)
            y_lim = (0, 1)
            barchart(k, em, em_str, label_list, value_list, y_lim, title, dose_response, GT_quality)
