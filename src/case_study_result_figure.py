"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads df_result and plots figures

Usage

$ python case_study_result_figure.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T

"""

import argparse

from utils.load_network import *
import pandas as pd
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def get_algorithm_name_mapping():
    mapping = {
            "GT": "Ground Truth",
            "Random(30rep)": "Random",
            "cult": "Cult",
            "P1_E_LazyGreedy": "Greedy",
            "P2_E_LazyISCK_greedy": "ISCK-Greedy",
            "P2_E_LazyISCK_multiplicative_update": "ISCK-M.Update",
            "P3_E_GreedyRatio": "GreedyRatio",
            "P3_E_GreedyRatio_ghit>50": "GreedyRatio-hit50",
            "linear_P1_LazyGreedy": "Greedy",
            "linear_P2_LazyISCK_greedy": "ISCK-Greedy",
            "linear_P2_LazyISCK_multiplicative_update": "ISCK-M.Update",
            "linear_P3_GreedyRatio": "GreedyRatio",
            "linear_P3_GreedyRatio_ghit>50": "GreedyRatio-hit50",
            "P1_LazyGreedy": "Greedy",
            "P2_LazyISCK_greedy": "ISCK-Greedy",
            "P2_LazyISCK_multiplicative_update": "ISCK-M.Update",
            "P3_GreedyRatio": "GreedyRatio",
            "P3_GreedyRatio_ghit>50": "GreedyRatio-hit50",
            }
    return mapping

def get_graph_name_mapping():
    mapping = {
            "Karate_temporal": "Karate",
            "G_Carilion": "Carilion",
            "UIHC_HCP_patient_room_withinHCPxPx_2011": "Hospital1",
            "UIHC_HCP_patient_room_withinHCPxPx_2011_sampled": "Hospital1-sub",
            }
    return mapping

def plot_bar(score_array, label_list, title, outpath, outfile):
    x_array = np.arange(len(label_list))
    fig, ax = plt.subplots()
    
    for x, label, score in zip(x_array, label_list, score_array):
        ax.bar(x=x, height=score, label=label)

    ax.set_ylabel("Score")
    ax.set_xticks(x_array)
    ax.set_xticklabels(label_list, rotation = 45)
    # ax.set_ylim(y_lim)
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath+outfile, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='greedy source detection, missing infection')
    parser.add_argument('-name', '--name', type=str, default="Karate_temporal",
                        help= 'network to use. Karate_temporal | UIHC_Jan2010_patient_room_temporal | UIHC_HCP_patient_room_withinHCPxPx | UVA_temporal')
    parser.add_argument('-year', '--year', type=int, default=2011,
                        help= '2007 | 2011')
    parser.add_argument('-sampled', '--sampled', type=bool, default=False,
                        help= 'set it True to use sampled data.')
    parser.add_argument('-dose_response', '--dose_response', type=str, default="exponential",
                        help= 'dose-response function')
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    args = parser.parse_args()

    graph_name = get_graph_name(args)

    np.set_printoptions(suppress=True)

    # -------------------------------------------
    path = "../tables/case_study/{}/".format(graph_name)

    if args.dose_response=="exponential":
        infile = "result_concat.csv"
    elif args.dose_response=="linear":
        infile = "linear_result_concat.csv"
    df_result = pd.read_csv(path + infile)

    outpath = "../plots/case_study/{}/".format(graph_name)

    algorithm_array = df_result["Algorithm"].values
    algorithm_name_mapping = get_algorithm_name_mapping()
    label_list = [algorithm_name_mapping[algo] for algo in algorithm_array]

    graph_name_mapping = get_graph_name_mapping()
    name = graph_name_mapping[graph_name]
    
    # MCC
    outfile = "{}_MCC_{}.png".format(name, args.dose_response)
    title = "Source detection on {}. Metric: MCC".format(name)
    plot_bar(df_result["MCC"].values, label_list, title, outpath, outfile)
    
    # F1
    outfile = "{}_F1_{}.png".format(name, args.dose_response)
    title = "Source detection on {}. Metric: F1".format(name)
    plot_bar(df_result["F1"].values, label_list, title, outpath, outfile)

    # TP
    outfile = "{}_TP_{}.png".format(name, args.dose_response)
    title = "Source detection on {}. Metric: true positive".format(name)
    plot_bar(df_result["TP"].values, label_list, title, outpath, outfile)

    # hops
    outfile = "{}_hops_{}.png".format(name, args.dose_response)
    title = "Source detection on {}.\nMetric: average distance to ground truth seeds".format(name)
    plot_bar(df_result["hops"].values, label_list, title, outpath, outfile)


