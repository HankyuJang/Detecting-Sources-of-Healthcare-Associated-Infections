"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads df_result and plots figures

Usage

UIHC_S

$ python result_figure_k246810.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T -dose_response exponential

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
            "netsleuth": "Netsleuth",
            "reachability": "Reachability",
            "LOS": "LOS",
            "P1_E_LazyGreedy": "CardinalitySD",
            "P2_E_LazyISCK_greedy": "KnapSackSD",
            "P2_E_LazyISCK_multiplicative_update": "KnapSackSD",
            "P3_E_GreedyRatio": "RatioSD",
            "P3_E_GreedyRatio_ghit>50": "RatioSD",
            "linear_P1_LazyGreedy": "CardinalitySD",
            "linear_P2_LazyISCK_greedy": "KnapSackSD",
            "linear_P2_LazyISCK_multiplicative_update": "KnapSackSD",
            "linear_P3_GreedyRatio": "RatioSD",
            "linear_P3_GreedyRatio_ghit>50": "RatioSD",
            "P1_LazyGreedy": "CardinalitySD",
            "P2_LazyISCK_greedy": "KnapSackSD",
            "P2_LazyISCK_multiplicative_update": "KnapSackSD",
            "P3_GreedyRatio": "RatioSD",
            "P3_GreedyRatio_ghit>50": "RatioSD",
            }
    return mapping

def get_graph_name_mapping():
    mapping = {
            "Karate_temporal": "Karate",
            "G_Carilion": "Carilion",
            "G_UVA": "UVA-PreCovid-Emergency",
            "G_UVA_v2": "UVA-PreCovid",
            "G_UVA_v3": "UVA-PostCovid-Emergency",
            "G_UVA_v4": "UVA-PostCovid",
            "UIHC_HCP_patient_room_withinHCPxPx_2011": "UIHC",
            "UIHC_HCP_patient_room_withinHCPxPx_2011_sampled": "UIHC-Unit",
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

# line for GT, Baseline random
# Select one for each P1, P2, P3
def plot_bar_v2(GT_score, Random_score, score_array, label_list, title, outpath, outfile):
    x_array = np.arange(len(label_list))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = '20'
    
    for x, label, score in zip(x_array, label_list, score_array):
        ax.bar(x=x, height=score, label=label)

    ax.axhline(y=GT_score, linestyle='-', color="black")
    ax.axhline(y=Random_score, linestyle=':', color="black")

    ax.set_ylabel("Score")
    ax.set_xticks(x_array)
    ax.set_xticklabels(label_list, rotation = 45)
    # ax.set_ylim(y_lim)
    # ax.legend()
    plt.title(title, fontsize=25)
    plt.tight_layout()
    plt.savefig(outpath+outfile, dpi=300)
    plt.close()

# ref: https://stackoverflow.com/a/35710894
# For exp k246810
def plot_bar_v3(score_array, label_list, x_label_list, color_list, title, outpath, outfile):
    x_array = np.arange(len(label_list))
    fig, ax = plt.subplots(figsize=(8, 3))
    # plt.rcParams['font.size'] = '20'
    
    for idx, (x, label, score, color) in enumerate(zip(x_array, label_list, score_array, color_list)):
        if idx <= 3:
            ax.bar(x=x, height=score, label=label, color=color)
        else:
            ax.bar(x=x, height=score, label='_nolegend_', color=color)

    ax.set_ylabel("Score")
    ax.set_xticks(x_array)
    ax.set_xticklabels(x_label_list)
    # ax.set_xticklabels(label_list, rotation = 45)
    # ax.set_xticklabels(["" for label in label_list], rotation = 45)
    # ax.set_ylim(y_lim)
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath+outfile, dpi=300)
    plt.close()

def plot_bar_v4(score_array1, score_array2, label_list, x_label_list, color_list, title1, title2, outpath, outfile):
    x_array = np.arange(len(label_list))

    fig = plt.figure(figsize=(8, 3), dpi=300)

    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    (ax1, ax2) = gs.subplots(sharex='col', sharey='row')

    # plt.rcParams['font.size'] = '20'
    
    for idx, (x, label, score, color) in enumerate(zip(x_array, label_list, score_array1, color_list)):
        if idx <= 3:
            ax1.bar(x=x, height=score, label=label, color=color)
        else:
            ax1.bar(x=x, height=score, label='_nolegend_', color=color)

    ax1.set_title(title1, fontsize=20)
    ax1.set_ylabel("Score", fontsize=16)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x_array)
    ax1.set_xticklabels(x_label_list, fontsize=12)
    ax1.legend(fontsize=12)
    #FIG2#################################################
    for idx, (x, label, score, color) in enumerate(zip(x_array, label_list, score_array2, color_list)):
        if idx <= 3:
            ax2.bar(x=x, height=score, label=label, color=color)
        else:
            ax2.bar(x=x, height=score, label='_nolegend_', color=color)

    # ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.set_title(title2, fontsize=20)
    ax2.set_xticks(x_array)
    ax2.set_xticklabels(x_label_list, fontsize=12)

    #plt#################################################
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
    parser.add_argument('-GT_quality', '--GT_quality', type=str, default="best",
                        help= 'Quality of the ground truth simulation. best | median. Always use best')
    parser.add_argument('-seeds_per_t', '--seeds_per_t', type=int, default=1,
                        help= 'number of seeds per timestep')
    parser.add_argument('-n_t_seeds', '--n_t_seeds', type=int, default=2,
                        help= 'number of timesteps for seeds')
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    parser.add_argument('-type', '--type', type=str, default="expectedT",
                        help= 'set of experiments to summarize. expectedT | original | all')
    args = parser.parse_args()

    seeds_per_t = args.seeds_per_t
    n_t_seeds = args.n_t_seeds
    n_t_for_eval = args.n_t_for_eval
    graph_name = get_graph_name(args)

    np.set_printoptions(suppress=True)

    # -------------------------------------------
    path = "../tables/exp_k246810/"

    exp_k246810_algorithm_list = []
    exp_k246810_MCC_list = []
    exp_k246810_F1_list = []
    color_list = ["black", "#9467bd", "#8c564b", "white"] * 5
    x_label_list = \
        ["", r"|S$^{+}$|=2", "", "",\
         "", r"|S$^{+}$|=4", "", "",\
         "", r"|S$^{+}$|=6", "", "",\
         "", r"|S$^{+}$|=8", "", "",\
         "", r"|S$^{+}$|=10", "", ""]

    for k in [2, 4, 6, 8, 10]:
        infile = "result_k{}.csv".format(k)

        df_result = pd.read_csv(path + infile)

        df_result = df_result.fillna(0)

        GT_F1 = df_result[df_result["Algorithm"] == "GT"]["F1"].values[0]
        GT_MCC = df_result[df_result["Algorithm"] == "GT"]["MCC"].values[0]
        GT_TP = df_result[df_result["Algorithm"] == "GT"]["TP"].values[0]

        # Random_F1 = df_result[df_result["Algorithm"] == "Random(30rep)"]["F1"].values[0]
        # Random_MCC = df_result[df_result["Algorithm"] == "Random(30rep)"]["MCC"].values[0]
        # Random_TP = df_result[df_result["Algorithm"] == "Random(30rep)"]["TP"].values[0]

        P1_LazyGreedy_list = ["P1_E_LazyGreedy", "linear_P1_LazyGreedy", "P1_LazyGreedy"]
        P1_LazyGreedy = df_result[df_result["Algorithm"].isin(P1_LazyGreedy_list)]["Algorithm"].values[0]

        P2_LazyISCK_greedy_list = ["P2_LazyISCK_greedy", "linear_P2_LazyISCK_greedy", "P2_E_LazyISCK_greedy"]
        P2_LazyISCK_greedy_MCC = df_result[df_result["Algorithm"].isin(P2_LazyISCK_greedy_list)]["MCC"].values[0]
        P2_LazyISCK_greedy = df_result[df_result["Algorithm"].isin(P2_LazyISCK_greedy_list)]["Algorithm"].values[0]

        P2_LazyISCK_multiplicative_update_list = ["P2_LazyISCK_multiplicative_update", "linear_P2_LazyISCK_multiplicative_update", "P2_E_LazyISCK_multiplicative_update"]
        P2_LazyISCK_multiplicative_update_MCC = df_result[df_result["Algorithm"].isin(P2_LazyISCK_multiplicative_update_list)]["MCC"].values[0]
        P2_LazyISCK_multiplicative_update = df_result[df_result["Algorithm"].isin(P2_LazyISCK_multiplicative_update_list)]["Algorithm"].values[0]

        if P2_LazyISCK_greedy_MCC > P2_LazyISCK_multiplicative_update_MCC:
            P2_algorithm_to_plot = P2_LazyISCK_greedy
        else:
            P2_algorithm_to_plot = P2_LazyISCK_multiplicative_update

        P3_GreedyRatio_list = ["P3_GreedyRatio", "linear_P3_GreedyRatio", "P3_E_GreedyRatio"]
        P3_GreedyRatio_MCC = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_list)]["MCC"].values[0]
        P3_GreedyRatio = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_list)]["Algorithm"].values[0]

        P3_GreedyRatio_ghit50_list = ["P3_GreedyRatio_ghit>50", "linear_P3_GreedyRatio_ghit>50", "P3_E_GreedyRatio_ghit>50"]
        P3_GreedyRatio_ghit50_MCC = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_ghit50_list)]["MCC"].values[0]
        P3_GreedyRatio_ghit50 = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_ghit50_list)]["Algorithm"].values[0]

        if P3_GreedyRatio_MCC > P3_GreedyRatio_ghit50_MCC:
            P3_algorithm_to_plot = P3_GreedyRatio
        else:
            P3_algorithm_to_plot = P3_GreedyRatio_ghit50

        # Prep
        # algorithm_array_bar = ["cult", "netsleuth", "reachability", "LOS", P1_LazyGreedy, P2_algorithm_to_plot, P3_algorithm_to_plot]
        # NOTE: Feb 10 - only include GT, P2, P3
        algorithm_array_bar = ["GT", P2_algorithm_to_plot, P3_algorithm_to_plot]
        algorithm_name_mapping = get_algorithm_name_mapping()
        label_list = [algorithm_name_mapping[algo] for algo in algorithm_array_bar]

        df_result_bar = df_result[df_result.Algorithm.isin(algorithm_array_bar)].copy()
        # print(df_result_bar[["Algorithm", "TP", "TN", "FP", "FN", "F1", "MCC"]])

        algorithm_array_bar = df_result_bar["Algorithm"].values
        algorithm_name_mapped_array_bar = [algorithm_name_mapping[algo] for algo in algorithm_array_bar]
        df_result_bar["Algorithm"] = algorithm_name_mapped_array_bar
        print(df_result_bar[["Algorithm", "TP", "TN", "FP", "FN", "F1", "MCC"]])

        exp_k246810_algorithm_list.extend(list(df_result_bar["Algorithm"].values))
        exp_k246810_MCC_list.extend(list(df_result_bar["MCC"].values))
        exp_k246810_F1_list.extend(list(df_result_bar["F1"].values))

        exp_k246810_algorithm_list.append("")
        exp_k246810_MCC_list.append(0)
        exp_k246810_F1_list.append(0)

    # print("exp_k246810_algorithm_list: {}".format(exp_k246810_algorithm_list))
    # print("exp_k246810_MCC_list: {}".format(exp_k246810_MCC_list))
    # print("exp_k246810_F1_list: {}".format(exp_k246810_F1_list))

    graph_name_mapping = get_graph_name_mapping()
    name = graph_name_mapping[graph_name]

    outpath = "../plots/exp_k246810/"
    outfile = "exp_k246810_MCC.png"
    title = "{}. Metric: MCC".format(name)

    outfile = "exp_k246810_F1.png"

    # algorithm_array = df_result["Algorithm"].values

    graph_name_mapping = get_graph_name_mapping()
    name = graph_name_mapping[graph_name]
    
    # MCC
    outfile = "{}_MCC_{}.png".format(name, args.dose_response)
    title = "{}. Metric: MCC".format(name)
    plot_bar_v3(exp_k246810_MCC_list, exp_k246810_algorithm_list, x_label_list, color_list, title, outpath, outfile)

    # F1
    outfile = "{}_F1_{}.png".format(name, args.dose_response)
    title = "{}. Metric: F1".format(name)
    plot_bar_v3(exp_k246810_F1_list, exp_k246810_algorithm_list, x_label_list, color_list, title, outpath, outfile)

    # F1, MCC side by side
    outfile = "{}_F1_MCC_{}.png".format(name, args.dose_response)
    title1 = "{}. Metric: F1".format(name)
    title2 = "{}. Metric: MCC".format(name)
    plot_bar_v4(exp_k246810_F1_list, exp_k246810_MCC_list, exp_k246810_algorithm_list, x_label_list, color_list, title1, title2, outpath, outfile)


