"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads df_result and plots figures

Usage

Carilion (EXPECTED) dose-response: exponential, seeds_per_t: 3

$ python final_exp_result_figures_for_one_setting.py -name G_Carilion -dose_response exponential -seeds_per_t 3 -type expectedT

UIHC (EXPECTED) dose-response: exponential, seeds_per_t: 1, 3

$ python final_exp_result_figures_for_one_setting.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -dose_response exponential -seeds_per_t 1 -type expectedT
$ python final_exp_result_figures_for_one_setting.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -dose_response exponential -seeds_per_t 3 -type expectedT

Karate (ORIGINAL) dose-response: linear, exponential, seeds_per_t: 1, 3

$ python final_exp_result_figures_for_one_setting.py -name Karate_temporal -dose_response exponential -seeds_per_t 1 -type original
$ python final_exp_result_figures_for_one_setting.py -name Karate_temporal -dose_response exponential -seeds_per_t 3 -type original
$ python final_exp_result_figures_for_one_setting.py -name Karate_temporal -dose_response linear -seeds_per_t 1 -type original
$ python final_exp_result_figures_for_one_setting.py -name Karate_temporal -dose_response linear -seeds_per_t 3 -type original

UIHC_S (ORIGINAL) dose-response: linear, exponential, seeds_per_t: 1, 3

$ python final_exp_result_figures_for_one_setting.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T -dose_response exponential -seeds_per_t 1 -type original
$ python final_exp_result_figures_for_one_setting.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T -dose_response exponential -seeds_per_t 3 -type original
$ python final_exp_result_figures_for_one_setting.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T -dose_response linear -seeds_per_t 1 -type original
$ python final_exp_result_figures_for_one_setting.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T -dose_response linear -seeds_per_t 3 -type original

tables are saved in the folder where pickled objects are at
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
            "cult": "CuLT",
            "netsleuth": "NetSleuth",
            "reachability": "PathFinder",
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
    # mapping_prev = {
            # "GT": "Ground Truth",
            # "Random(30rep)": "Random",
            # "cult": "Cult",
            # "reachability": "Reachability",
            # "LOS": "LOS",
            # "P1_E_LazyGreedy": "Greedy",
            # "P2_E_LazyISCK_greedy": "ISCK-Greedy",
            # "P2_E_LazyISCK_multiplicative_update": "ISCK-M.Update",
            # "P3_E_GreedyRatio": "GreedyRatio",
            # "P3_E_GreedyRatio_ghit>50": "GreedyRatio-hit50",
            # "linear_P1_LazyGreedy": "Greedy",
            # "linear_P2_LazyISCK_greedy": "ISCK-Greedy",
            # "linear_P2_LazyISCK_multiplicative_update": "ISCK-M.Update",
            # "linear_P3_GreedyRatio": "GreedyRatio",
            # "linear_P3_GreedyRatio_ghit>50": "GreedyRatio-hit50",
            # "P1_LazyGreedy": "Greedy",
            # "P2_LazyISCK_greedy": "ISCK-Greedy",
            # "P2_LazyISCK_multiplicative_update": "ISCK-M.Update",
            # "P3_GreedyRatio": "GreedyRatio",
            # "P3_GreedyRatio_ghit>50": "GreedyRatio-hit50",
            # }
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
    path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

    if args.dose_response=="exponential":
        infile = "result_concat.csv"
    elif args.dose_response=="linear":
        infile = "linear_result_concat.csv"
    df_result = pd.read_csv(path + infile)

    df_result = df_result.fillna(0)

    # Map the algorithm names
    # algorithm_name_mapping = get_algorithm_name_mapping()
    # algorithm_array = df_result["Algorithm"].values
    # algorithm_name_mapped_array = [algorithm_name_mapping[algo] for algo in algorithm_array]
    # df_result["Algorithm"] = algorithm_name_mapped_array

    GT_F1 = df_result[df_result["Algorithm"] == "GT"]["F1"].values[0]
    GT_MCC = df_result[df_result["Algorithm"] == "GT"]["MCC"].values[0]
    GT_TP = df_result[df_result["Algorithm"] == "GT"]["TP"].values[0]

    Random_F1 = df_result[df_result["Algorithm"] == "Random(30rep)"]["F1"].values[0]
    Random_MCC = df_result[df_result["Algorithm"] == "Random(30rep)"]["MCC"].values[0]
    Random_TP = df_result[df_result["Algorithm"] == "Random(30rep)"]["TP"].values[0]

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
    # NOTE: Feb 9 - remove Problem1
    algorithm_array_bar = ["cult", "netsleuth", "reachability", "LOS", P2_algorithm_to_plot, P3_algorithm_to_plot]
    algorithm_name_mapping = get_algorithm_name_mapping()
    label_list = [algorithm_name_mapping[algo] for algo in algorithm_array_bar]

    df_result_bar = df_result[df_result.Algorithm.isin(algorithm_array_bar)].copy()
    print(df_result_bar[["Algorithm", "TP", "TN", "FP", "FN", "F1", "MCC"]])

    algorithm_array_bar = df_result_bar["Algorithm"].values
    algorithm_name_mapped_array_bar = [algorithm_name_mapping[algo] for algo in algorithm_array_bar]
    df_result_bar["Algorithm"] = algorithm_name_mapped_array_bar
    print(df_result_bar[["Algorithm", "TP", "TN", "FP", "FN", "F1", "MCC"]])

    outpath = "../plots/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

    # algorithm_array = df_result["Algorithm"].values

    graph_name_mapping = get_graph_name_mapping()
    name = graph_name_mapping[graph_name]

    # TP
    outfile = "{}_TP_{}.png".format(name, args.dose_response)
    title = "{}. Metric: true positive".format(name)
    # plot_bar(df_result["TP"].values, label_list, title, outpath, outfile)
    plot_bar_v2(GT_TP, Random_TP, df_result_bar["TP"].values, df_result_bar["Algorithm"].values, title, outpath, outfile)
    
    # MCC
    outfile = "{}_MCC_{}.png".format(name, args.dose_response)
    title = "{}. Metric: MCC".format(name)
    # plot_bar(df_result["MCC"].values, label_list, title, outpath, outfile)
    plot_bar_v2(GT_MCC, Random_MCC, df_result_bar["MCC"].values, df_result_bar["Algorithm"].values, title, outpath, outfile)
    
