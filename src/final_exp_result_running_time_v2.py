"""
Author: -
Email: -
Last Modified: Feb 2022

Description: 

This script loads pickle objects and prepare result tables for runtime analysis.

Do plotting in the script as well.

Usage


UIHC_S (ORIGINAL) dose-response: exponential, seeds_per_t: 3

$ python final_exp_result_running_time_v2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T -dose_response exponential -seeds_per_t 3

tables are saved in the folder where pickled objects are at
"""

import argparse
from tqdm import tqdm

from utils.load_network import *
from prep_result_dataframes import *
import pandas as pd
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import math
import os

def plot_bar(time_array, label_list, color_list, ylabel, title, outpath, outfile):
    x_array = np.arange(len(label_list))
    empty_list = ['' for x in x_array]
    fig, ax = plt.subplots()
    
    for x, label, color, time in zip(x_array, label_list, color_list, time_array):
        ax.bar(x=x, height=time, label=label, color=color)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x_array)
    # ax.set_xticklabels(label_list, rotation = 45)
    ax.set_xticklabels(empty_list)
    # ax.set_ylim(y_lim)
    ax.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath+outfile, dpi=300)
    plt.close()

# ref: https://stackoverflow.com/a/35710894
# For exp k246810
# Run time comparison
def plot_bar_v3(score_array, label_list, x_label_list, color_list, title,  ylabel,outpath, outfile):
    x_array = np.arange(len(label_list))
    fig, ax = plt.subplots(figsize=(8, 3))
    # plt.rcParams['font.size'] = '20'
    
    for idx, (x, label, score, color) in enumerate(zip(x_array, label_list, score_array, color_list)):
        if idx <= 3:
            ax.bar(x=x, height=score, label=label, color=color)
        else:
            ax.bar(x=x, height=score, label='_nolegend_', color=color)

    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(x_array)
    ax.set_xticklabels(x_label_list, fontsize=16)
    # ax.set_xticklabels(label_list, rotation = 45)
    # ax.set_xticklabels(["" for label in label_list], rotation = 45)
    # ax.set_ylim(y_lim)
    ax.legend(loc="upper center", fontsize=16)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outpath+outfile, dpi=300)
    plt.close()

# Run time comparison & evaluation score comparison
def plot_bar_v4(score_array1, score_array2, label_list, x_label_list, color_list, title1, title2, ylabel1, ylabel2, outpath, outfile):
    x_array = np.arange(len(label_list))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    # plt.rcParams['font.size'] = '20'
    
    for idx, (x, label, score, color) in enumerate(zip(x_array, label_list, score_array1, color_list)):
        if idx <= 3:
            ax1.bar(x=x, height=score, label=label, color=color)
        else:
            ax1.bar(x=x, height=score, label='_nolegend_', color=color)
    ax1.set_title(title1, fontsize=20)
    ax1.set_ylabel(ylabel1, fontsize=16)
    ax1.set_xticks(x_array)
    ax1.set_xticklabels(x_label_list, fontsize=16)
    ax1.legend(loc="best", fontsize=16)

    for idx, (x, label, score, color) in enumerate(zip(x_array, label_list, score_array2, color_list)):
        if idx <= 3:
            ax2.bar(x=x, height=score, label=label, color=color)
        else:
            ax2.bar(x=x, height=score, label='_nolegend_', color=color)
    ax2.set_title(title2, fontsize=20)
    ax2.set_ylabel(ylabel2, fontsize=16)
    ax2.set_xticks(x_array)
    ax2.set_xticklabels(x_label_list, fontsize=16)
    # ax2.legend(loc="best", fontsize=16)

    # plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outpath+outfile, dpi=300)
    plt.close()

def load_Px_results_for_runtime(Px_evaluation_dict):

    column_list=['S_detected', 'S_timesteps', 'n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']

    # For P1
    if "df_greedy" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_greedy"]
        df_algo_results = df_algo_results.reset_index(drop=True, inplace=False)
        # print(df_algo_results)
        i_max_k = np.argmax(df_algo_results["k"].values)
        df_temp = df_algo_results.iloc[[i_max_k]][column_list]
        # print(df_temp)
    # For P2
    elif "df_ISCK_results" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_ISCK_results"]
        df_algo_results = df_algo_results.reset_index(drop=True, inplace=False)
        # print(df_algo_results)
        i = 0
        df_temp = df_algo_results.iloc[[i]][column_list]
        # print(df_temp)
    # For P3
    elif "df_greedy_ratio" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_greedy_ratio"]
        df_algo_results = df_algo_results.reset_index(drop=True, inplace=False)
        # print(df_algo_results)
        i = 0
        df_temp = df_algo_results.iloc[[i]][column_list]
        # print(df_temp)

    return df_temp

def load_Px_results(Px_evaluation_dict):

    column_list=['S_detected', 'S_timesteps', 'n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']

    # For P1
    if "df_greedy" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_greedy"]
    # For P2
    elif "df_ISCK_results" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_ISCK_results"]
    # For P3
    elif "df_greedy_ratio" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_greedy_ratio"]

    df_algo_results = df_algo_results.reset_index(drop=True, inplace=False)
    MCC_array = df_algo_results["MCC"].values
    i_best = np.nanargmax(MCC_array)

    return df_algo_results.iloc[[i_best]][column_list]

def get_infile_name_and_name_list(args, graph_name):
    if args.dose_response=="exponential":
        txt_dose_response = ""
    elif args.dose_response=="linear":
        txt_dose_response = "linear_"


    if args.type == "expectedT" and graph_name=="G_Carilion":
        P1_infile_name_list = [
                "{}P1_greedy_evalution_lazyT_expectedT.pickle".format(txt_dose_response)
                ]
        P1_name_list = [
                "{}P1_E_LazyGreedy".format(txt_dose_response),
                ]
        P2_infile_name_list = [
                "{}P2_ISCK_greedy_evalution_lazyT_expectedT.pickle".format(txt_dose_response),
                "{}P2_ISCK_multiplicative_update_evalution_lazyT_expectedT.pickle".format(txt_dose_response)
                ]
        P2_name_list = [
                "{}P2_E_LazyISCK_greedy".format(txt_dose_response),
                "{}P2_E_LazyISCK_multiplicative_update".format(txt_dose_response)
                ]
        P3_infile_name_list = [
                "{}P3_GR_gconstraintF_evalution_lazyF_expectedT.pickle".format(txt_dose_response),
                "{}P3_GR_gconstraintT_evalution_lazyF_expectedT.pickle".format(txt_dose_response),
                ]
        P3_name_list = [
                "{}P3_E_GreedyRatio".format(txt_dose_response),
                "{}P3_E_GreedyRatio_ghit>50".format(txt_dose_response),
                ]
    elif args.type == "expectedT" and graph_name=="UIHC_HCP_patient_room_withinHCPxPx_2011":
        P1_infile_name_list = [
                "{}P1_greedy_evalution_lazyT_expectedT.pickle".format(txt_dose_response)
                ]
        P1_name_list = [
                "{}P1_E_LazyGreedy".format(txt_dose_response),
                ]
        P2_infile_name_list = [
                "{}P2_ISCK_greedy_evalution_lazyT_expectedT.pickle".format(txt_dose_response),
                # "{}P2_ISCK_multiplicative_update_evalution_lazyT_expectedT.pickle".format(txt_dose_response)
                ]
        P2_name_list = [
                "{}P2_E_LazyISCK_greedy".format(txt_dose_response),
                # "{}P2_E_LazyISCK_multiplicative_update".format(txt_dose_response)
                ]
        P3_infile_name_list = [
                "{}P3_GR_gconstraintF_evalution_lazyF_expectedT.pickle".format(txt_dose_response),
                "{}P3_GR_gconstraintT_evalution_lazyF_expectedT.pickle".format(txt_dose_response),
                ]
        P3_name_list = [
                "{}P3_E_GreedyRatio".format(txt_dose_response),
                "{}P3_E_GreedyRatio_ghit>50".format(txt_dose_response),
                ]
    elif args.type == "original" and graph_name=="Karate_temporal":
        P1_infile_name_list = [
                # "{}P1_greedy_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P1_greedy_evalution_lazyT_expectedF.pickle".format(txt_dose_response),
                ]
        P1_name_list = [
                # "{}P1_Greedy".format(txt_dose_response),
                "{}P1_LazyGreedy".format(txt_dose_response),
                ]
        P2_infile_name_list = [
                # "{}P2_ISCK_greedy_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P2_ISCK_greedy_evalution_lazyT_expectedF.pickle".format(txt_dose_response),
                # "{}P2_ISCK_multiplicative_update_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P2_ISCK_multiplicative_update_evalution_lazyT_expectedF.pickle".format(txt_dose_response),
                ]
        P2_name_list = [
                # "{}P2_ISCK_greedy".format(txt_dose_response),
                "{}P2_LazyISCK_greedy".format(txt_dose_response),
                # "{}P2_ISCK_multiplicative_update".format(txt_dose_response),
                "{}P2_LazyISCK_multiplicative_update".format(txt_dose_response),
                ]
        P3_infile_name_list = [
                "{}P3_GR_gconstraintF_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P3_GR_gconstraintT_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                ]
        P3_name_list = [
                "{}P3_GreedyRatio".format(txt_dose_response),
                "{}P3_GreedyRatio_ghit>50".format(txt_dose_response),
                ]
    elif args.type == "original" and graph_name=="UIHC_HCP_patient_room_withinHCPxPx_2011_sampled":
        P1_infile_name_list = [
                # "{}P1_greedy_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P1_greedy_evalution_lazyT_expectedF.pickle".format(txt_dose_response),
                ]
        P1_name_list = [
                # "{}P1_Greedy".format(txt_dose_response),
                "{}P1_LazyGreedy".format(txt_dose_response),
                ]
        P2_infile_name_list = [
                # "{}P2_ISCK_greedy_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P2_ISCK_greedy_evalution_lazyT_expectedF.pickle".format(txt_dose_response),
                # "{}P2_ISCK_multiplicative_update_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P2_ISCK_multiplicative_update_evalution_lazyT_expectedF.pickle".format(txt_dose_response),
                ]
        P2_name_list = [
                # "{}P2_ISCK_greedy".format(txt_dose_response),
                "{}P2_LazyISCK_greedy".format(txt_dose_response),
                # "{}P2_ISCK_multiplicative_update".format(txt_dose_response),
                "{}P2_LazyISCK_multiplicative_update".format(txt_dose_response),
                ]
        P3_infile_name_list = [
                "{}P3_GR_gconstraintF_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                "{}P3_GR_gconstraintT_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
                ]
        P3_name_list = [
                "{}P3_GreedyRatio".format(txt_dose_response),
                "{}P3_GreedyRatio_ghit>50".format(txt_dose_response),
                ]
    elif args.type == "all":
        pass
    return P1_infile_name_list, P1_name_list, P2_infile_name_list, P2_name_list, P3_infile_name_list, P3_name_list

def recompute_n_S_correct(df_result, start_idx_of_algo):
    n_S_correct_list = list(df_result["n_S_correct"].values[:start_idx_of_algo]) # get those till random baseline

    GT_S_detected = eval(df_result.loc["GT"]["S_detected"])
    set_GT_S_detected = set(GT_S_detected)

    S_detected_list_algo = list(df_result["S_detected"].values[start_idx_of_algo:])
    for S_detected in S_detected_list_algo:
        set_S_detected = set(eval(S_detected))
        n_S_correct = len(set_GT_S_detected.intersection(set_S_detected))
        n_S_correct_list.append(n_S_correct)
    df_result["n_S_correct"] = n_S_correct_list

def compute_hop_v1(G_over_time, df_result, node_idx_to_name_mapping, P1_name_list, P2_name_list, P3_name_list):
    # Compute avg hop distance of detected seeds to GT source
    # NOTE: this implementation assumes that GT seeds are at time 0 and time 1
    GT_S = eval(df_result.loc["GT"]["S_detected"])
    GT_S_timesteps = eval(df_result.loc["GT"]["S_timesteps"])

    GT_S_at_t0 = []
    GT_S_at_t1 = []
    for seed, time in zip(GT_S, GT_S_timesteps):
        seed = node_idx_to_name_mapping[seed]
        if time == 0:
            GT_S_at_t0.append(seed)
        elif time == 1:
            GT_S_at_t1.append(seed)

    n_GT_S_at_t0 = len(GT_S_at_t0)
    n_GT_S_at_t1 = len(GT_S_at_t1)

    G0 = G_over_time[0]
    G1 = G_over_time[1]
    P123_name_list = P1_name_list + P2_name_list + P3_name_list

    print("computing hop distances...")
    avg_of_min_hop_list = [0, np.nan] # GT and random.
    for algo_name in P123_name_list:
    # for algo_name in ["linear_P1_Greedy"]:
        algo_S_detected = eval(df_result.loc[algo_name]["S_detected"])
        algo_S_detected = np.unique(np.array(algo_S_detected)) # Remove duplicates (there could be same seed in time 0 and time 1)

        n_algo_S_detected = algo_S_detected.shape[0]

        hop_array_seed_to_GT_seed_at_t0 = np.zeros((n_algo_S_detected, n_GT_S_at_t0))
        hop_array_seed_to_GT_seed_at_t1 = np.zeros((n_algo_S_detected, n_GT_S_at_t1))

        for idx_seed, seed in enumerate(algo_S_detected):
            seed = node_idx_to_name_mapping[seed]
            # Compute min hop at time 0
            for idx_GT_seed, GT_seed in enumerate(GT_S_at_t0):
                try:
                    seed_to_GT_seed_hop = nx.shortest_path_length(G0, source=seed, target=GT_seed)
                except Exception as e:
                    print("{}\n".format(e))
                    seed_to_GT_seed_hop = np.nan
                hop_array_seed_to_GT_seed_at_t0[idx_seed, idx_GT_seed] = seed_to_GT_seed_hop

            # Compute min hop at time 1
            for idx_GT_seed, GT_seed in enumerate(GT_S_at_t1):
                try:
                    seed_to_GT_seed_hop = nx.shortest_path_length(G1, source=seed, target=GT_seed)
                except Exception as e:
                    print("{}\n".format(e))
                    seed_to_GT_seed_hop = np.nan
                hop_array_seed_to_GT_seed_at_t1[idx_seed, idx_GT_seed] = seed_to_GT_seed_hop

        # Now, for each detected seed, get the closest distance to GT seeds. 
        # Then take average of such distances over detected seeds
        hop_array_seed_to_GT_seed = np.concatenate((hop_array_seed_to_GT_seed_at_t0, hop_array_seed_to_GT_seed_at_t1), axis=1)
        print("hop_array_seed_to_GT_seed.shape: {}".format(hop_array_seed_to_GT_seed.shape))

        min_hop_array = np.nanmin(hop_array_seed_to_GT_seed, axis=1)
        print("min_hop_array: {}".format(min_hop_array))

        avg_of_min_hop = np.nanmean(min_hop_array)

        avg_of_min_hop_list.append(avg_of_min_hop)

    print("Complete: computing hop distances...")

    df_result.insert(loc=4, column="hops", value=avg_of_min_hop_list)

def get_S_list_of_tuples(df_result, algo_name):
    S_list_of_tuples = []

    S = eval(df_result.loc[algo_name]["S_detected"])
    S_timesteps = eval(df_result.loc[algo_name]["S_timesteps"])
    for seed, time in zip(S, S_timesteps):
        seed = node_idx_to_name_mapping[seed]
        S_list_of_tuples.append((seed, time))
    print("S_list_of_tuples: {}".format(S_list_of_tuples))
    return S_list_of_tuples

def get_GT_S_list_of_tuples(df_result):
    GT_S_list_of_tuples = []

    GT_S = eval(df_result.loc["GT"]["S_detected"])
    GT_S_timesteps = eval(df_result.loc["GT"]["S_timesteps"])
    for seed, time in zip(GT_S, GT_S_timesteps):
        seed = node_idx_to_name_mapping[seed]
        GT_S_list_of_tuples.append((seed, time))
    print("GT_S_list_of_tuples: {}".format(GT_S_list_of_tuples))
    return GT_S_list_of_tuples

def get_L_B_random_list_of_tuples(L_B_random_S_detected, L_B_random_S_timesteps):
    print("prepare node tuples for baselines")
    n_rep_B_random = len(L_B_random_S_detected)
    L_B_random_list_of_tuples = []

    for B_random_S_detected, B_random_S_timesteps in zip(L_B_random_S_detected, L_B_random_S_timesteps):
        B_random_list_of_tuples = []

        for seed, time in zip(B_random_S_detected, B_random_S_timesteps):
            seed = node_idx_to_name_mapping[seed]
            B_random_list_of_tuples.append((seed, time))
        L_B_random_list_of_tuples.append(B_random_list_of_tuples)
    print("L_B_random_list_of_tuples: {}".format(L_B_random_list_of_tuples))
    return L_B_random_list_of_tuples

def get_algo_P123_list_of_tuples(df_result, P123_name_list):
    print("prepare node tuples for our algorithms")
    algo_P123_list_of_tuples = []
    for algo_name in P123_name_list:
        algo_list_of_tuples = []

        algo_S_detected = eval(df_result.loc[algo_name]["S_detected"])
        algo_S_timesteps = eval(df_result.loc[algo_name]["S_timesteps"])
        for seed, time in zip(algo_S_detected, algo_S_timesteps):
            seed = node_idx_to_name_mapping[seed]
            algo_list_of_tuples.append((seed, time))
        algo_P123_list_of_tuples.append(algo_list_of_tuples)
    # print("algo_P123_list_of_tuples: {}".format(algo_P123_list_of_tuples))
    return algo_P123_list_of_tuples

def get_H_large(G_over_time):
    H_over_time = []

    print("Relabeling nodes...")
    G_node_list = [v for v in G_over_time[0].nodes]
    for t, G in enumerate(G_over_time):
        mapping = dict([(v, (v, t)) for v in G.nodes])
        H = nx.relabel_nodes(G, mapping)
        H_over_time.append(H)

    # Connect nodes in time t to t+1
    print("Copy nodes and edges to H_large...")
    H_large = nx.Graph()
    for t, H in enumerate(H_over_time):
        H_nodes = [v for v in H.nodes]
        H_large.add_nodes_from(H_nodes)
        print(nx.info(H_large))

        H_edges = [e for e in H.edges]
        H_large.add_edges_from(H_edges)
        print(nx.info(H_large))

    print("Add edges over time on same node")
    for t in range(len(H_over_time)-1):
        H_prev = H_over_time[t]
        H_next = H_over_time[t+1]

        H_prev_nodes = [v for v in H_prev]
        H_next_nodes = [v for v in H_next]

        H_edges_over_time = [(v1, v2) for v1, v2 in zip(H_prev_nodes, H_next_nodes)]
        H_large.add_edges_from(H_edges_over_time)
        print(nx.info(H_large))
    return H_large

def get_shortest_path_fromGT_source_nodes(H_large, GT_S_list_of_tuples):
    print("Precompute shortest path from GT source nodes to all the remaining nodes")
    sp_from_GT_S_tuples = dict()
    for GT_S_tuple in tqdm(GT_S_list_of_tuples):
        p = nx.shortest_path_length(H_large, source=GT_S_tuple)
        sp_from_GT_S_tuples[GT_S_tuple] = p
    print("Complete - shortest path from GT source nodes to all the remaining nodes")
    return sp_from_GT_S_tuples

def get_avg_of_min_hop(detected_S_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples):
    n_detected_S_S_detected = len(detected_S_list_of_tuples)
    hop_array_GT_seed_to_detected_S_seed = np.zeros((n_detected_S_S_detected, n_GT_S))
    for idx_detected_S_S, detected_S_S_tuple in enumerate(detected_S_list_of_tuples):
        for idx_GT_S, GT_S_tuple in enumerate(GT_S_list_of_tuples):
            try:
                distance = sp_from_GT_S_tuples[GT_S_tuple][detected_S_S_tuple]
            except Exception as e:
                print("{}\n".format(e))
                distance = np.nan
            hop_array_GT_seed_to_detected_S_seed[idx_detected_S_S][idx_GT_S] = distance
    print("hop_array_GT_seed_to_detected_S_seed.shape: {}".format(hop_array_GT_seed_to_detected_S_seed.shape))
    # NOTE: set this to zero gives min hop from each GT seed.
    # NOTE: set this to one gives min hop from each seed from our algo
    min_hop_array = np.nanmin(hop_array_GT_seed_to_detected_S_seed, axis=0) 
    print("min_hop_array: {}".format(min_hop_array))
    avg_of_min_hop = np.nanmean(min_hop_array)
    return avg_of_min_hop

def compute_avg_of_min_hop_list(GT_S_list_of_tuples, L_B_random_list_of_tuples, cult_list_of_tuples, \
        algo_P123_list_of_tuples):

    avg_of_min_hop_list = [0] # GT

    # n_our_algorithms = len(P123_name_list)
    n_GT_S = len(GT_S_list_of_tuples)

    n_our_algorithms = len(algo_P123_list_of_tuples)

    n_rep_B_random = len(L_B_random_list_of_tuples)

    # ------------------------------------------
    print("Get the hops Baseline random")
    B_random_avg_of_min_hop = np.zeros((n_rep_B_random))
    for idx_rep, B_random_list_of_tuples in enumerate(L_B_random_list_of_tuples):
        avg_of_min_hop = get_avg_of_min_hop(B_random_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples)
        B_random_avg_of_min_hop[idx_rep] = avg_of_min_hop
    print("B_random_avg_of_min_hop: \n{}".format(B_random_avg_of_min_hop))
    avg_of_min_hop_list.append(np.nanmean(B_random_avg_of_min_hop))

    # ------------------------------------------
    print("Get the hops cult")
    avg_of_min_hop = get_avg_of_min_hop(cult_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples)
    avg_of_min_hop_list.append(avg_of_min_hop)

    # ------------------------------------------
    print("Get the hops our algo")
    for algo_list_of_tuples in algo_P123_list_of_tuples:
        avg_of_min_hop = get_avg_of_min_hop(algo_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples)
        avg_of_min_hop_list.append(avg_of_min_hop)

    return avg_of_min_hop_list

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

    if args.dose_response=="exponential":
        txt_dose_response = ""
    elif args.dose_response=="linear":
        txt_dose_response = "linear_"

    P2_ISCK_multiplicative_update_infile_name_list = [
            "{}P2_ISCK_multiplicative_update_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
            "{}P2_ISCK_multiplicative_update_evalution_lazyT_expectedF.pickle".format(txt_dose_response),
            "{}P2_ISCK_multiplicative_update_evalution_lazyT_expectedT.pickle".format(txt_dose_response),
            ]
    P2_ISCK_multiplicative_update_name_list = [
            "{}P2_ISCK_Multiplicative_update".format(txt_dose_response),
            "{}P2_ISCK_LazyMultiplicative_update".format(txt_dose_response),
            "{}P2_ISCK_E_LazyMultiplicative_update".format(txt_dose_response),
            ]

    P3_infile_name_list = [
            "{}P3_GR_gconstraintT_evalution_lazyF_expectedF.pickle".format(txt_dose_response),
            "{}P3_GR_gconstraintT_evalution_lazyF_expectedT.pickle".format(txt_dose_response),
            ]
    P3_name_list = [
            "{}P3_GreedyRatio".format(txt_dose_response),
            "{}P3_E_GreedyRatio".format(txt_dose_response),
            ]

    np.set_printoptions(suppress=True)

    # -------------------------------------------
    path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

    list_of_df = []
    list_of_df_best_MCC = []

    print("P2 ISCK multiplicative_update...")
    for infile in P2_ISCK_multiplicative_update_infile_name_list:
        with open(path + infile, 'rb') as handle:
            P2_ISCK_evaluation_dict = pickle.load(handle)
        df_P2_ISCK = load_Px_results_for_runtime(P2_ISCK_evaluation_dict)
        list_of_df.append(df_P2_ISCK)
        df_best_MCC_P2_ISCK = load_Px_results(P2_ISCK_evaluation_dict)
        list_of_df_best_MCC.append(df_best_MCC_P2_ISCK)

    df_P2_ISCK_MU_result = concat_result_dataframes(P2_ISCK_multiplicative_update_name_list, list_of_df)
    print(df_P2_ISCK_MU_result[['n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']].round(3))
    df_best_MCC_P2_ISCK_MU_result = concat_result_dataframes(P2_ISCK_multiplicative_update_name_list, list_of_df_best_MCC)
    print(df_best_MCC_P2_ISCK_MU_result[['n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']].round(3))

    list_of_df = []
    list_of_df_best_MCC = []

    print("P3 ...")
    for infile in P3_infile_name_list:
        with open(path + infile, 'rb') as handle:
            P3_greedy_ratio_evaluation_dict = pickle.load(handle)
        df_P3_greedy_ratio = load_Px_results_for_runtime(P3_greedy_ratio_evaluation_dict)
        list_of_df.append(df_P3_greedy_ratio)
        df_best_MCC_P3_greedy_ratio = load_Px_results_for_runtime(P3_greedy_ratio_evaluation_dict)
        list_of_df_best_MCC.append(df_best_MCC_P3_greedy_ratio)

    df_P3_greedy_ratio_result = concat_result_dataframes(P3_name_list, list_of_df)
    print(df_P3_greedy_ratio_result[['n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']].round(3))
    df_best_MCC_P3_greedy_ratio_result = concat_result_dataframes(P3_name_list, list_of_df_best_MCC)
    print(df_best_MCC_P3_greedy_ratio_result[['n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']].round(3))

    # -------------------------------------------
    # plot Runtime

    # process
    # color_list = ["C0", "C1", "C2", "whilte", "C0", "C2"]
    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "white", "#1f77b4", "#2ca02c"]
    algorithm_type_list = ["Original", "Lazy", "Expected", "", "Original", "Expected"]
    # x_label_list = ["KnapSackSD", "KnapSackSD", "KnapSackSD", "", "RatioSD", "RatioSD"]
    x_label_list = ["", "KnapSackSD", "", "", "               RatioSD", ""]
    time_list = list(df_best_MCC_P2_ISCK_MU_result["Time(s)"]) + [0] + list(df_best_MCC_P3_greedy_ratio_result["Time(s)"])
    MCC_list = list(df_best_MCC_P2_ISCK_MU_result["MCC"]) + [0] + list(df_best_MCC_P3_greedy_ratio_result["MCC"])
    F1_list = list(df_best_MCC_P2_ISCK_MU_result["F1"]) + [0] + list(df_best_MCC_P3_greedy_ratio_result["F1"])

    outpath = "../plots/runtime/seedspert{}/".format(seeds_per_t)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outfile = "runtime_comparison.png"
    title = "Run time comparison"
    ylabel = "Time (s)"
    plot_bar_v3(time_list, algorithm_type_list, x_label_list, color_list, title, ylabel, outpath, outfile)

    print("also prep log time")
    log10_time_list = []
    for time in time_list:
        if time == 0:
            log10_time_list.append(time)
        else:
            log10_time_list.append(math.log10(time))
    
    outfile = "runtime_comparison_log10.png"
    title = "Run time comparison"
    ylabel = r"Log$_{10}$(Time) seconds"

    plot_bar_v3(log10_time_list, algorithm_type_list, x_label_list, color_list, title, ylabel, outpath, outfile)

    # ADDED for AAAI submission
    outfile = "runtime_MCC_comparison.png"
    title1 = "Running time"
    ylabel1 = "Time (s)"
    title2 = "MCC score"
    ylabel2 = "MCC score"
    x_label_list = ["", "KnapSackSD", "", "", "       RatioSD", ""]
    plot_bar_v4(time_list, MCC_list, algorithm_type_list, x_label_list, color_list, title1, title2, ylabel1, ylabel2, outpath, outfile)

    outfile = "runtime_F1_comparison.png"
    title1 = "Running time"
    ylabel1 = "Time (s)"
    title2 = "F1 score"
    ylabel2 = "F1 score"
    x_label_list = ["", "KnapSackSD", "", "", "       RatioSD", ""]
    plot_bar_v4(time_list, F1_list, algorithm_type_list, x_label_list, color_list, title1, title2, ylabel1, ylabel2, outpath, outfile)

