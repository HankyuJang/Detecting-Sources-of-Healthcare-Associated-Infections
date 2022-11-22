"""
Author: -
Email: -
Last Modified: Aug 2022

Description: 

This script loads df_result and plots figures

Usage

python final_exp_result_hop_distance.py -seeds_per_t 1

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
    dose_response = args.dose_response

    graph_name = get_graph_name(args)

    k_total = n_t_seeds * seeds_per_t

    np.set_printoptions(suppress=True)

    # -------------------------------------------
    graph_name_mapping = get_graph_name_mapping()
    algorithm_name_mapping = get_algorithm_name_mapping()
    #NOTE: keep the order of the below three lists same.
    # graph_name_list = ["UIHC_HCP_patient_room_withinHCPxPx_2011_sampled", "UIHC_HCP_patient_room_withinHCPxPx_2011", "G_UVA_v2", "G_UVA_v4", "G_Carilion"]
    graph_name_list = ["UIHC_HCP_patient_room_withinHCPxPx_2011_sampled", "UIHC_HCP_patient_room_withinHCPxPx_2011", "G_Carilion"]
    name_list = [graph_name_mapping[graph_name] for graph_name in graph_name_list]

    P1_LazyGreedy_list = ["P1_E_LazyGreedy", "linear_P1_LazyGreedy", "P1_LazyGreedy"]

    P2_LazyISCK_greedy_list = ["P2_LazyISCK_greedy", "linear_P2_LazyISCK_greedy", "P2_E_LazyISCK_greedy"]
    P2_LazyISCK_multiplicative_update_list = ["P2_LazyISCK_multiplicative_update", "linear_P2_LazyISCK_multiplicative_update", "P2_E_LazyISCK_multiplicative_update"]

    P3_GreedyRatio_list = ["P3_GreedyRatio", "linear_P3_GreedyRatio", "P3_E_GreedyRatio"]
    P3_GreedyRatio_ghit50_list = ["P3_GreedyRatio_ghit>50", "linear_P3_GreedyRatio_ghit>50", "P3_E_GreedyRatio_ghit>50"]

    result_hop_dict = dict()

    for name, graph_name in zip(name_list, graph_name_list):
        path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)
        if dose_response=="exponential":
            infile = "result_concat.csv"
        elif dose_response=="linear":
            infile = "linear_result_concat.csv"
        try:
            df_result = pd.read_csv(path + infile)
            print(df_result)

            P1_LazyGreedy = df_result[df_result["Algorithm"].isin(P1_LazyGreedy_list)]["Algorithm"].values[0]

            P2_LazyISCK_greedy_MCC = df_result[df_result["Algorithm"].isin(P2_LazyISCK_greedy_list)]["MCC"].values[0]
            P2_LazyISCK_greedy = df_result[df_result["Algorithm"].isin(P2_LazyISCK_greedy_list)]["Algorithm"].values[0]

            P2_LazyISCK_multiplicative_update_MCC = df_result[df_result["Algorithm"].isin(P2_LazyISCK_multiplicative_update_list)]["MCC"].values[0]
            P2_LazyISCK_multiplicative_update = df_result[df_result["Algorithm"].isin(P2_LazyISCK_multiplicative_update_list)]["Algorithm"].values[0]

            if P2_LazyISCK_greedy_MCC > P2_LazyISCK_multiplicative_update_MCC:
                P2_algorithm_to_keep = P2_LazyISCK_greedy
            else:
                P2_algorithm_to_keep = P2_LazyISCK_multiplicative_update

            P3_GreedyRatio_MCC = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_list)]["MCC"].values[0]
            P3_GreedyRatio = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_list)]["Algorithm"].values[0]

            P3_GreedyRatio_ghit50_MCC = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_ghit50_list)]["MCC"].values[0]
            P3_GreedyRatio_ghit50 = df_result[df_result["Algorithm"].isin(P3_GreedyRatio_ghit50_list)]["Algorithm"].values[0]

            if P3_GreedyRatio_MCC > P3_GreedyRatio_ghit50_MCC:
                P3_algorithm_to_keep = P3_GreedyRatio
            else:
                P3_algorithm_to_keep = P3_GreedyRatio_ghit50

            # algorithm_array_to_keep = ["GT", "Random(30rep)", "cult", "netsleuth", "reachability", "LOS", P1_LazyGreedy, P2_algorithm_to_keep, P3_algorithm_to_keep]
            algorithm_array_to_keep = ["GT", "Random(30rep)", "cult", "netsleuth", "reachability", "LOS", P2_algorithm_to_keep, P3_algorithm_to_keep]
            label_list = [algorithm_name_mapping[algo] for algo in algorithm_array_to_keep]

            df_result_to_keep = df_result[df_result.Algorithm.isin(algorithm_array_to_keep)]

        except:
            print("No result on this setting yet")
            # for method in method_list:
                # result_hop_dict[method].append(np.nan)
            continue

        print(df_result_to_keep)
        
        result_hop_dict[name] = df_result_to_keep["hops"].values

    print("avg hop distance from GT")
    df_result_hop = pd.DataFrame(data=result_hop_dict, index=label_list)
    df_result_hop = df_result_hop.round(1)
    print(df_result_hop)

    outpath = "../tables/final_exp/hop/"

    outfile = "harmonic_mean_result_hop_k{}_{}.csv".format(k_total, args.dose_response)
    df_result_hop.to_csv(outpath+outfile, index=True)

