"""
Author: -
Email: -
Last Modified: Feb 2022

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
            "reachability": "Reachability",
            "LOS": "LOS",
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
            "G_UVA": "UVA",
            "UIHC_HCP_patient_room_withinHCPxPx_2011": "UIHC",
            "UIHC_HCP_patient_room_withinHCPxPx_2011_sampled": "UIHC-S"
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
    dose_response = args.dose_response

    graph_name = get_graph_name(args)

    k_total = n_t_seeds * seeds_per_t

    np.set_printoptions(suppress=True)

    # -------------------------------------------
    graph_name_mapping = get_graph_name_mapping()
    algorithm_name_mapping = get_algorithm_name_mapping()

    #NOTE: keep the order of the below three lists same.
    graph_name_list = ["UIHC_HCP_patient_room_withinHCPxPx_2011_sampled", "UIHC_HCP_patient_room_withinHCPxPx_2011", "G_Carilion", "G_UVA"]

    name_list = [graph_name_mapping[graph_name] for graph_name in graph_name_list]

    method_list = ["Ground Truth", "Random", "Cult", "Reachability", "LOS", "Greedy", "ISCK-Greedy", "ISCK-M.Update", "GreedyRatio", "GreedyRatio-hit50"]

    result_hop_dict = dict()
    result_hop_v2_dict = dict()

    for method in method_list:
        result_hop_dict[method] = []
        result_hop_v2_dict[method] = []

    for graph_name in graph_name_list:
        path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)
        if dose_response=="exponential":
            infile = "result_concat.csv"
        elif dose_response=="linear":
            infile = "linear_result_concat.csv"
        try:
            df_result = pd.read_csv(path + infile)
        except:
            print("No result on this setting yet")
            for method in method_list:
                result_hop_dict[method].append(np.nan)
            continue

        algorithm_array = df_result["Algorithm"].values
        label_list = [algorithm_name_mapping[algo] for algo in algorithm_array]
        hop_list = df_result["hops"].values
        label_hop_dict = dict([(label, hop) for label, hop in zip(label_list, hop_list)])

        # v2: 
        hop_v2_list = (df_result["hops"] * df_result["n_S"]).values
        label_hop_v2_dict = dict([(label, hop_v2) for label, hop_v2 in zip(label_list, hop_v2_list)])

        for method in method_list:
            if method in label_hop_dict:
                hop = label_hop_dict[method]
                result_hop_dict[method].append(hop)

                hop_v2 = label_hop_v2_dict[method]
                result_hop_v2_dict[method].append(hop_v2)
            else:
                result_hop_dict[method].append(np.nan)

                result_hop_v2_dict[method].append(np.nan)
    
    print("avg hop distance from GT")
    df_result_hop = pd.DataFrame(data=result_hop_dict, index=name_list)
    df_result_hop = df_result_hop.T.round(1)
    print(df_result_hop)

    print("\navg hop distance from GT / detected seed cardinality")
    df_result_hop_v2 = pd.DataFrame(data=result_hop_v2_dict, index=name_list)
    df_result_hop_v2 = df_result_hop_v2.T.round(1)
    print(df_result_hop_v2)

    outpath = "../tables/final_exp/hop/"

    outfile = "result_hop_k{}_{}.csv".format(k_total, args.dose_response)
    df_result_hop.to_csv(outpath+outfile, index=True)

    outfile = "result_hop_v2_k{}_{}.csv".format(k_total, args.dose_response)
    df_result_hop_v2.to_csv(outpath+outfile, index=True)

