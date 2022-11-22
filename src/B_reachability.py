"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Feb, 2022

NOTE: use networkx 2.5

Description: This script runs reachability baseline on the time expanded graphs

Usage

- 4 graphs for UIHC sampled

$ python B_reachability.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 1
$ python B_reachability.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 3
$ python B_reachability.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response linear -seeds_per_t 1
$ python B_reachability.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response linear -seeds_per_t 3

- 2 graphs for UIHC whole

$ python B_reachability.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -dose_response exponential -seeds_per_t 1
$ python B_reachability.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -dose_response exponential -seeds_per_t 3

- 2 graphs for G_Carilion

$ python B_reachability.py -name G_Carilion -dose_response exponential -seeds_per_t 1
$ python B_reachability.py -name G_Carilion -dose_response exponential -seeds_per_t 3
"""

import argparse

from utils.networkx_operations import *
from utils.pandas_operations import *
from utils.time_operations import *
from utils.steiner_tree_te import *

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
from approx_algorithms import evaluate_solution_seeds

from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

import pickle
import random

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# return a list of nodes that branches out from the root node
def get_seed_nodes(graph, r):
    return [dst for src, dst in graph.edges() if src == r]

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
    args = parser.parse_args()

    graph_name = get_graph_name(args)

    path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    if args.dose_response == "exponential":
        infile = "GT_observation_evalution.pickle"
    elif args.dose_response == "linear":
        infile = "linear_GT_observation_evalution.pickle"
    with open(path + infile, 'rb') as handle:
        GT_output_dict = pickle.load(handle)

    n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

    ####################################################################
    print("Load network for evaluating seeds...\n")
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, _ = process_data_for_experiments(args, area_people, area_location, flag_increase_area)

    print("list_of_sets_of_P at T: {}".format(list_of_sets_of_P[T]))

    ####################################################################
    # 0. Create simulation instance with empty seeds list
    rho, d, q, pi, contact_area = set_simulation_parameters(args, k_total)
    print("rho: {}".format(rho))
    print("d: {}".format(d))
    print("q: {}".format(q))
    print("pi: {}".format(pi))
    print("contact_area: {}".format(contact_area))

    simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    simul.set_n_replicates(n_replicates)
    
    ####################################################################

    path = "../cult/graph/"
    infile = "G_time_expanded_{}_seedspert{}_ntseeds{}_ntforeval{}.graphml".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)

    print("load time expanded graph")
    G = nx.read_graphml(path+infile)
    G = relabel_nodes_str_to_tuple(G) 



    column_names = ["Algorithm","S_detected","S_timesteps","n_S","n_S_correct","hops","TP","TN","FP","FN","F1","MCC","Time(s)"]

    L_B_reachability_S_detected = []
    L_B_reachability_S_timesteps = []

    L_B_reachability_n_S = []
    L_B_reachability_n_S_correct = []
    L_B_reachability_TP = []
    L_B_reachability_TN = []
    L_B_reachability_FP = []
    L_B_reachability_FN = []
    L_B_reachability_F1 = []
    L_B_reachability_MCC = []
    L_B_reachability_time_elapsed = []

    day_list = [d for v,d in G.nodes()]
    max_day = max(day_list)
    # X holds the steiner points
    X_original = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
    X = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
    len_X = len(X)
    k = len_X
    print("Number of terminals: {}".format(k))

    candidate_seed_list = [ v for v in G.nodes() if G.nodes[v]["S_candidate"]==1 ]


    start = timer()
    # Step1: Precompute shortest path to all the terminal nodes.
    # Step2: For each candidate seeds, count the number of terminal nodes that are reachable.
    reachable_dict = dict()
    for v in candidate_seed_list:
        reachable_dict[v] = 0

    G_reversed = G.reverse()
    for t in tqdm(X):
        # This SP algorithm can handle negative weights
        p = nx.single_source_bellman_ford_path(G_reversed, source=t)
        for src in p.keys():
            if src in reachable_dict:
                reachable_dict[src] += 1

    # print(reachable_dict)

    unique_values = np.unique(np.array(list(reachable_dict.values())))
    unique_values_sorted = np.sort(unique_values)[::-1] # sort in decreasing order

    # Step3: Sort candidate seeds in terms of the number of reachable nodes. Shuffle within those with the same number of seeds
    total_seed_list = [] # This list of seeds are randomly sorted within the seeds that have same reachability.
    for value in unique_values_sorted:
        seed_list_w_same_value = [v for v in reachable_dict if reachable_dict[v] == value]
        random.shuffle(seed_list_w_same_value)
        total_seed_list.extend(seed_list_w_same_value)
    end = timer()
    B_reachability_time_elapsed = get_elapsed_time(start, end)

    ####################################################################
    # Additional input for problem 1
    print("range of seed set size")
    epsilon = k_total * 0.5 # k_total is the ground truth number of seeds
    seed_set_size_constraint_min = int(k_total - epsilon)
    seed_set_size_constraint_max = int(k_total + epsilon)
    seed_set_size_constraint_list = [k for k in range(seed_set_size_constraint_min, seed_set_size_constraint_max+1)]
    print("Seed set size constraint: {}".format(seed_set_size_constraint_list))

    # Step4: Select top k seeds.
    for idx_seed_set_size, seed_set_size in enumerate(seed_set_size_constraint_list):

        B_reachability_seed_list = total_seed_list[:seed_set_size]

        B_reachability_seeds_array = np.zeros((seeds_array.shape)).astype(bool)

        for seed_name, time in B_reachability_seed_list:
            seed_idx = node_name_to_idx_mapping[seed_name]
            B_reachability_seeds_array[time, seed_idx] = True

        _, B_reachability_n_S, B_reachability_n_S_correct, B_reachability_loss_1, B_reachability_loss_total, \
            B_reachability_list_of_P_hit, B_reachability_list_of_N_hit, \
            B_reachability_TP, B_reachability_TN, B_reachability_FP, B_reachability_FN, B_reachability_F1, B_reachability_MCC = \
                evaluate_solution_seeds(simul, list_of_people_idx_arrays, seeds_array, B_reachability_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval)

        L_B_reachability_S_detected.append(str(list(B_reachability_seeds_array.nonzero()[1])))
        L_B_reachability_S_timesteps.append(str(list(B_reachability_seeds_array.nonzero()[0])))

        L_B_reachability_n_S.append(B_reachability_n_S)
        L_B_reachability_n_S_correct.append(B_reachability_n_S_correct)
        L_B_reachability_TP.append(B_reachability_TP)
        L_B_reachability_TN.append(B_reachability_TN)
        L_B_reachability_FP.append(B_reachability_FP)
        L_B_reachability_FN.append(B_reachability_FN)
        L_B_reachability_F1.append(B_reachability_F1)
        L_B_reachability_MCC.append(B_reachability_MCC)
        L_B_reachability_time_elapsed.append(B_reachability_time_elapsed)

        df_B_reachability= pd.DataFrame({
            "seed_set_size": seed_set_size_constraint_list[ : idx_seed_set_size+1],
            "S_detected": L_B_reachability_S_detected,
            "S_timesteps": L_B_reachability_S_timesteps,
            "n_S": L_B_reachability_n_S,
            "n_S_correct": L_B_reachability_n_S_correct,
            "TP": L_B_reachability_TP,
            "TN": L_B_reachability_TN,
            "FP": L_B_reachability_FP,
            "FN": L_B_reachability_FN,
            "F1": L_B_reachability_F1,
            "MCC": L_B_reachability_MCC,
            "Time(s)": L_B_reachability_time_elapsed
            })

    print("\nB_reachability results")
    print(df_B_reachability.round(2))

    # output
    path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)

    if args.dose_response == "exponential":
        outfile = "B_reachability.csv"
    elif args.dose_response == "linear":
        outfile = "linear_B_reachability.csv"

    df_B_reachability.to_csv(path+outfile, index=False)

