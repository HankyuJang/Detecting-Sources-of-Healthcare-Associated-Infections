"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Feb, 2022

NOTE: use networkx 2.5

Description: This script runs d_steiner on graphs

alpha = 0
beta = 0 (infection flow within itself has 0 cost)
gamma = 0, 16, 128

Usage

- 4 graphs for UIHC sampled

$ python bad_quality_d_steiner_alpha_0.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 1
$ python bad_quality_d_steiner_alpha_0.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 3
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

    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
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

    i = 1
    alpha=0
    beta=0
    gamma_list = [0.0, 16.0, 128.0]
    # gamma_list = [0.0]

    column_names = ["Algorithm","S_detected","S_timesteps","n_S","n_S_correct","hops","TP","TN","FP","FN","F1","MCC","Time(s)"]
    L_cult_S_detected = []
    L_cult_S_timesteps = []

    L_cult_n_S = []
    L_cult_n_S_correct = []
    # L_cult_loss_1 = []
    # L_cult_loss_total = []
    # L_cult_list_of_P_hit = []
    # L_cult_list_of_N_hit = []
    L_cult_TP = []
    L_cult_TN = []
    L_cult_FP = []
    L_cult_FN = []
    L_cult_F1 = []
    L_cult_MCC = []
    L_cult_time_elapsed = []
    
    for idx_gamma, gamma in enumerate(gamma_list):

        day_list = [d for v,d in G.nodes()]
        max_day = max(day_list)
        # X holds the steiner points
        X_original = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
        X = set([v for v in G.nodes() if G.nodes[v]["terminal"]])
        len_X = len(X)
        k = len_X
        print("Number of terminals: {}".format(k))
        # print(nx.info(G))
        # r is the root node
        r = (0, -1)
        # Add a dummy node, then connect it to all other nodes

        # NOTE: only connect dummy node to candidate seeds
        # add_dummy_node(G, r, 0.0, gamma)
        add_dummy_node_to_candidate_seeds(G, r, 0.0, gamma)

        print("Compute SP for i={}".format(i))
        SP = dict()
        # (1) Generate shortest path from r to all nodes
        # p = nx.shortest_path(G, source=r, weight="weight")
        # This SP algorithm can handle negative weights
        p = nx.single_source_bellman_ford_path(G, source=r, weight="weight")
        SP[r] = copy.deepcopy(p)

        # (2) Generate shortest path from all nodes to all terminal nodes
        # IDEA: reverse all edges, then compute shortest path from t to all nodes. Then, reverse the direction of final solution
        G_reversed = G.reverse()
        for t in tqdm(X):
            # p = nx.shortest_path(G, target=t, weight="weight")
            # This SP algorithm can handle negative weights
            p = nx.single_source_bellman_ford_path(G_reversed, source=t, weight="weight")
            for src in p.keys():
                if src in SP:
                    SP[src][t] = copy.deepcopy(p[src][::-1])
                else:
                    SP[src] = dict()
                    SP[src][t] = copy.deepcopy(p[src][::-1])

        print("computing the tree using d_steiner algorithm with i={}, gamma={}...".format(i, gamma))
        start = timer()
        T = d_steiner(SP, i, G, X_original, k, r, X)
        end = timer()
        cult_time_elapsed = get_elapsed_time(start, end)

        # Add node weights
        # node_weight_list = []
        # for node in T.nodes:
            # weight=G.nodes[node]["prob"]
            # node_weight_list.append((node, {"prob": weight}))
        # T.add_nodes_from(node_weight_list)
        # Save the graph
        if args.dose_response == "exponential":
            nx.write_graphml(T, "../cult/result/T_{}_seedspert{}_ntseeds{}_ntforeval{}_.graphml".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval), named_key_ids=True)
        elif args.dose_response == "linear":
            nx.write_graphml(T, "../cult/result/linear_T_{}_seedspert{}_ntseeds{}_ntforeval{}_.graphml".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval), named_key_ids=True)

        cost = sum([T.edges[e]["weight"] for e in T.edges()])
        # cost_array[idx_gamma, idx_beta, idx_alpha] = cost

        print(nx.info(T))
        print("Elapsed_time: {:.1f} mins, Cost: {:.1f}".format(cult_time_elapsed/60, cost))
        
        # get seeds (those that branch out from root)
        T_seed_list = get_seed_nodes(T, r)

        cult_seeds_array = np.zeros((seeds_array.shape)).astype(bool)

        for seed_name, time in T_seed_list:
            seed_idx = node_name_to_idx_mapping[seed_name]
            cult_seeds_array[time, seed_idx] = True


        _, cult_n_S, cult_n_S_correct, cult_loss_1, cult_loss_total, \
            cult_list_of_P_hit, cult_list_of_N_hit, \
            cult_TP, cult_TN, cult_FP, cult_FN, cult_F1, cult_MCC = \
                evaluate_solution_seeds(simul, list_of_people_idx_arrays, seeds_array, cult_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval)

        L_cult_S_detected.append(str(list(cult_seeds_array.nonzero()[1])))
        L_cult_S_timesteps.append(str(list(cult_seeds_array.nonzero()[0])))

        L_cult_n_S.append(cult_n_S)
        L_cult_n_S_correct.append(cult_n_S_correct)
        # L_cult_loss_1.append(cult_loss_1)
        # L_cult_loss_total.append(cult_loss_total)
        # L_cult_list_of_P_hit.append(cult_list_of_P_hit)
        # L_cult_list_of_N_hit.append(cult_list_of_N_hit)
        L_cult_TP.append(cult_TP)
        L_cult_TN.append(cult_TN)
        L_cult_FP.append(cult_FP)
        L_cult_FN.append(cult_FN)
        L_cult_F1.append(cult_F1)
        L_cult_MCC.append(cult_MCC)
        L_cult_time_elapsed.append(cult_time_elapsed)

        df_cult= pd.DataFrame({
            "gamma": gamma_list[:idx_gamma+1],
            "S_detected": L_cult_S_detected,
            "S_timesteps": L_cult_S_timesteps,
            "n_S": L_cult_n_S,
            "n_S_correct": L_cult_n_S_correct,
            "TP": L_cult_TP,
            "TN": L_cult_TN,
            "FP": L_cult_FP,
            "FN": L_cult_FN,
            "F1": L_cult_F1,
            "MCC": L_cult_MCC,
            "Time(s)": L_cult_time_elapsed
            })
        print("\ncult results")
        print(df_cult.round(2))

    # output
    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)

    if args.dose_response == "exponential":
        outfile = "cult.csv"
    elif args.dose_response == "linear":
        outfile = "linear_cult.csv"

    df_cult.to_csv(path+outfile, index=False)

