"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Feb, 2022

NOTE: use networkx 2.5

Description: This script loads netsleuth result seeds then evalaute

Usage

- 4 graphs for UIHC sampled

$ python bad_quality_netsleuth_evaluation.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 1
$ python bad_quality_netsleuth_evaluation.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 3
"""

import argparse

from utils.networkx_operations import *
from utils.pandas_operations import *

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
from approx_algorithms import evaluate_solution_seeds

from tqdm import tqdm
import pandas as pd
import numpy as np
import copy

import pickle
import csv

def get_graphname_mapping():
    mapping = {
            "UIHC_HCP_patient_room_withinHCPxPx_2011_sampled": "UIHC_S",
            "UIHC_HCP_patient_room_withinHCPxPx_2011": "UIHC",
            "G_Carilion": "Carilion"
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
    args = parser.parse_args()

    graph_name = get_graph_name(args)

    print("Load GT observation")
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
    # 
    print("Load mappings for netsleuth")
    netsleuth_inpath = "../Netsleuth/GT_bad_node_mapping/"
    infile = "node_mapping_{}_seedspert{}_ntseeds{}_ntforeval{}.pickle".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    with open(netsleuth_inpath + infile, 'rb') as handle:
        node_mapping_for_netsleuth = pickle.load(handle)

    matlab_idx_to_node_mapping = node_mapping_for_netsleuth["matlab_idx_to_node_mapping"]
    node_to_matlab_idx_mapping = node_mapping_for_netsleuth["node_to_matlab_idx_mapping"]

    netsleuth_graphname_mapping = get_graphname_mapping()
    if graph_name in netsleuth_graphname_mapping:
        netsleuth_graph_name = netsleuth_graphname_mapping[graph_name]
    else:
        netsleuth_graph_name = graph_name

    print("Load netsleuth seeds")
    netsleuth_inpath = "../Netsleuth/GT_bad_output/"
    if args.dose_response == "exponential":
        infile = "Seed_{}_s{}.csv".format(netsleuth_graph_name, args.seeds_per_t)
    elif args.dose_response == "linear":
        infile = "linear_Seed_{}_s{}.csv".format(netsleuth_graph_name, args.seeds_per_t)

    with open(netsleuth_inpath+infile, newline="") as f:
        reader = csv.reader(f)
        seed_matlab_idx_list = next(reader)

    seed_matlab_idx_list = [int(seed_matlab_idx) for seed_matlab_idx in seed_matlab_idx_list]
    seed_node_list = [matlab_idx_to_node_mapping[seed_matlab_idx] for seed_matlab_idx in seed_matlab_idx_list]

    column_names = ["Algorithm","S_detected","S_timesteps","n_S","n_S_correct","hops","TP","TN","FP","FN","F1","MCC","Time(s)"]
    L_netsleuth_S_detected = []
    L_netsleuth_S_timesteps = []

    L_netsleuth_n_S = []
    L_netsleuth_n_S_correct = []
    L_netsleuth_TP = []
    L_netsleuth_TN = []
    L_netsleuth_FP = []
    L_netsleuth_FN = []
    L_netsleuth_F1 = []
    L_netsleuth_MCC = []
    L_netsleuth_time_elapsed = []

    netsleuth_time_elapsed = 0

    beta_list = [0.1]

    for idx_beta, beta in enumerate(beta_list):

        netsleuth_seeds_array = np.zeros((seeds_array.shape)).astype(bool)

        for seed_name in seed_node_list:
            seed_idx = node_name_to_idx_mapping[seed_name]

            time = 0
            netsleuth_seeds_array[time, seed_idx] = True

            time = 1
            netsleuth_seeds_array[time, seed_idx] = True


        _, netsleuth_n_S, netsleuth_n_S_correct, netsleuth_loss_1, netsleuth_loss_total, \
            netsleuth_list_of_P_hit, netsleuth_list_of_N_hit, \
            netsleuth_TP, netsleuth_TN, netsleuth_FP, netsleuth_FN, netsleuth_F1, netsleuth_MCC = \
                evaluate_solution_seeds(simul, list_of_people_idx_arrays, seeds_array, netsleuth_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval)

        L_netsleuth_S_detected.append(str(list(netsleuth_seeds_array.nonzero()[1])))
        L_netsleuth_S_timesteps.append(str(list(netsleuth_seeds_array.nonzero()[0])))

        L_netsleuth_n_S.append(netsleuth_n_S)
        L_netsleuth_n_S_correct.append(netsleuth_n_S_correct)
        L_netsleuth_TP.append(netsleuth_TP)
        L_netsleuth_TN.append(netsleuth_TN)
        L_netsleuth_FP.append(netsleuth_FP)
        L_netsleuth_FN.append(netsleuth_FN)
        L_netsleuth_F1.append(netsleuth_F1)
        L_netsleuth_MCC.append(netsleuth_MCC)
        L_netsleuth_time_elapsed.append(netsleuth_time_elapsed)

        df_netsleuth= pd.DataFrame({
            "beta": beta_list[:idx_beta+1],
            "S_detected": L_netsleuth_S_detected,
            "S_timesteps": L_netsleuth_S_timesteps,
            "n_S": L_netsleuth_n_S,
            "n_S_correct": L_netsleuth_n_S_correct,
            "TP": L_netsleuth_TP,
            "TN": L_netsleuth_TN,
            "FP": L_netsleuth_FP,
            "FN": L_netsleuth_FN,
            "F1": L_netsleuth_F1,
            "MCC": L_netsleuth_MCC,
            "Time(s)": L_netsleuth_time_elapsed
            })
        print("\nnetsleuth results")
        print(df_netsleuth.round(2))

    # output
    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)

    if args.dose_response == "exponential":
        outfile = "netsleuth.csv"
    elif args.dose_response == "linear":
        outfile = "linear_netsleuth.csv"

    df_netsleuth.to_csv(path+outfile, index=False)

