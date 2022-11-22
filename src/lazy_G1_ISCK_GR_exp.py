"""
Author: -
Email: -
Last Modified: Dec, 2021 

Description: 

results saved in `tables/lazy_G1_ISCK_GR_exp/`

**Problem setup**

1. Data duration: 1 month (timesteps 0, 1, ..., 30)
    1. Seeds: timesteps 0, 1
    2. Observe infections in (+) set and (-) set in 29, 30.

Usage

To run it on Karate graph,
$ python lazy_G1_ISCK_GR_exp.py -seeds_per_t 1

To run it on UIHC sampled graph,
$ python lazy_G1_ISCK_GR_exp.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1

To run it on UIHC original graph,
$ python lazy_G1_ISCK_GR_exp.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -seeds_per_t 1
"""

from utils.load_network import *
from utils.set_parameters import *
from simulator_load_sharing_temporal_v2 import *
from approx_algorithms import *
from prep_GT_observation import *
from get_people_nodes import *

import argparse
import pandas as pd
import random as random
import timeit

def prepare_ground_truth_table(k, GT_seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval):
    S = list(GT_seeds_array.nonzero()[1])
    S_timestep = list(GT_seeds_array.nonzero()[0])
    len_P_t_over_time = [len(P_t) for P_t in list_of_sets_of_P[-n_t_for_eval:]]
    len_N_t_over_time = [len(N_t) for N_t in list_of_sets_of_N[-n_t_for_eval:]]
    df_ground_truth = pd.DataFrame(
            {"Seed_idx": [str(S)],
                "Seed_timesteps": [str(S_timestep)],
                "|P_t|last{}ts".format(n_t_for_eval): str(len_P_t_over_time),
                "|N_t|last{}ts".format(n_t_for_eval): str(len_N_t_over_time)
                })
    return df_ground_truth

def prepare_df_exp(detected_seeds_array, n_S, n_S_correct, \
                    TP, TN, FP, FN, F1, MCC, time_elapsed):
    S_detected = list(detected_seeds_array.nonzero()[1])
    S_timesteps = list(detected_seeds_array.nonzero()[0])
    df_exp = pd.DataFrame({
                "S_detected": [str(S_detected)],
                "S_timesteps": [str(S_timesteps)],
                "n_S": [n_S],
                "n_S_correct": [n_S_correct],
                "TP": [TP],
                "TN": [TN],
                "FP": [FP],
                "FN": [FN],
                "F1": [F1],
                "MCC": [MCC],
                "Time": ["{:.3f} s".format(time_elapsed)],
            })
    return df_exp

def prepare_result_dataframes():
    GT_n_S = np.sum(number_of_seeds_over_time)

    df_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC, GT_time_elapsed)
    df_BR = prepare_df_exp(BR_seeds_array, BR_n_S, BR_n_S_correct, \
            BR_TP, BR_TN, BR_FP, BR_FN, BR_F1, BR_MCC, BR_time_elapsed)
    df_G1_Lazy = prepare_df_exp(G1_Lazy_seeds_array, G1_Lazy_n_S, G1_Lazy_n_S_correct, \
            G1_Lazy_TP, G1_Lazy_TN, G1_Lazy_FP, G1_Lazy_FN, G1_Lazy_F1, G1_Lazy_MCC, G1_Lazy_time_elapsed)
    df_ISCK_Lazy = prepare_df_exp(ISCK_Lazy_seeds_array, ISCK_Lazy_n_S, ISCK_Lazy_n_S_correct, \
            ISCK_Lazy_TP, ISCK_Lazy_TN, ISCK_Lazy_FP, ISCK_Lazy_FN, ISCK_Lazy_F1, ISCK_Lazy_MCC, ISCK_Lazy_time_elapsed)
    df_GR = prepare_df_exp(GR_seeds_array, GR_n_S, GR_n_S_correct, \
            GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC, GR_time_elapsed)

    return df_GT, df_BR, df_G1_Lazy, df_ISCK_Lazy, df_GR

def concat_result_dataframes():
    index_list = ["GT", "Random", "P1-LazyGreedy", "P2-LazyISCK", "P3-GreedyRatio"]
    df_result = pd.concat([df_GT, df_BR, df_G1_Lazy, df_ISCK_Lazy, df_GR])
    df_result["Algorithm"] = index_list
    df_result.set_index("Algorithm", inplace=True)
    return df_result

def save_result_dataframes(name, k_total):
    # Save datasets
    df_GT.to_csv("../tables/lazy_G1_ISCK_GR_exp/{}/k{}/GT.csv".format(name, k_total), index=False)
    df_BR.to_csv("../tables/lazy_G1_ISCK_GR_exp/{}/k{}/BR.csv".format(name, k_total), index=False)
    df_G1_Lazy.to_csv("../tables/lazy_G1_ISCK_GR_exp/{}/k{}/G1_Lazy.csv".format(name, k_total), index=False)
    df_ISCK_Lazy.to_csv("../tables/lazy_G1_ISCK_GR_exp/{}/k{}/ISCK_Lazy.csv".format(name, k_total), index=False)
    df_GR.to_csv("../tables/lazy_G1_ISCK_GR_exp/{}/k{}/GR.csv".format(name, k_total), index=False)

    df_result.to_csv("../tables/lazy_G1_ISCK_GR_exp/{}/k{}/result_concat.csv".format(name, k_total), index=True)
    df_ground_truth_observations.to_csv("../tables/lazy_G1_ISCK_GR_exp/{}/k{}/GT_observations.csv".format(name, k_total), index=False)

def print_result_dataframes():
    print("\nGround Truth")
    print(df_GT.round(2))
    print("\nBaseline. Random")
    print(df_BR.round(2))
    print("\nProblem1: Lazy Greedy")
    print(df_G1_Lazy.round(2))
    print("\nProblem2: Lazy ISCK")
    print(df_ISCK_Lazy.round(2))
    print("\nProblem3: Greedy Ratio")
    print(df_GR.round(2))

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
                        help= 'Quality of the ground truth simulation. best | median')
    parser.add_argument('-seeds_per_t', '--seeds_per_t', type=int, default=1,
                        help= 'number of seeds per timestep')
    parser.add_argument('-n_t_seeds', '--n_t_seeds', type=int, default=2,
                        help= 'number of timesteps for seeds')
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    ####################################################################
    # Parameters for the simulation. These are same regardless of the graph
    n_timesteps = 31
    n_replicates = 100 
    area_people = 2000 # area of patient. 2000cm^2
    area_location = 40000 # area of room. 40000cm^2
    ####################################################################
    # Parameters for experiments
    # NOTE: treat T as a global variable. Used anywhere in this script.
    T = n_timesteps-1 # T is the index of the last timestep
    n_t_for_eval = args.n_t_for_eval # Use the latest n timesteps for evaluation. e.g., T and T-1

    flag_increase_area = True # If this is set to True, then increase area of each node based on their max degree over grpahs

    ####################################################################
    # Additional input for problem 2
    array_of_knapsack_constraints_on_f = np.zeros((n_timesteps))
    if args.name=="Karate_temporal":
        array_of_knapsack_constraints_on_f[-1] = 2
        array_of_knapsack_constraints_on_f[-2] = 1
    else:
        array_of_knapsack_constraints_on_f[-1] = 2
        array_of_knapsack_constraints_on_f[-2] = 1

    ####################################################################
    # Additional input for problem 3
    array_of_penalty_on_f = np.zeros((n_timesteps))
    array_of_penalty_on_f[-1] = 1
    array_of_penalty_on_f[-2] = 2

    ####################################################################
    # Ground truth seeds over time
    number_of_seeds_over_time = np.zeros((n_timesteps)).astype(int)
    for t in range(args.n_t_seeds):
        number_of_seeds_over_time[t] = args.seeds_per_t

    k_total = np.sum(number_of_seeds_over_time)
    print("Set number of seeds at various timesteps\ntime 0: 1 seed\ntime 1: 1 seed")
    print("number_of_seeds_over_time: {}\n".format(number_of_seeds_over_time))

    ####################################################################
    print("Load network...\n")
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, graph_name = process_data_for_experiments(args, area_people, area_location, flag_increase_area)

    # NOTE: Make sure all the graphs in different time snapshots have same set of nodes
    node_name_to_idx_mapping = dict([(node_name, node_idx) for node_idx, node_name in enumerate(G_over_time[0].nodes())])
    node_idx_to_name_mapping = dict([(node_idx, node_name) for node_idx, node_name in enumerate(G_over_time[0].nodes())])

    # Get a list of people index arrays. Array at each idx correspond to the indicies of people at that day
    list_of_people_idx_arrays = get_people_idx_array_over_time(G_over_time, node_name_to_idx_mapping, people_nodes_idx)
    # List of sets of people indicies
    list_of_sets_of_V = [set(arr) for arr in list_of_people_idx_arrays]

    ####################################################################
    # 0. Create simulation instance with empty seeds list
    rho, d, q, pi, contact_area = set_simulation_parameters(args, k_total)
    simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)

    ####################################################################
    # NOTE: For all experiments, run it for n_replicates per seed set
    simul.set_n_replicates(n_replicates)
    ####################################################################
    # Set random seed, and observe infections
    # 1. Data generation
    print("Generate seed set w/ the best quality. Get ground truth observations...")
    seeds_array, obs_state, I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N \
            = prepare_GT_data(simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, args.GT_quality)

    ####################################################################
    # 2. Compute ground truth loss per timestep
    # We're not interested in loss over timestep (e.g. missing infection) in this project, so just take the loss at the last timestep.
    start = timeit.default_timer()
    print("Compute GT losses")
    GT_loss_1, GT_loss_total, \
        GT_list_of_P_hit, GT_list_of_N_hit, \
        GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC = \
        compute_GT_loss_per_timestep(simul, args, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
    stop = timeit.default_timer()
    GT_time_elapsed = stop - start

    ####################################################################
    # Baselines
    # Randomly selected seed out of people nodes
    start = timeit.default_timer()
    print("-"*20)
    print("Compute random baseline")
    BR_seeds_array, BR_n_S, BR_n_S_correct, BR_loss_1, BR_loss_total, \
        BR_list_of_P_hit, BR_list_of_N_hit, \
        BR_TP, BR_TN, BR_FP, BR_FN, BR_F1, BR_MCC = \
            run_BR_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                                            seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
    stop = timeit.default_timer()
    BR_time_elapsed = stop - start

    ####################################################################
    # 3. Greedy source detection
    start = timeit.default_timer()
    MTP = (False, -1) # Do not use multicores in replicates. Not implemented yet.
    print("-"*20)
    print("Run Lazy greedy source detection, compute loss per timestep for the best nodeset")
    print("Greedy1 Lazy: optimize on loss_1")
    focus_obs1 = True
    G1_Lazy_seeds_array, G1_Lazy_n_S, G1_Lazy_n_S_correct, G1_Lazy_loss_1, G1_Lazy_loss_total, \
        G1_Lazy_list_of_P_hit, G1_Lazy_list_of_N_hit, \
        G1_Lazy_TP, G1_Lazy_TN, G1_Lazy_FP, G1_Lazy_FN, G1_Lazy_F1, G1_Lazy_MCC = \
            run_greedy_source_detection_report_loss_per_timestep(simul, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, MTP, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=True)
    stop = timeit.default_timer()
    G1_Lazy_time_elapsed = stop - start

    ####################################################################
    # 3. Greedy source detection
    start = timeit.default_timer()
    MTP = (False, -1) # Do not use multicores in replicates. Not implemented yet.
    print("-"*20)
    print("Run Lazy ISCK source detection, compute loss per timestep for the best nodeset")
    print("ISCK Lazy")
    focus_obs1 = True
    ISCK_Lazy_seeds_array, ISCK_Lazy_n_S, ISCK_Lazy_n_S_correct, ISCK_Lazy_loss_1, ISCK_Lazy_loss_total, \
        ISCK_Lazy_list_of_P_hit, ISCK_Lazy_list_of_N_hit, \
        ISCK_Lazy_TP, ISCK_Lazy_TN, ISCK_Lazy_FP, ISCK_Lazy_FN, ISCK_Lazy_F1, ISCK_Lazy_MCC = \
            run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                array_of_knapsack_constraints_on_f, flag_lazy=True)

    stop = timeit.default_timer()
    ISCK_Lazy_time_elapsed = stop - start

    ####################################################################
    # Greedy ratio
    start = timeit.default_timer()
    print("-"*20)
    print("Run greedy ratio, compute loss per timestep for the best nodeset")
    print("Greedy Ratio")
    GR_seeds_array, GR_n_S, GR_n_S_correct, GR_loss_1, GR_loss_total, \
        GR_list_of_P_hit, GR_list_of_N_hit, \
        GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC = \
            run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_memoize=True)

    stop = timeit.default_timer()
    GR_time_elapsed = stop - start

    ####################################################################
    # Generate result tables
    df_GT, df_BR, df_G1_Lazy, df_ISCK_Lazy, df_GR = prepare_result_dataframes()
    df_result = concat_result_dataframes()

    df_ground_truth_observations = prepare_ground_truth_table(sum(number_of_seeds_over_time), seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
    print("\nGround Truth observations. Seeds and number of observed outcomes at last n timesteps")
    print(df_ground_truth_observations)

    # Print result tables
    print_result_dataframes()

    # Save the result tables
    save_result_dataframes(graph_name, k_total)

