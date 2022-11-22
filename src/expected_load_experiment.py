"""
Author: -
Email: -
Last Modified: Dec, 2021 

Description: 

Has result on original algo, lazy version, and lazy + expected version

results saved in `tables/expected_load/`

**Problem setup**

1. Data duration: 1 month (timesteps 0, 1, ..., 30)
    1. Seeds: timesteps 0, 1
    2. Observe infections in (+) set and (-) set in 29, 30.
        1. We penalize more on earlier timestep on N(s). E.g., 
            1. 2 x N(s) in time 29
            2. N(s) in time 30
        2. Upper bound constraint on N(s)
            1. <= 5 in time 29
            2. <= 10 in time 30

Usage

To run it on Karate graph,
$ python expected_load_experiment.py -seeds_per_t 1

To run it on UIHC sampled graph,
$ python expected_load_experiment.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1

To run it on UIHC original graph,
$ python expected_load_experiment.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 
"""

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
import simulator_expected_load_sharing_temporal as expected_load_sharing
from approx_algorithms import *
from prep_GT_observation import *
from get_people_nodes import *
from prep_result_dataframes import *

import argparse
import pandas as pd
import random as random
import timeit

def prepare_result_dataframes():
    GT_n_S = np.sum(number_of_seeds_over_time)

    df_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC, GT_time_elapsed)
    df_BR = prepare_df_exp(BR_seeds_array, BR_n_S, BR_n_S_correct, \
            BR_TP, BR_TN, BR_FP, BR_FN, BR_F1, BR_MCC, BR_time_elapsed)

    df_G1 = prepare_df_exp(G1_seeds_array, G1_n_S, G1_n_S_correct, \
            G1_TP, G1_TN, G1_FP, G1_FN, G1_F1, G1_MCC, G1_time_elapsed)
    # i_ISCK is the iteration with the best MCC score for ISCK
    i_ISCK = np.argmax(np.array(L_ISCK_MCC))
    df_ISCK = prepare_df_exp(L_ISCK_seeds_array[i_ISCK], L_ISCK_n_S[i_ISCK], L_ISCK_n_S_correct[i_ISCK], \
            L_ISCK_TP[i_ISCK], L_ISCK_TN[i_ISCK], L_ISCK_FP[i_ISCK], L_ISCK_FN[i_ISCK], L_ISCK_F1[i_ISCK], L_ISCK_MCC[i_ISCK], ISCK_time_elapsed)

    i_MU_ISCK = np.argmax(np.array(L_MU_ISCK_MCC))
    df_MU_ISCK = prepare_df_exp(L_MU_ISCK_seeds_array[i_MU_ISCK], L_MU_ISCK_n_S[i_MU_ISCK], L_MU_ISCK_n_S_correct[i_MU_ISCK], \
            L_MU_ISCK_TP[i_MU_ISCK], L_MU_ISCK_TN[i_MU_ISCK], L_MU_ISCK_FP[i_MU_ISCK], L_MU_ISCK_FN[i_MU_ISCK], L_MU_ISCK_F1[i_MU_ISCK], L_MU_ISCK_MCC[i_MU_ISCK], MU_ISCK_time_elapsed)

    df_GR = prepare_df_exp(GR_seeds_array, GR_n_S, GR_n_S_correct, \
            GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC, GR_time_elapsed)

    #--------------------------------------------------------------------------------------------
    # Lazy
    df_lazy_G1 = prepare_df_exp(lazy_G1_seeds_array, lazy_G1_n_S, lazy_G1_n_S_correct, \
            lazy_G1_TP, lazy_G1_TN, lazy_G1_FP, lazy_G1_FN, lazy_G1_F1, lazy_G1_MCC, lazy_G1_time_elapsed)
    # i_lazy_ISCK is the iteration with the best MCC score for lazy_ISCK
    i_lazy_ISCK = np.argmax(np.array(L_lazy_ISCK_MCC))
    df_lazy_ISCK = prepare_df_exp(L_lazy_ISCK_seeds_array[i_lazy_ISCK], L_lazy_ISCK_n_S[i_lazy_ISCK], L_lazy_ISCK_n_S_correct[i_lazy_ISCK], \
            L_lazy_ISCK_TP[i_lazy_ISCK], L_lazy_ISCK_TN[i_lazy_ISCK], L_lazy_ISCK_FP[i_lazy_ISCK], L_lazy_ISCK_FN[i_lazy_ISCK], L_lazy_ISCK_F1[i_lazy_ISCK], L_lazy_ISCK_MCC[i_lazy_ISCK], lazy_ISCK_time_elapsed)

    i_lazy_MU_ISCK = np.argmax(np.array(L_lazy_MU_ISCK_MCC))
    df_lazy_MU_ISCK = prepare_df_exp(L_lazy_MU_ISCK_seeds_array[i_lazy_MU_ISCK], L_lazy_MU_ISCK_n_S[i_lazy_MU_ISCK], L_lazy_MU_ISCK_n_S_correct[i_lazy_MU_ISCK], \
            L_lazy_MU_ISCK_TP[i_lazy_MU_ISCK], L_lazy_MU_ISCK_TN[i_lazy_MU_ISCK], L_lazy_MU_ISCK_FP[i_lazy_MU_ISCK], L_lazy_MU_ISCK_FN[i_lazy_MU_ISCK], L_lazy_MU_ISCK_F1[i_lazy_MU_ISCK], L_lazy_MU_ISCK_MCC[i_lazy_MU_ISCK], lazy_MU_ISCK_time_elapsed)

    df_lazy_GR = prepare_df_exp(lazy_GR_seeds_array, lazy_GR_n_S, lazy_GR_n_S_correct, \
            lazy_GR_TP, lazy_GR_TN, lazy_GR_FP, lazy_GR_FN, lazy_GR_F1, lazy_GR_MCC, lazy_GR_time_elapsed)

    #--------------------------------------------------------------------------------------------
    # Expected
    df_E_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            E_GT_TP, E_GT_TN, E_GT_FP, E_GT_FN, E_GT_F1, E_GT_MCC, E_GT_time_elapsed)
    df_E_G1 = prepare_df_exp(E_G1_seeds_array, E_G1_n_S, E_G1_n_S_correct, \
            E_G1_TP, E_G1_TN, E_G1_FP, E_G1_FN, E_G1_F1, E_G1_MCC, E_G1_time_elapsed)
    # E_i_ISCK is the iteration with the best MCC score for E_ISCK
    i_E_ISCK = np.argmax(np.array(L_E_ISCK_MCC))
    df_E_ISCK = prepare_df_exp(L_E_ISCK_seeds_array[i_E_ISCK], L_E_ISCK_n_S[i_E_ISCK], L_E_ISCK_n_S_correct[i_E_ISCK], \
            L_E_ISCK_TP[i_E_ISCK], L_E_ISCK_TN[i_E_ISCK], L_E_ISCK_FP[i_E_ISCK], L_E_ISCK_FN[i_E_ISCK], L_E_ISCK_F1[i_E_ISCK], L_E_ISCK_MCC[i_E_ISCK], E_ISCK_time_elapsed)

    i_E_MU_ISCK = np.argmax(np.array(L_E_MU_ISCK_MCC))
    df_E_MU_ISCK = prepare_df_exp(L_E_MU_ISCK_seeds_array[i_E_MU_ISCK], L_E_MU_ISCK_n_S[i_E_MU_ISCK], L_E_MU_ISCK_n_S_correct[i_E_MU_ISCK], \
            L_E_MU_ISCK_TP[i_E_MU_ISCK], L_E_MU_ISCK_TN[i_E_MU_ISCK], L_E_MU_ISCK_FP[i_E_MU_ISCK], L_E_MU_ISCK_FN[i_E_MU_ISCK], L_E_MU_ISCK_F1[i_E_MU_ISCK], L_E_MU_ISCK_MCC[i_E_MU_ISCK], E_MU_ISCK_time_elapsed)

    df_E_GR = prepare_df_exp(E_GR_seeds_array, E_GR_n_S, E_GR_n_S_correct, \
            E_GR_TP, E_GR_TN, E_GR_FP, E_GR_FN, E_GR_F1, E_GR_MCC, E_GR_time_elapsed)

    return df_GT, df_BR, df_G1, df_ISCK, df_MU_ISCK, df_GR, \
            df_lazy_G1, df_lazy_ISCK, df_lazy_MU_ISCK, df_lazy_GR, \
            df_E_GT, df_E_G1, df_E_ISCK, df_E_MU_ISCK, df_E_GR

def prepare_ISCK_dataframes_per_iteration():
    # ISCK
    list_of_df_ISCK = []
    for i in range(len(L_ISCK_seeds_array)):
        list_of_df_ISCK.append(prepare_df_exp(L_ISCK_seeds_array[i], L_ISCK_n_S[i], L_ISCK_n_S_correct[i], \
                L_ISCK_TP[i], L_ISCK_TN[i], L_ISCK_FP[i], L_ISCK_FN[i], L_ISCK_F1[i], L_ISCK_MCC[i], ISCK_time_elapsed))

    # Lazy ISCK
    list_of_df_lazy_ISCK = []
    for i in range(len(L_lazy_ISCK_seeds_array)):
        list_of_df_lazy_ISCK.append(prepare_df_exp(L_lazy_ISCK_seeds_array[i], L_lazy_ISCK_n_S[i], L_lazy_ISCK_n_S_correct[i], \
                L_lazy_ISCK_TP[i], L_lazy_ISCK_TN[i], L_lazy_ISCK_FP[i], L_lazy_ISCK_FN[i], L_lazy_ISCK_F1[i], L_lazy_ISCK_MCC[i], lazy_ISCK_time_elapsed))

    # Expected Lazy ISCK
    list_of_df_E_ISCK = []
    for i in range(len(L_E_ISCK_seeds_array)):
        list_of_df_E_ISCK.append(prepare_df_exp(L_E_ISCK_seeds_array[i], L_E_ISCK_n_S[i], L_E_ISCK_n_S_correct[i], \
                L_E_ISCK_TP[i], L_E_ISCK_TN[i], L_E_ISCK_FP[i], L_E_ISCK_FN[i], L_E_ISCK_F1[i], L_E_ISCK_MCC[i], E_ISCK_time_elapsed))

    #################################################
    # multiplicative updates
    # MU_ISCK
    list_of_df_MU_ISCK = []
    for i in range(len(L_MU_ISCK_seeds_array)):
        list_of_df_MU_ISCK.append(prepare_df_exp(L_MU_ISCK_seeds_array[i], L_MU_ISCK_n_S[i], L_MU_ISCK_n_S_correct[i], \
                L_MU_ISCK_TP[i], L_MU_ISCK_TN[i], L_MU_ISCK_FP[i], L_MU_ISCK_FN[i], L_MU_ISCK_F1[i], L_MU_ISCK_MCC[i], MU_ISCK_time_elapsed))

    # Lazy MU_ISCK
    list_of_df_lazy_MU_ISCK = []
    for i in range(len(L_lazy_MU_ISCK_seeds_array)):
        list_of_df_lazy_MU_ISCK.append(prepare_df_exp(L_lazy_MU_ISCK_seeds_array[i], L_lazy_MU_ISCK_n_S[i], L_lazy_MU_ISCK_n_S_correct[i], \
                L_lazy_MU_ISCK_TP[i], L_lazy_MU_ISCK_TN[i], L_lazy_MU_ISCK_FP[i], L_lazy_MU_ISCK_FN[i], L_lazy_MU_ISCK_F1[i], L_lazy_MU_ISCK_MCC[i], lazy_MU_ISCK_time_elapsed))

    # Expected Lazy MU_ISCK
    list_of_df_E_MU_ISCK = []
    for i in range(len(L_E_MU_ISCK_seeds_array)):
        list_of_df_E_MU_ISCK.append(prepare_df_exp(L_E_MU_ISCK_seeds_array[i], L_E_MU_ISCK_n_S[i], L_E_MU_ISCK_n_S_correct[i], \
                L_E_MU_ISCK_TP[i], L_E_MU_ISCK_TN[i], L_E_MU_ISCK_FP[i], L_E_MU_ISCK_FN[i], L_E_MU_ISCK_F1[i], L_E_MU_ISCK_MCC[i], E_MU_ISCK_time_elapsed))

    return pd.concat(list_of_df_ISCK), pd.concat(list_of_df_lazy_ISCK), pd.concat(list_of_df_E_ISCK), \
            pd.concat(list_of_df_MU_ISCK), pd.concat(list_of_df_lazy_MU_ISCK), pd.concat(list_of_df_E_MU_ISCK)

def save_result_dataframes(name, k_total):
    # Save datasets
    df_GT.to_csv("../tables/expected_load/{}/k{}/GT.csv".format(name, k_total), index=False)
    df_BR.to_csv("../tables/expected_load/{}/k{}/BR.csv".format(name, k_total), index=False)
    df_G1.to_csv("../tables/expected_load/{}/k{}/G1.csv".format(name, k_total), index=False)
    df_ISCK.to_csv("../tables/expected_load/{}/k{}/ISCK.csv".format(name, k_total), index=False)
    df_MU_ISCK.to_csv("../tables/expected_load/{}/k{}/MU_ISCK.csv".format(name, k_total), index=False)
    df_GR.to_csv("../tables/expected_load/{}/k{}/GR.csv".format(name, k_total), index=False)

    df_lazy_G1.to_csv("../tables/expected_load/{}/k{}/lazy_G1.csv".format(name, k_total), index=False)
    df_lazy_ISCK.to_csv("../tables/expected_load/{}/k{}/lazy_ISCK.csv".format(name, k_total), index=False)
    df_lazy_MU_ISCK.to_csv("../tables/expected_load/{}/k{}/lazy_MU_ISCK.csv".format(name, k_total), index=False)
    df_lazy_GR.to_csv("../tables/expected_load/{}/k{}/lazy_GR.csv".format(name, k_total), index=False)

    df_E_GT.to_csv("../tables/expected_load/{}/k{}/E_GT.csv".format(name, k_total), index=False)
    df_E_G1.to_csv("../tables/expected_load/{}/k{}/E_G1.csv".format(name, k_total), index=False)
    df_E_ISCK.to_csv("../tables/expected_load/{}/k{}/E_ISCK.csv".format(name, k_total), index=False)
    df_E_MU_ISCK.to_csv("../tables/expected_load/{}/k{}/E_MU_ISCK.csv".format(name, k_total), index=False)
    df_E_GR.to_csv("../tables/expected_load/{}/k{}/E_GR.csv".format(name, k_total), index=False)

    df_result.to_csv("../tables/expected_load/{}/k{}/result_concat.csv".format(name, k_total), index=True)
    df_ground_truth_observations.to_csv("../tables/expected_load/{}/k{}/GT_observations.csv".format(name, k_total), index=False)

    df_ISCK_results.to_csv("../tables/expected_load/{}/k{}/ISCK_over_time.csv".format(name, k_total), index=False)
    df_lazy_ISCK_results.to_csv("../tables/expected_load/{}/k{}/lazy_ISCK_over_time.csv".format(name, k_total), index=False)
    df_E_ISCK_results.to_csv("../tables/expected_load/{}/k{}/E_ISCK_over_time.csv".format(name, k_total), index=False)
    df_MU_ISCK_results.to_csv("../tables/expected_load/{}/k{}/MU_ISCK_over_time.csv".format(name, k_total), index=False)
    df_lazy_MU_ISCK_results.to_csv("../tables/expected_load/{}/k{}/lazy_MU_ISCK_over_time.csv".format(name, k_total), index=False)
    df_E_MU_ISCK_results.to_csv("../tables/expected_load/{}/k{}/E_MU_ISCK_over_time.csv".format(name, k_total), index=False)


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
    parser.add_argument('-n_ISCK_iter', '--n_ISCK_iter', type=int, default=10,
                        help= 'Number of iterations for ISCK')
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
        array_of_knapsack_constraints_on_f[-1] = 1 #2
        array_of_knapsack_constraints_on_f[-2] = 0.5 #1
    else:
        array_of_knapsack_constraints_on_f[-1] = 1 #2
        array_of_knapsack_constraints_on_f[-2] = 0.5 #1

    ####################################################################
    # Additional input for problem 3
    array_of_penalty_on_f = np.zeros((n_timesteps))
    # array_of_penalty_on_f[-1] = 1
    # array_of_penalty_on_f[-2] = 2
    array_of_penalty_on_f[-1] = 5 #1
    array_of_penalty_on_f[-2] = 10 #2

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
    print("rho: {}".format(rho))
    print("d: {}".format(d))
    print("q: {}".format(q))
    print("pi: {}".format(pi))
    print("contact_area: {}".format(contact_area))
    simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)

    ####################################################################
    # NOTE: For all experiments, run it for n_replicates per seed set
    simul.set_n_replicates(n_replicates)
    ####################################################################
    # Set random seed, and observe infections
    # 1. Data generation
    print("Generate seed set w/ the best quality. Get ground truth observations...")
    seeds_array, obs_state, I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N \
            = prepare_GT_data(args, simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, args.GT_quality)

    ####################################################################
    # 2. Compute ground truth loss per timestep
    # We're not interested in loss over timestep (e.g. missing infection) in this project, so just take the loss at the last timestep.
    start = timeit.default_timer()
    print("Compute GT losses")
    GT_loss_1, GT_loss_total, \
        GT_list_of_P_hit, GT_list_of_N_hit, \
        GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC = \
        compute_GT_loss_per_timestep(simul, list_of_people_idx_arrays, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
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
    # Greedy source detection
    start = timeit.default_timer()
    print("-"*20)
    print("P1 Greedy")
    focus_obs1 = True
    G1_seeds_array, G1_n_S, G1_n_S_correct, G1_loss_1, G1_loss_total, \
        G1_list_of_P_hit, G1_list_of_N_hit, \
        G1_TP, G1_TN, G1_FP, G1_FN, G1_F1, G1_MCC = \
            run_greedy_source_detection_report_loss_per_timestep(simul, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=False)
    stop = timeit.default_timer()
    G1_time_elapsed = stop - start

    ####################################################################
    # ISCK
    # NOTE: setting the penalty array to all ones does not do anything.
    start = timeit.default_timer()
    print("-"*20)
    print("P2 ISCK - compute pi greedy")
    # NOTE: L_ denotes list of 
    # NOTE: compute_pi in ["greedy", "multiplicative_update"]
    L_ISCK_seeds_array, L_ISCK_n_S, L_ISCK_n_S_correct, L_ISCK_loss_1, L_ISCK_loss_total, \
        L_ISCK_list_of_P_hit, L_ISCK_list_of_N_hit, \
        L_ISCK_TP, L_ISCK_TN, L_ISCK_FP, L_ISCK_FN, L_ISCK_F1, L_ISCK_MCC = \
            run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                array_of_knapsack_constraints_on_f, flag_lazy=False, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="greedy")
    stop = timeit.default_timer()
    ISCK_time_elapsed = stop - start

    ####################################################################
    
    start = timeit.default_timer()
    print("-"*20)
    print("P2 ISCK - compute pi multiplicative update")
    L_MU_ISCK_seeds_array, L_MU_ISCK_n_S, L_MU_ISCK_n_S_correct, L_MU_ISCK_loss_1, L_MU_ISCK_loss_total, \
        L_MU_ISCK_list_of_P_hit, L_MU_ISCK_list_of_N_hit, \
        L_MU_ISCK_TP, L_MU_ISCK_TN, L_MU_ISCK_FP, L_MU_ISCK_FN, L_MU_ISCK_F1, L_MU_ISCK_MCC = \
            run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                array_of_knapsack_constraints_on_f, flag_lazy=False, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="multiplicative_update")

    stop = timeit.default_timer()
    MU_ISCK_time_elapsed = stop - start

    ####################################################################
    # Greedy ratio
    start = timeit.default_timer()
    print("-"*20)
    print("P3 Greedy Ratio")
    # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
    GR_seeds_array, GR_n_S, GR_n_S_correct, GR_loss_1, GR_loss_total, \
        GR_list_of_P_hit, GR_list_of_N_hit, \
        GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC = \
            run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=False)

    stop = timeit.default_timer()
    GR_time_elapsed = stop - start

    ####################################################################
    # Lazy implementations
    ####################################################################
    # Greedy source detection
    start = timeit.default_timer()
    print("-"*20)
    print("P1 Lazy Greedy")
    focus_obs1 = True
    lazy_G1_seeds_array, lazy_G1_n_S, lazy_G1_n_S_correct, lazy_G1_loss_1, lazy_G1_loss_total, \
        lazy_G1_list_of_P_hit, lazy_G1_list_of_N_hit, \
        lazy_G1_TP, lazy_G1_TN, lazy_G1_FP, lazy_G1_FN, lazy_G1_F1, lazy_G1_MCC = \
            run_greedy_source_detection_report_loss_per_timestep(simul, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=True)
    stop = timeit.default_timer()
    lazy_G1_time_elapsed = stop - start

    ####################################################################
    # ISCK
    # NOTE: setting the penalty array to all ones does not do anything.
    start = timeit.default_timer()
    print("-"*20)
    print("P2 Lazy ISCK")
    L_lazy_ISCK_seeds_array, L_lazy_ISCK_n_S, L_lazy_ISCK_n_S_correct, L_lazy_ISCK_loss_1, L_lazy_ISCK_loss_total, \
        L_lazy_ISCK_list_of_P_hit, L_lazy_ISCK_list_of_N_hit, \
        L_lazy_ISCK_TP, L_lazy_ISCK_TN, L_lazy_ISCK_FP, L_lazy_ISCK_FN, L_lazy_ISCK_F1, L_lazy_ISCK_MCC = \
            run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                array_of_knapsack_constraints_on_f, flag_lazy=True, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="greedy")
    stop = timeit.default_timer()
    lazy_ISCK_time_elapsed = stop - start

    ####################################################################
    start = timeit.default_timer()
    print("-"*20)
    print("P2 Lazy ISCK - compute pi multiplicative update")
    L_lazy_MU_ISCK_seeds_array, L_lazy_MU_ISCK_n_S, L_lazy_MU_ISCK_n_S_correct, L_lazy_MU_ISCK_loss_1, L_lazy_MU_ISCK_loss_total, \
        L_lazy_MU_ISCK_list_of_P_hit, L_lazy_MU_ISCK_list_of_N_hit, \
        L_lazy_MU_ISCK_TP, L_lazy_MU_ISCK_TN, L_lazy_MU_ISCK_FP, L_lazy_MU_ISCK_FN, L_lazy_MU_ISCK_F1, L_lazy_MU_ISCK_MCC = \
            run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                array_of_knapsack_constraints_on_f, flag_lazy=True, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="multiplicative_update")

    stop = timeit.default_timer()
    lazy_MU_ISCK_time_elapsed = stop - start
    ####################################################################
    # Greedy ratio
    start = timeit.default_timer()
    print("-"*20)
    print("P3 Lazy Greedy Ratio")
    # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
    lazy_GR_seeds_array, lazy_GR_n_S, lazy_GR_n_S_correct, lazy_GR_loss_1, lazy_GR_loss_total, \
        lazy_GR_list_of_P_hit, lazy_GR_list_of_N_hit, \
        lazy_GR_TP, lazy_GR_TN, lazy_GR_FP, lazy_GR_FN, lazy_GR_F1, lazy_GR_MCC = \
            run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=True)

    stop = timeit.default_timer()
    lazy_GR_time_elapsed = stop - start

    ####################################################################
    # Expected load
    ####################################################################
    print("Same set of experiments, but on the expected loads")
    expected_simul = expected_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    expected_simul.set_n_replicates(1)
    expected_simul.set_n_t_for_eval(n_t_for_eval)

    ####################################################################
    start = timeit.default_timer()
    print("Compute expected GT losses")
    E_GT_loss_1, E_GT_loss_total, \
        E_GT_list_of_P_hit, E_GT_list_of_N_hit, \
        E_GT_TP, E_GT_TN, E_GT_FP, E_GT_FN, E_GT_F1, E_GT_MCC = \
        compute_GT_loss_per_timestep(expected_simul, list_of_people_idx_arrays, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
    stop = timeit.default_timer()
    E_GT_time_elapsed = stop - start

    ####################################################################
    # 3. Greedy source detection
    start = timeit.default_timer()
    print("-"*20)
    print("P1 Expected Lazy Greedy")
    focus_obs1 = True
    E_G1_seeds_array, E_G1_n_S, E_G1_n_S_correct, E_G1_loss_1, E_G1_loss_total, \
        E_G1_list_of_P_hit, E_G1_list_of_N_hit, \
        E_G1_TP, E_G1_TN, E_G1_FP, E_G1_FN, E_G1_F1, E_G1_MCC = \
            run_greedy_source_detection_report_loss_per_timestep(expected_simul, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=True)
    stop = timeit.default_timer()
    E_G1_time_elapsed = stop - start

    ####################################################################
    # 4. ISCK
    # NOTE: setting the penalty array to all ones does not do anything.
    start = timeit.default_timer()
    print("-"*20)
    print("P2 Expected Lazy ISCK - compute pi - greedy")
    L_E_ISCK_seeds_array, L_E_ISCK_n_S, L_E_ISCK_n_S_correct, L_E_ISCK_loss_1, L_E_ISCK_loss_total, \
        L_E_ISCK_list_of_P_hit, L_E_ISCK_list_of_N_hit, \
        L_E_ISCK_TP, L_E_ISCK_TN, L_E_ISCK_FP, L_E_ISCK_FN, L_E_ISCK_F1, L_E_ISCK_MCC = \
            run_ISCK_report_loss_per_timestep(expected_simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                array_of_knapsack_constraints_on_f, flag_lazy=True, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="greedy")
    stop = timeit.default_timer()
    E_ISCK_time_elapsed = stop - start
    ####################################################################
    start = timeit.default_timer()
    print("-"*20)
    print("P2 Expected Lazy ISCK - compute pi - multiplicative update")
    L_E_MU_ISCK_seeds_array, L_E_MU_ISCK_n_S, L_E_MU_ISCK_n_S_correct, L_E_MU_ISCK_loss_1, L_E_MU_ISCK_loss_total, \
        L_E_MU_ISCK_list_of_P_hit, L_E_MU_ISCK_list_of_N_hit, \
        L_E_MU_ISCK_TP, L_E_MU_ISCK_TN, L_E_MU_ISCK_FP, L_E_MU_ISCK_FN, L_E_MU_ISCK_F1, L_E_MU_ISCK_MCC = \
            run_ISCK_report_loss_per_timestep(expected_simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                array_of_knapsack_constraints_on_f, flag_lazy=True, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="multiplicative_update")
    stop = timeit.default_timer()
    E_MU_ISCK_time_elapsed = stop - start

    ####################################################################
    # Greedy ratio
    start = timeit.default_timer()
    print("-"*20)
    print("P3 Expected Lazy Greedy Ratio")
    # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
    E_GR_seeds_array, E_GR_n_S, E_GR_n_S_correct, E_GR_loss_1, E_GR_loss_total, \
        E_GR_list_of_P_hit, E_GR_list_of_N_hit, \
        E_GR_TP, E_GR_TN, E_GR_FP, E_GR_FN, E_GR_F1, E_GR_MCC = \
            run_greedy_ratio_report_loss_per_timestep(expected_simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=True)

    stop = timeit.default_timer()
    E_GR_time_elapsed = stop - start

    ####################################################################

    df_GT, df_BR, df_G1, df_ISCK, df_MU_ISCK, df_GR, \
        df_lazy_G1, df_lazy_ISCK, df_lazy_MU_ISCK, df_lazy_GR, \
        df_E_GT, df_E_G1, df_E_ISCK, df_E_MU_ISCK, df_E_GR = prepare_result_dataframes()

    index_list = ["GT", "Random", "P1-Greedy", "P2-ISCK-greedy", "P2-ISCK-multiplicateive_update", "P3-GreedyRatio", \
                    "P1-LazyGreedy", "P2-LazyISCK-greedy", "P2-LazyISCK-multiplicateive_update", "P3-LazyGreedyRatio", \
                    "E_GT", "P1-E_LazyGreedy", "P2-E_LazyISCK-greedy", "P2-E_LazyISCK-multiplicateive_update", "P3-E_LazyGreedyRatio"]
    list_of_df = [df_GT, df_BR, df_G1, df_ISCK, df_MU_ISCK, df_GR, \
                    df_lazy_G1, df_lazy_ISCK, df_lazy_MU_ISCK, df_lazy_GR, \
                    df_E_GT, df_E_G1, df_E_ISCK, df_E_MU_ISCK, df_E_GR
            ]
    df_result = concat_result_dataframes(index_list, list_of_df)

    df_ground_truth_observations = prepare_ground_truth_table(sum(number_of_seeds_over_time), seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)

    # Prepare ISCK results over iteration
    df_ISCK_results, df_lazy_ISCK_results, df_E_ISCK_results, \
            df_MU_ISCK_results, df_lazy_MU_ISCK_results, df_E_MU_ISCK_results \
            = prepare_ISCK_dataframes_per_iteration()
    print("\nISCK results per iteration")
    print(df_ISCK_results.round(2))
    print("\nlazy_ISCK results per iteration")
    print(df_lazy_ISCK_results.round(2))
    print("\nE_ISCK results per iteration")
    print(df_E_ISCK_results.round(2))
    print("\nMU_ISCK results per iteration")
    print(df_MU_ISCK_results.round(2))
    print("\nlazy_MU_ISCK results per iteration")
    print(df_lazy_MU_ISCK_results.round(2))
    print("\nE_MU_ISCK results per iteration")
    print(df_E_MU_ISCK_results.round(2))

    print("\nGround Truth observations. Seeds and number of observed outcomes at last n timesteps")
    print(df_ground_truth_observations)

    # Print result tables
    # print_result_dataframes()
    print_result_dataframes(df_ground_truth_observations, df_result)

    # Save the result tables
    save_result_dataframes(graph_name, k_total)

