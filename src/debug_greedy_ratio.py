"""
Author: -
Email: -
Last Modified: Jan, 2022

Description: 

    Debug greedyratio

Usage

To run it on Karate graph,
$ python debug_greedy_ratio.py -seeds_per_t 1

To run it on UIHC sampled graph,
$ python debug_greedy_ratio.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1

To run it on UIHC original graph,
$ python debug_greedy_ratio.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 
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

def initilize_n_empty_lists(n):
    list_to_return = []
    for i in range(n):
        list_to_return.append([])
    return list_to_return

def prepare_result_dataframes():
    GT_n_S = np.sum(number_of_seeds_over_time)

    df_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC, GT_time_elapsed)
    df_BR = prepare_df_exp(BR_seeds_array, BR_n_S, BR_n_S_correct, \
            BR_TP, BR_TN, BR_FP, BR_FN, BR_F1, BR_MCC, BR_time_elapsed)

    df_G1 = prepare_df_exp(G1_seeds_array, G1_n_S, G1_n_S_correct, \
            G1_TP, G1_TN, G1_FP, G1_FN, G1_F1, G1_MCC, G1_time_elapsed)

    #--------------------------------------------------------------------------------------------
    # Lazy
    df_lazy_G1 = prepare_df_exp(lazy_G1_seeds_array, lazy_G1_n_S, lazy_G1_n_S_correct, \
            lazy_G1_TP, lazy_G1_TN, lazy_G1_FP, lazy_G1_FN, lazy_G1_F1, lazy_G1_MCC, lazy_G1_time_elapsed)

    #--------------------------------------------------------------------------------------------
    # Expected
    df_E_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            E_GT_TP, E_GT_TN, E_GT_FP, E_GT_FN, E_GT_F1, E_GT_MCC, E_GT_time_elapsed)
    df_E_G1 = prepare_df_exp(E_G1_seeds_array, E_G1_n_S, E_G1_n_S_correct, \
            E_G1_TP, E_G1_TN, E_G1_FP, E_G1_FN, E_G1_F1, E_G1_MCC, E_G1_time_elapsed)

    return df_GT, df_BR, df_G1, \
            df_lazy_G1, \
            df_E_GT, df_E_G1


def save_result_dataframes(name, k_total, n_t_for_eval):
    # Save datasets
    df_GT.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_GT_v2.csv".format(name, k_total, n_t_for_eval), index=False)
    df_BR.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_BR_v2.csv".format(name, k_total, n_t_for_eval), index=False)
    df_G1.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_G1_v2.csv".format(name, k_total, n_t_for_eval), index=False)

    df_lazy_G1.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_lazy_G1_v2.csv".format(name, k_total, n_t_for_eval), index=False)

    df_E_GT.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_E_GT_v2.csv".format(name, k_total, n_t_for_eval), index=False)
    df_E_G1.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_E_G1_v2.csv".format(name, k_total, n_t_for_eval), index=False)

    df_greedy_ratio.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_greedy_ratio_v2.csv".format(name, k_total, n_t_for_eval), index=True)

    df_result.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_result_concat_v2.csv".format(name, k_total, n_t_for_eval), index=True)
    df_ground_truth_observations.to_csv("../tables/debug_greedy_ratio/{}/k{}/nteval{}_GT_observations_v2.csv".format(name, k_total, n_t_for_eval), index=False)

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
    ####################################################################
    # Greedy ratio
    ####################################################################
    L_penalty_array = []
    L_penalty_array_last_n_t_for_eval = []
    for i in range(1, 11):
        array_of_penalty_on_f = np.zeros((n_timesteps))
        # array_of_penalty_on_f[-1] = i
        # array_of_penalty_on_f[-2] = i*2
        array_of_penalty_on_f[-1] = pow(2, -i)
        array_of_penalty_on_f[-2] = pow(2, -i)*2
        L_penalty_array.append(array_of_penalty_on_f)
        L_penalty_array_last_n_t_for_eval.append(array_of_penalty_on_f[-n_t_for_eval:])
    ####################################################################
    # array_of_penalty_on_f = np.zeros((n_timesteps))
    # array_of_penalty_on_f[-1] = 5 #1
    # array_of_penalty_on_f[-2] = 10 #2
    ####################################################################
    L_GR_S_detected, L_GR_S_timesteps, L_GR_n_S, L_GR_n_S_correct, \
            L_GR_TP, L_GR_TN, L_GR_FP, L_GR_FN, L_GR_F1, L_GR_MCC, L_GR_time_elapsed = initilize_n_empty_lists(11)

    for array_of_penalty_on_f in L_penalty_array:
        start = timeit.default_timer()
        print("-"*20)
        print("Penalty array: {}".format(array_of_penalty_on_f))
        print("P3 Greedy Ratio")
        # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
        GR_seeds_array, GR_n_S, GR_n_S_correct, GR_loss_1, GR_loss_total, \
            GR_list_of_P_hit, GR_list_of_N_hit, \
            GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC = \
                run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                    seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=False)

        stop = timeit.default_timer()
        GR_time_elapsed = stop - start

        L_GR_S_detected.append(str(list(GR_seeds_array.nonzero()[1])))
        L_GR_S_timesteps.append(str(list(GR_seeds_array.nonzero()[0])))
        L_GR_n_S.append(GR_n_S)
        L_GR_n_S_correct.append(GR_n_S_correct)
        L_GR_TP.append(GR_TP)
        L_GR_TN.append(GR_TN)
        L_GR_FP.append(GR_FP)
        L_GR_FN.append(GR_FN)
        L_GR_F1.append(GR_F1)
        L_GR_MCC.append(GR_MCC)
        L_GR_time_elapsed.append(GR_time_elapsed)

        ####################################################################
        # Greedy ratio
        # start = timeit.default_timer()
        # print("-"*20)
        # print("P3 Lazy Greedy Ratio")
        # # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
        # lazy_GR_seeds_array, lazy_GR_n_S, lazy_GR_n_S_correct, lazy_GR_loss_1, lazy_GR_loss_total, \
            # lazy_GR_list_of_P_hit, lazy_GR_list_of_N_hit, \
            # lazy_GR_TP, lazy_GR_TN, lazy_GR_FP, lazy_GR_FN, lazy_GR_F1, lazy_GR_MCC = \
                # run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                    # seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=True)

        # stop = timeit.default_timer()
        # lazy_GR_time_elapsed = stop - start

        # ####################################################################
        # # Greedy ratio
        # start = timeit.default_timer()
        # print("-"*20)
        # print("P3 Expected Lazy Greedy Ratio")
        # # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
        # E_GR_seeds_array, E_GR_n_S, E_GR_n_S_correct, E_GR_loss_1, E_GR_loss_total, \
            # E_GR_list_of_P_hit, E_GR_list_of_N_hit, \
            # E_GR_TP, E_GR_TN, E_GR_FP, E_GR_FN, E_GR_F1, E_GR_MCC = \
                # run_greedy_ratio_report_loss_per_timestep(expected_simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                    # seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=True)

        # stop = timeit.default_timer()
        # E_GR_time_elapsed = stop - start
    ####################################################################

    df_GT, df_BR, df_G1, \
        df_lazy_G1, \
        df_E_GT, df_E_G1, = prepare_result_dataframes()

    index_list = ["GT", "Random", "P1-Greedy", \
                    "P1-LazyGreedy", \
                    "E_GT", "P1-E_LazyGreedy"]
    list_of_df = [df_GT, df_BR, df_G1, \
                    df_lazy_G1, \
                    df_E_GT, df_E_G1 
            ]
    df_result = concat_result_dataframes(index_list, list_of_df)

    df_greedy_ratio = pd.DataFrame({
        "L_penalty_array": L_penalty_array_last_n_t_for_eval,
        "S_detected": L_GR_S_detected,
        "S_timesteps": L_GR_S_timesteps,
        "n_S": L_GR_n_S,
        "n_S_correct": L_GR_n_S_correct,
        "TP": L_GR_TP,
        "TN": L_GR_TN,
        "FP": L_GR_FP,
        "FN": L_GR_FN,
        "F1": L_GR_F1,
        "MCC": L_GR_MCC,
        "Time(s)": L_GR_time_elapsed
        })

    df_ground_truth_observations = prepare_ground_truth_table(sum(number_of_seeds_over_time), seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)

    print("\nGround Truth observations. Seeds and number of observed outcomes at last n timesteps")
    print(df_ground_truth_observations)

    print_result_dataframes(df_ground_truth_observations, df_result)
    print(df_greedy_ratio)

    save_result_dataframes(graph_name, k_total, n_t_for_eval)

