"""
Author: -
Email: -
Last Modified: Dec, 2021 

Description: 

Script for debugging

- Debugging Greedy Ratio

$ python debug_experiments.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

To run it on UIHC original graph,
$ python debug_experiments.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 
"""

from utils.load_network import *
from utils.set_parameters import *
from simulator_load_sharing_temporal_v2 import *
import simulator_expected_load_sharing_temporal as expected_load_sharing
from approx_algorithms import *
from prep_GT_observation import *
from get_people_nodes import *
from prep_result_dataframes import *

import argparse
import random as random
import timeit


def prepare_result_dataframes():
    GT_n_S = np.sum(number_of_seeds_over_time)

    df_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC, GT_time_elapsed)
    df_BR = prepare_df_exp(BR_seeds_array, BR_n_S, BR_n_S_correct, \
            BR_TP, BR_TN, BR_FP, BR_FN, BR_F1, BR_MCC, BR_time_elapsed)
    # df_G1 = prepare_df_exp(G1_seeds_array, G1_n_S, G1_n_S_correct, \
            # G1_TP, G1_TN, G1_FP, G1_FN, G1_F1, G1_MCC, G1_time_elapsed)
    # i_ISCK is the iteration with the best MCC score
    # i_ISCK = np.argmax(np.array(MU_ISCK_MCC))
    # df_MU_ISCK = prepare_df_exp(MU_ISCK_seeds_array[i_ISCK], MU_ISCK_n_S[i_ISCK], MU_ISCK_n_S_correct[i_ISCK], \
            # MU_ISCK_TP[i_ISCK], MU_ISCK_TN[i_ISCK], MU_ISCK_FP[i_ISCK], MU_ISCK_FN[i_ISCK], MU_ISCK_F1[i_ISCK], MU_ISCK_MCC[i_ISCK], MU_ISCK_time_elapsed)

    # i_ISCK = np.argmax(np.array(MU_L_ISCK_MCC))
    # df_MU_L_ISCK = prepare_df_exp(MU_L_ISCK_seeds_array[i_ISCK], MU_L_ISCK_n_S[i_ISCK], MU_L_ISCK_n_S_correct[i_ISCK], \
            # MU_L_ISCK_TP[i_ISCK], MU_L_ISCK_TN[i_ISCK], MU_L_ISCK_FP[i_ISCK], MU_L_ISCK_FN[i_ISCK], MU_L_ISCK_F1[i_ISCK], MU_L_ISCK_MCC[i_ISCK], MU_L_ISCK_time_elapsed)

    # df_lazy_GR = prepare_df_exp(lazy_GR_seeds_array, lazy_GR_n_S, lazy_GR_n_S_correct, \
            # lazy_GR_TP, lazy_GR_TN, lazy_GR_FP, lazy_GR_FN, lazy_GR_F1, lazy_GR_MCC, lazy_GR_time_elapsed)
    # df_GR = prepare_df_exp(GR_seeds_array, GR_n_S, GR_n_S_correct, \
            # GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC, GR_time_elapsed)

    #--------------------------------------------------------------------------------------------
    # Expected
    df_E_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            E_GT_TP, E_GT_TN, E_GT_FP, E_GT_FN, E_GT_F1, E_GT_MCC, E_GT_time_elapsed)

    return df_GT, df_BR, df_E_GT

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
    simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)

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
    # 3. Greedy source detection
    # start = timeit.default_timer()
    # print("-"*20)
    # print("Run greedy source detection, compute loss per timestep for the best nodeset")
    # print("Lazy Greedy1: optimize on loss_1")
    # focus_obs1 = True
    # G1_seeds_array, G1_n_S, G1_n_S_correct, G1_loss_1, G1_loss_total, \
        # G1_list_of_P_hit, G1_list_of_N_hit, \
        # G1_TP, G1_TN, G1_FP, G1_FN, G1_F1, G1_MCC = \
            # run_greedy_source_detection_report_loss_per_timestep(simul, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                # seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=True)
    # stop = timeit.default_timer()
    # G1_time_elapsed = stop - start

    # ####################################################################
    # 4. ISCK
    # NOTE: setting the penalty array to all ones does not do anything.
    # start = timeit.default_timer()
    # print("-"*20)
    # print("Run ISCK, compute loss per timestep for the best nodeset")
    # print("ISCK - multiplicative update")

    # MU_ISCK_seeds_array, MU_ISCK_n_S, MU_ISCK_n_S_correct, MU_ISCK_loss_1, MU_ISCK_loss_total, \
        # MU_ISCK_list_of_P_hit, MU_ISCK_list_of_N_hit, \
        # MU_ISCK_TP, MU_ISCK_TN, MU_ISCK_FP, MU_ISCK_FN, MU_ISCK_F1, MU_ISCK_MCC = \
            # run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                # seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                # array_of_knapsack_constraints_on_f, flag_lazy=False, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="multiplicative_update")

    # stop = timeit.default_timer()
    # MU_ISCK_time_elapsed = stop - start

    # start = timeit.default_timer()
    # print("-"*20)
    # print("Run ISCK, compute loss per timestep for the best nodeset")
    # print("Lazy ISCK - multiplicative update")

    # MU_L_ISCK_seeds_array, MU_L_ISCK_n_S, MU_L_ISCK_n_S_correct, MU_L_ISCK_loss_1, MU_L_ISCK_loss_total, \
        # MU_L_ISCK_list_of_P_hit, MU_L_ISCK_list_of_N_hit, \
        # MU_L_ISCK_TP, MU_L_ISCK_TN, MU_L_ISCK_FP, MU_L_ISCK_FN, MU_L_ISCK_F1, MU_L_ISCK_MCC = \
            # run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                # seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                # array_of_knapsack_constraints_on_f, flag_lazy=True, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="multiplicative_update")

    # stop = timeit.default_timer()
    # MU_L_ISCK_time_elapsed = stop - start

    ####################################################################
    # Lazy Greedy ratio
    # start = timeit.default_timer()
    # print("-"*20)
    # print("Run Lazy greedy ratio, compute loss per timestep for the best nodeset")
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
    # print("Run greedy ratio, compute loss per timestep for the best nodeset")
    # # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
    # GR_seeds_array, GR_n_S, GR_n_S_correct, GR_loss_1, GR_loss_total, \
        # GR_list_of_P_hit, GR_list_of_N_hit, \
        # GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC = \
            # run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                # seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=False)

    # stop = timeit.default_timer()
    # GR_time_elapsed = stop - start

    ####################################################################
    # Expected load
    ####################################################################
    print("Same set of experiments, but on the expected loads")
    expected_simul = expected_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    expected_simul.set_n_replicates(1)
    expected_simul.set_n_t_for_eval(n_t_for_eval)

    start = timeit.default_timer()
    print("Compute expected GT losses")
    E_GT_loss_1, E_GT_loss_total, \
        E_GT_list_of_P_hit, E_GT_list_of_N_hit, \
        E_GT_TP, E_GT_TN, E_GT_FP, E_GT_FN, E_GT_F1, E_GT_MCC = \
        compute_GT_loss_per_timestep(expected_simul, list_of_people_idx_arrays, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
    stop = timeit.default_timer()
    E_GT_time_elapsed = stop - start

    # print("Infected nodes at T: {}".format(expected_simul.infection_array[0,-1,:]))
    # print("Infected nodes at T-1: {}".format(expected_simul.infection_array[0,-2,:]))

    ####################################################################
    # Generate result tables
    # df_GT, df_BR, df_G1, df_ISCK, df_GR = prepare_result_dataframes()
    # df_GT, df_BR, df_G1, df_lazy_GR, df_GR = prepare_result_dataframes()
    df_GT, df_BR, df_E_GT = prepare_result_dataframes()
    # df_result = concat_result_dataframes()
    # index_list = ["GT", "Random", "P2-MU_ISCK", "P2-MU_L_ISCK"]
    index_list = ["GT", "Random", "E_GT"]
    list_of_df = [df_GT, df_BR, df_E_GT]
    df_result = concat_result_dataframes(index_list, list_of_df)

    df_ground_truth_observations = prepare_ground_truth_table(sum(number_of_seeds_over_time), seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
    # Print result tables
    print_result_dataframes(df_ground_truth_observations, df_result)

    # Prepare ISCK results over iteration
    # df_ISCK_result_per_iteration = prepare_ISCK_dataframes_per_iteration()
    # print("\nISCK result per iteration")
    # print(df_ISCK_result_per_iteration)

    # # Save the result tables
    # save_result_dataframes(graph_name, k_total)

