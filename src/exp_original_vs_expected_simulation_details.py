"""
Author: -
Email: -
Last Modified: Jan, 2022

Description: 

    Original vs expected simulation

    Compare F1, MCC

Usage

To run it on Karate graph,
$ python exp_original_vs_expected_simulation_details.py -seeds_per_t 1
$ python exp_original_vs_expected_simulation_details.py -seeds_per_t 3

To run it on UIHC sampled graph,
$ python exp_original_vs_expected_simulation_details.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1
$ python exp_original_vs_expected_simulation_details.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

#------------------------------------------

Dose response = linear

To run it on Karate graph,
$ python exp_original_vs_expected_simulation_details.py -seeds_per_t 1 -dose_response linear
$ python exp_original_vs_expected_simulation_details.py -seeds_per_t 3 -dose_response linear

To run it on UIHC sampled graph,
$ python exp_original_vs_expected_simulation_details.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1 -dose_response linear
$ python exp_original_vs_expected_simulation_details.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3 -dose_response linear

"""

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
import simulator_expected_load_sharing_temporal_v2 as expected_load_sharing
import simulator_truncated_expected_load_sharing_temporal as truncated_expected_load_sharing
from approx_algorithms import *
from prep_GT_observation import *
from get_people_nodes import *
from prep_result_dataframes import *

import argparse
import pandas as pd
import random as random
import timeit
from tqdm import tqdm

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
                        help= 'Quality of the ground truth simulation. best | median | any')
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
    # Ground truth seeds over time
    number_of_seeds_over_time = np.zeros((n_timesteps)).astype(int)
    for t in range(args.n_t_seeds):
        number_of_seeds_over_time[t] = args.seeds_per_t

    k_total = np.sum(number_of_seeds_over_time)
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
    simul.set_n_replicates(n_replicates)

    expected_simul = expected_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response, n_replicates=n_replicates, n_t_for_original_simulation=n_t_for_eval)

    truncated_expected_simul = truncated_expected_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response, n_replicates=1, n_t_for_eval=n_t_for_eval, truncate_threshold=0.01)

    ####################################################################
    # Repeat GT seed selection and the following experiments for 10 times
    n_experiment_repeat = 10
    # n_experiment_repeat = 1
    GT_TP_experiment_repeat_array = np.zeros((n_experiment_repeat))
    GT_FP_experiment_repeat_array = np.zeros((n_experiment_repeat))
    GT_time_experiment_repeat_array = np.zeros((n_experiment_repeat))
    GT_MCC_experiment_repeat_array = np.zeros((n_experiment_repeat))
    GT_F1_experiment_repeat_array = np.zeros((n_experiment_repeat))

    E_th05_GT_TP_experiment_repeat_array = np.zeros((n_experiment_repeat))
    E_th05_GT_FP_experiment_repeat_array = np.zeros((n_experiment_repeat))
    E_th05_GT_time_experiment_repeat_array = np.zeros((n_experiment_repeat))
    E_th05_GT_MCC_experiment_repeat_array = np.zeros((n_experiment_repeat))
    E_th05_GT_F1_experiment_repeat_array = np.zeros((n_experiment_repeat))

    for experiment_repeat_idx in tqdm(range(n_experiment_repeat)):
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
        # Expected load - truncated
        ####################################################################
        print("Same set of experiments, but on the expected truncated loads")
        ####################################################################
        threshold = 0.05
        truncated_expected_simul.set_truncate_threshold(threshold)
        E_th05_GT_TP_array = np.zeros((n_replicates))
        E_th05_GT_FP_array = np.zeros((n_replicates))
        E_th05_GT_MCC_array = np.zeros((n_replicates))
        E_th05_GT_F1_array = np.zeros((n_replicates))

        print("Compute expected GT losses. Threshold: {}".format(threshold))
        for expected_idx in range(n_replicates):
            start = timeit.default_timer()
            E_th05_GT_loss_1, E_th05_GT_loss_total, \
                E_th05_GT_list_of_P_hit, E_th05_GT_list_of_N_hit, \
                E_th05_GT_TP, E_th05_GT_TN, E_th05_GT_FP, E_th05_GT_FN, E_th05_GT_F1, E_th05_GT_MCC = \
                compute_GT_loss_per_timestep(truncated_expected_simul, list_of_people_idx_arrays, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
            stop = timeit.default_timer()
            E_th05_GT_time_elapsed = stop - start

            # Given the seedset, simply re-do the evaluation using original simulation from the detected seeds
            # simul.set_seeds(seeds_array)

            # E_th05_GT_loss_1, E_th05_GT_loss_total, \
                # E_th05_GT_list_of_P_hit, E_th05_GT_list_of_N_hit, \
                # E_th05_GT_TP, E_th05_GT_TN, E_th05_GT_FP, E_th05_GT_FN, E_th05_GT_F1, E_th05_GT_MCC = \
                # compute_GT_loss_per_timestep(simul, list_of_people_idx_arrays, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
            E_th05_GT_TP_array[expected_idx] = E_th05_GT_TP
            E_th05_GT_FP_array[expected_idx] = E_th05_GT_FP
            E_th05_GT_MCC_array[expected_idx] = E_th05_GT_MCC
            E_th05_GT_F1_array[expected_idx] = E_th05_GT_F1
            ####################################################################

        E_th05_GT_TP_mean = np.mean(E_th05_GT_TP_array)
        E_th05_GT_FP_mean = np.mean(E_th05_GT_FP_array)
        E_th05_GT_MCC_mean = np.mean(E_th05_GT_MCC_array)
        E_th05_GT_F1_mean = np.mean(E_th05_GT_F1_array)
        ####################################################################

        GT_TP_experiment_repeat_array[experiment_repeat_idx] = GT_TP
        GT_FP_experiment_repeat_array[experiment_repeat_idx] = GT_FP
        GT_time_experiment_repeat_array[experiment_repeat_idx] = GT_time_elapsed
        GT_MCC_experiment_repeat_array[experiment_repeat_idx] = GT_MCC
        GT_F1_experiment_repeat_array[experiment_repeat_idx] = GT_F1

        E_th05_GT_TP_experiment_repeat_array[experiment_repeat_idx] = E_th05_GT_TP_mean
        E_th05_GT_FP_experiment_repeat_array[experiment_repeat_idx] = E_th05_GT_FP_mean
        E_th05_GT_time_experiment_repeat_array[experiment_repeat_idx] = E_th05_GT_time_elapsed
        E_th05_GT_MCC_experiment_repeat_array[experiment_repeat_idx] = E_th05_GT_MCC_mean
        E_th05_GT_F1_experiment_repeat_array[experiment_repeat_idx] = E_th05_GT_F1_mean

        # compute this table in each epoch
        df_result = pd.DataFrame(data={
            "GT_TP": GT_TP_experiment_repeat_array,
            "GT_FP": GT_FP_experiment_repeat_array,
            "E_th05_GT_TP": E_th05_GT_TP_experiment_repeat_array,
            "E_th05_GT_FP": E_th05_GT_FP_experiment_repeat_array,
            "original_time": GT_time_experiment_repeat_array,
            "E_thres_0.05_time": E_th05_GT_time_experiment_repeat_array,
            "original_MCC": GT_MCC_experiment_repeat_array,
            "original_F1": GT_F1_experiment_repeat_array,
            "E_thres_0.05_MCC": E_th05_GT_MCC_experiment_repeat_array,
            "E_thres_0.05_F1": E_th05_GT_F1_experiment_repeat_array,
            })

        df_result["original"] = df_result["GT_TP"] + df_result["GT_FP"]
        df_result["E_thres_0.05"] = df_result["E_th05_GT_TP"] + df_result["E_th05_GT_FP"]

        print(df_result[["original", "E_thres_0.05",
        # "original_time", "E_thres_0.05_time",
        "original_MCC", "E_thres_0.05_MCC",
        "original_F1", "E_thres_0.05_F1"]])

    k_total = np.sum(number_of_seeds_over_time)
    path = "../tables/exp_original_vs_expected_simulation/"

    if args.dose_response == "exponential":
        outfile = "EVAL_{}_k{}.csv".format(graph_name, k_total)
    elif args.dose_response == "linear":
        outfile = "EVAL_linear_{}_k{}.csv".format(graph_name, k_total)

    df_result[["original", "E_thres_0.05", \
            "original_time", "E_thres_0.05_time", \
            "original_MCC", "E_thres_0.05_MCC", \
            "original_F1", "E_thres_0.05_F1"
            ]].to_csv(path + outfile, index=False)

