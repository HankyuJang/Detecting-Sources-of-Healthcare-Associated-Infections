"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads ground truth observations
and runs baseline methods

Usage


To run it on UIHC sampled graph,
$ python bad_quality_B.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

$ python bad_quality_B.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1
"""

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
from prep_GT_observation import *
from get_people_nodes import *
from approx_algorithms import *

import argparse
import pandas as pd
import random as random
import timeit
import pickle

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

    np.set_printoptions(suppress=True)

    print("Load GT observations...\n")
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
    print("Load network...\n")
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, _ = process_data_for_experiments(args, area_people, area_location, flag_increase_area)

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
    # Baselines

    # Repeat baseline experiment for 30 times
    B_random_evaluation_rep_dict = dict()
    n_repetition_for_B_random = 30

    for i in tqdm(range(n_repetition_for_B_random)):
        # Randomly selected seed out of people nodes
        start = timeit.default_timer()
        print("-"*20)
        print("Compute random baseline")
        B_random_seeds_array, B_random_n_S, B_random_n_S_correct, B_random_loss_1, B_random_loss_total, \
            B_random_list_of_P_hit, B_random_list_of_N_hit, \
            B_random_TP, B_random_TN, B_random_FP, B_random_FN, B_random_F1, B_random_MCC = \
                run_BR_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                                                seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval)
        stop = timeit.default_timer()
        B_random_time_elapsed = stop - start

        print("Baseline random TP score: {}".format(B_random_TP))
        print("Baseline random MCC score: {}".format(B_random_MCC))

        B_random_evaluation_dict = dict()
        B_random_evaluation_dict["seeds_array"] = B_random_seeds_array
        B_random_evaluation_dict["n_S"] = B_random_n_S
        B_random_evaluation_dict["n_S_correct"] = B_random_n_S_correct
        B_random_evaluation_dict["loss_1"] = B_random_loss_1
        B_random_evaluation_dict["loss_total"] = B_random_loss_total
        B_random_evaluation_dict["list_of_P_hit"] = B_random_list_of_P_hit
        B_random_evaluation_dict["list_of_N_hit"] = B_random_list_of_N_hit
        B_random_evaluation_dict["TP"] = B_random_TP
        B_random_evaluation_dict["TN"] = B_random_TN
        B_random_evaluation_dict["FP"] = B_random_FP
        B_random_evaluation_dict["FN"] = B_random_FN
        B_random_evaluation_dict["F1"] = B_random_F1
        B_random_evaluation_dict["MCC"] = B_random_MCC
        B_random_evaluation_dict["time_elapsed"] = B_random_time_elapsed

        B_random_evaluation_rep_dict["rep{}".format(i)] = B_random_evaluation_dict

    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)

    if args.dose_response == "exponential":
        outfile = "B_random_evalution_{}rep.pickle".format(n_repetition_for_B_random)
    elif args.dose_response == "linear":
        outfile = "linear_B_random_evalution_{}rep.pickle".format(n_repetition_for_B_random)

    with open(path + outfile, "wb") as handle:
        pickle.dump(B_random_evaluation_rep_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

