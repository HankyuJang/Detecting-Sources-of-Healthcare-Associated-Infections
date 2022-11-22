"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script generates and saves ground truth seeds and the observations
+ Compute the quality of the ground truth seed


Usage

To run it on UIHC sampled graph,
$ python bad_quality_prep_GT.py -GT_quality bad -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1
$ python bad_quality_prep_GT.py -GT_quality bad -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

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
                        help= 'dose-response function. exponential | linear')
    parser.add_argument('-GT_quality', '--GT_quality', type=str, default="bad",
                        help= 'Quality of the ground truth simulation. best | median. Always use best')
    parser.add_argument('-seeds_per_t', '--seeds_per_t', type=int, default=1,
                        help= 'number of seeds per timestep')
    parser.add_argument('-n_t_seeds', '--n_t_seeds', type=int, default=2,
                        help= 'number of timesteps for seeds')
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    parser.add_argument('-n_timesteps', '--n_timesteps', type=int, default=31,
                        help= 'n_timesteps. default it 31. set it to 10 for G_UVA experiments')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    ####################################################################
    # Parameters for the simulation. These are same regardless of the graph
    # n_timesteps = 31
    n_timesteps = args.n_timesteps
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

    print("Initiate simul instance")
    simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    print("simul. complete.")

    ####################################################################
    # NOTE: For all experiments, run it for n_replicates per seed set
    simul.set_n_replicates(n_replicates)
    ####################################################################
    # Set random seed, and observe infections
    # 1. Data generation
    print("Generate seed set w/ the BAD quality. Get ground truth observations...")
    seeds_array, obs_state, I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N \
            = prepare_GT_data(args, simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, args.GT_quality)

    print("list_of_sets_of_P at T: {}".format(list_of_sets_of_P[T]))

    for t, P in enumerate(list_of_sets_of_P):
        print("time: {}, P: {}".format(t, P))

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

    print("GT TP score: {}".format(GT_TP))
    print("GT MCC score: {}".format(GT_MCC))

    ####################################################################
    # Save everything as a pickle object.
    # Store everything in a dictionary.
    GT_observation_dict = dict()

    GT_observation_dict["name"] = args.name
    GT_observation_dict["year"] = args.year
    GT_observation_dict["sampled"] = args.sampled
    GT_observation_dict["dose_response"] = args.dose_response
    GT_observation_dict["GT_quality"] = args.GT_quality
    GT_observation_dict["seeds_per_t"] = args.seeds_per_t
    GT_observation_dict["n_t_seeds"] = args.n_t_seeds
    GT_observation_dict["n_t_for_eval"] = args.n_t_for_eval
    GT_observation_dict["n_timesteps"] = n_timesteps
    GT_observation_dict["n_replicates"] = n_replicates
    GT_observation_dict["area_people"] = area_people
    GT_observation_dict["area_location"] = area_location
    GT_observation_dict["T"] = T
    GT_observation_dict["flag_increase_area"] = flag_increase_area
    GT_observation_dict["number_of_seeds_over_time"] = number_of_seeds_over_time
    GT_observation_dict["k_total"] = k_total
    GT_observation_dict["node_name_to_idx_mapping"] = node_name_to_idx_mapping
    GT_observation_dict["node_idx_to_name_mapping"] = node_idx_to_name_mapping
    GT_observation_dict["list_of_people_idx_arrays"] = list_of_people_idx_arrays
    GT_observation_dict["list_of_sets_of_V"] = list_of_sets_of_V
    GT_observation_dict["seeds_array"] = seeds_array
    GT_observation_dict["obs_state"] = obs_state
    GT_observation_dict["I1"] = I1
    GT_observation_dict["MCC_array"] = MCC_array
    GT_observation_dict["list_of_sets_of_P"] = list_of_sets_of_P
    GT_observation_dict["list_of_sets_of_N"] = list_of_sets_of_N

    GT_evaluation_dict = dict()
    GT_evaluation_dict["loss_1"] = GT_loss_1
    GT_evaluation_dict["loss_total"] = GT_loss_total
    GT_evaluation_dict["list_of_P_hit"] = GT_list_of_P_hit
    GT_evaluation_dict["list_of_N_hit"] = GT_list_of_N_hit
    GT_evaluation_dict["TP"] = GT_TP
    GT_evaluation_dict["TN"] = GT_TN
    GT_evaluation_dict["FP"] = GT_FP
    GT_evaluation_dict["FN"] = GT_FN
    GT_evaluation_dict["F1"] = GT_F1
    GT_evaluation_dict["MCC"] = GT_MCC
    GT_evaluation_dict["time_elapsed"] = GT_time_elapsed

    GT_output_dict = {
            "GT_observation_dict": GT_observation_dict,
            "GT_evaluation_dict": GT_evaluation_dict,
            }

    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    if args.dose_response == "exponential":
        outfile = "GT_observation_evalution.pickle"
    elif args.dose_response == "linear":
        outfile = "linear_GT_observation_evalution.pickle"

    with open(path + outfile, "wb") as handle:
        pickle.dump(GT_output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


