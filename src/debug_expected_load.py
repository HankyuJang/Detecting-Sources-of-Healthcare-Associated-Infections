"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

expected load sharing model debugging

Usage

To run it on Karate graph,
$ python debug_expected_load.py -seeds_per_t 1

To run it on UIHC sampled graph,
$ python debug_expected_load.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1

To run it on UIHC original graph,
$ python debug_expected_load.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -seeds_per_t 1
"""

from utils.load_network import *
from utils.set_parameters import *
# import simulator_load_sharing_temporal_v2 as load_sharing
# import simulator_load_sharing_temporal_sparse as load_sharing
import simulator_expected_load_sharing_temporal_v2 as load_sharing
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

    print("initializing simul obj...")
    start = timeit.default_timer()
    # simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response, n_replicates=n_replicates, n_t_for_original_simulation=10)
    stop = timeit.default_timer()
    duration = stop - start
    print("initialization complete. duration {}s".format(duration))

    ####################################################################
    # NOTE: For all experiments, run it for n_replicates per seed set
    simul.set_n_replicates(n_replicates)

    list_of_seed_idx_arrays = get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time)
    print("list_of_seed_idx_arrays: {}".format(list_of_seed_idx_arrays))

    seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    for t, seed_idx_array in enumerate(list_of_seed_idx_arrays):
        seeds_array[t, seed_idx_array] = True

    simul.set_seeds(seeds_array)

    start = timeit.default_timer()
    simul.simulate()
    stop = timeit.default_timer()
    duration = stop - start
    print("Simulation duration: {}s".format(duration))
