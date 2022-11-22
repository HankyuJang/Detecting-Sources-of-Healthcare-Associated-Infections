"""
Author: -
Email: -
Last Modified: Aug 2022

Description: 

This script prepares data for plotting case study

Usage
$ python case_study_prep_data_for_plotting.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T

Our algorithms detected these as seeds:

    753 P2448785
    409 P606948

Ground truth infections at time 25 and 26 are these guys

    559
    45
    654
    6

Start simulation with these guys. Keep track of those infected over time. Also keep track of the loads on these 4 guys.

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
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='case study')
    parser.add_argument('-name', '--name', type=str, default="Karate_temporal",
                        help= 'network to use. Karate_temporal | UIHC_Jan2010_patient_room_temporal | UIHC_HCP_patient_room_withinHCPxPx | UVA_temporal')
    parser.add_argument('-year', '--year', type=int, default=2011,
                        help= '2007 | 2011')
    parser.add_argument('-sampled', '--sampled', type=bool, default=False,
                        help= 'set it True to use sampled data.')
    parser.add_argument('-dose_response', '--dose_response', type=str, default="exponential",
                        help= 'dose-response function')
    args = parser.parse_args()

    print("Load GT observations...\n")
    graph_name = get_graph_name(args)
    path = "../tables/case_study/{}/".format(graph_name)
    if args.dose_response == "exponential":
        infile = "GT_observation_evalution.pickle"
    elif args.dose_response == "linear":
        infile = "linear_GT_observation_evalution.pickle"
    with open(path + infile, 'rb') as handle:
        GT_output_dict = pickle.load(handle)

    n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

    print("list_of_sets_of_P at T: {}".format(list_of_sets_of_P[T]))

    print("Load network...\n")
    flag_casestudy=True
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, _ = process_data_for_experiments(args, area_people, area_location, flag_increase_area, flag_casestudy)

    ####################################################################
    # 0. Create simulation instance with empty seeds list
    rho, d, q, pi, contact_area = set_simulation_parameters(args, k_total)
    print("rho: {}".format(rho))
    print("d: {}".format(d))
    print("q: {}".format(q))
    print("pi: {}".format(pi))
    print("contact_area: {}".format(contact_area))

    simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    list_of_tuple_of_seeds = [(0, 753), (1, 409)]

    seeds_2d_bool = prep_seeds_2d_bool_array(simul, list_of_tuple_of_seeds)
    simul.set_seeds(seeds_2d_bool)

    ####################################################################
    # SEEDS = use those detected from our methods
    n_replicates = 1000

    simul.set_n_replicates(n_replicates)
    simul.simulate()
    
    timestep_list = [0, 1, 2, T-2, T-1, T]
    last_three_timesteps = [T-2, T-1, T]
    GT_P_list = [6, 45, 559, 654]
    GT_P_set = set(GT_P_list)

    observed_inf_cnt_dict = dict()
    for observed_inf in GT_P_list:
        observed_inf_cnt_dict[observed_inf] = 0
    unobserved_inf_cnt_dict = dict()

    for simul_idx in range(n_replicates):
        # simul_idx = 0

        infection_array = simul.infection_array
        list_of_sets_of_P_from_our_seeds, list_of_sets_of_N_from_our_seeds = get_P_N_over_time(infection_array, simul_idx, list_of_sets_of_V[:n_timesteps])
        list_of_sets_of_P_last_three_timesteps = [P for t, P in enumerate(list_of_sets_of_P_from_our_seeds) if t in last_three_timesteps]

        P = set()
        for P_t in list_of_sets_of_P_last_three_timesteps: 
            P.update(P_t)

        for node in P:
            if node in observed_inf_cnt_dict:
                observed_inf_cnt_dict[node] += 1
            elif node in unobserved_inf_cnt_dict:
                unobserved_inf_cnt_dict[node] += 1
            else:
                unobserved_inf_cnt_dict[node] = 1
    
    print("For each simulation replicate, increment a counter of a node if a node is infected at any time in the last 3 timesteps")
    print("Simulation, number of replicates: {}".format(n_replicates))

    print("Seeds from our methods. node: 753 at time 0, node: 409 at time 1")
    # print("\nobserved_inf_cnt_dict: {}".format(observed_inf_cnt_dict))
    # print("unobserved_inf_cnt_dict: {}".format(unobserved_inf_cnt_dict))
    print("Observed infection count sum over nodes: {}".format(sum([observed_inf_cnt_dict[key] for key in observed_inf_cnt_dict])))
    print("Unobserved infection count sum over nodes: {}".format(sum([unobserved_inf_cnt_dict[key] for key in unobserved_inf_cnt_dict])))
    print("\nObserved infection counts over 1000 replicates")
    print("node id: count")
    for key in observed_inf_cnt_dict:
        print("{}: {}".format(key, observed_inf_cnt_dict[key]))
    print("Unobserved infection counts over 1000 replicates")
    print("node id: count")
    for key in unobserved_inf_cnt_dict:
        print("{}: {}".format(key, unobserved_inf_cnt_dict[key]))


    ####################################################################
    # SEEDS = use infected node at 1. that is 214.
    list_of_tuple_of_seeds = [(1, 214)]
    seeds_2d_bool = prep_seeds_2d_bool_array(simul, list_of_tuple_of_seeds)
    simul.set_seeds(seeds_2d_bool)
    simul.simulate()

    observed_inf_cnt_dict_using_inf_at_1 = dict()
    for observed_inf in GT_P_list:
        observed_inf_cnt_dict_using_inf_at_1[observed_inf] = 0
    unobserved_inf_cnt_dict_using_inf_at_1 = dict()

    for simul_idx in range(n_replicates):
        # simul_idx = 0

        infection_array = simul.infection_array
        list_of_sets_of_P_from_our_seeds, list_of_sets_of_N_from_our_seeds = get_P_N_over_time(infection_array, simul_idx, list_of_sets_of_V[:n_timesteps])
        list_of_sets_of_P_last_three_timesteps = [P for t, P in enumerate(list_of_sets_of_P_from_our_seeds) if t in last_three_timesteps]

        P = set()
        for P_t in list_of_sets_of_P_last_three_timesteps: 
            P.update(P_t)

        for node in P:
            if node in observed_inf_cnt_dict_using_inf_at_1:
                observed_inf_cnt_dict_using_inf_at_1[node] += 1
            elif node in unobserved_inf_cnt_dict_using_inf_at_1:
                unobserved_inf_cnt_dict_using_inf_at_1[node] += 1
            else:
                unobserved_inf_cnt_dict_using_inf_at_1[node] = 1
    
    print("\nSeeds is CDI case at time 1. node: 214")
    # print("\nobserved_inf_cnt_dict_using_inf_at_1: {}".format(observed_inf_cnt_dict_using_inf_at_1))
    # print("unobserved_inf_cnt_dict_using_inf_at_1: {}".format(unobserved_inf_cnt_dict_using_inf_at_1))
    print("Observed infection count sum over nodes: {}".format(sum([observed_inf_cnt_dict_using_inf_at_1[key] for key in observed_inf_cnt_dict_using_inf_at_1])))
    print("Unobserved infection count sum over nodes: {}".format(sum([unobserved_inf_cnt_dict_using_inf_at_1[key] for key in unobserved_inf_cnt_dict_using_inf_at_1])))
    print("\nObserved infection counts over 1000 replicates")
    print("node id: count")
    for key in observed_inf_cnt_dict_using_inf_at_1:
        print("{}: {}".format(key, observed_inf_cnt_dict_using_inf_at_1[key]))
    print("Unobserved infection counts over 1000 replicates")
    print("node id: count")
    for key in unobserved_inf_cnt_dict_using_inf_at_1:
        print("{}: {}".format(key, unobserved_inf_cnt_dict_using_inf_at_1[key]))