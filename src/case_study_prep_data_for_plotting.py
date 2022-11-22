"""
Author: -
Email: -
Last Modified: Feb 2022

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
    list_of_tuple_of_seeds = [(0, 753), (1, 409)]

    seeds_2d_bool = prep_seeds_2d_bool_array(simul, list_of_tuple_of_seeds)
    simul.set_seeds(seeds_2d_bool)

    ####################################################################
    # 
    n_replicates = 1000

    simul.set_n_replicates(n_replicates)
    simul.simulate()
    
    timestep_list = [0, 1, 2, T-2, T-1, T]
    GT_P_list = [6, 45, 559, 654]
    GT_P_set = set(GT_P_list)

    for simul_idx in range(n_replicates):
        # simul_idx = 0

        infection_array = simul.infection_array

        list_of_sets_of_P_from_our_seeds, list_of_sets_of_N_from_our_seeds = get_P_N_over_time(infection_array, simul_idx, list_of_sets_of_V[:n_timesteps])
        list_of_sets_of_P_for_plotting = [P for t, P in enumerate(list_of_sets_of_P_from_our_seeds) if t in timestep_list]

        P = list_of_sets_of_P_for_plotting[-1].union(list_of_sets_of_P_for_plotting[-2])
        P_overlap_w_GT_infection = P.intersection(GT_P_set)
        print("simul_idx: {}: P: {}, Overlap: {}".format(simul_idx, P, P_overlap_w_GT_infection))
        if len(P_overlap_w_GT_infection) >= 3:
            break

    print(list_of_sets_of_P_for_plotting)

    load_array_at_idx = simul.load_array[simul_idx, :, :]
    load_array_for_plotting = load_array_at_idx[timestep_list, :][:, GT_P_list]
    df_load = pd.DataFrame(data=load_array_for_plotting, index=timestep_list, columns=GT_P_list)
    print(df_load)

    outpath = "../tables/case_study/"
    outfile = "load.csv"
    df_load.to_csv(outpath+outfile, index=True)

    outfile = "P_over_time.pkl"
    with open(outpath + outfile, "wb") as handle:
        pickle.dump(list_of_sets_of_P_for_plotting, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dict_for_casestudy = {
            "simul_idx": simul_idx,
            "timestep_list": timestep_list,
            "GT_P_list": GT_P_list,
            "list_of_tuple_of_seeds": list_of_tuple_of_seeds,
            "infection_array": infection_array,
            "load_array": simul.load_array,
            "list_of_sets_of_P_for_plotting": list_of_sets_of_P_for_plotting
            }
    outfile = "dict_for_casestudy.pkl"
    with open(outpath + outfile, "wb") as handle:
        pickle.dump(dict_for_casestudy, handle, protocol=pickle.HIGHEST_PROTOCOL)

