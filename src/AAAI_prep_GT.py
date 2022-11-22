"""
Author: -
Email: -
Last Modified: July 2022

Description: 

This script generates and saves ground truth seeds and the observations
+ Compute the quality of the ground truth seed

"""

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
import simulator_by_second as non_discrete_load_sharing
import simulator_by_second_detailed as non_discrete_load_sharing_detailed
from prep_GT_observation import *
from get_people_nodes import *
from approx_algorithms import *

import argparse
import pandas as pd
import random as random
import timeit
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def compute_avg_contact_duration(G_over_time):
    total_days = len(G_over_time)
    G = G_over_time[0]
    n_edges = G.number_of_edges()
    duration_array = np.zeros(n_edges)

    for i, e in enumerate(G.edges()):
        duration = 0
        attrs = G.edges[e]
        for start in attrs:
            end = attrs[start]
            duration += (int(end)-int(start))
        duration_array[i] = duration
    
    print(duration_array)
    return np.mean(duration_array)

def compute_incoming_node_and_load_arrays(simul):
    
    n_nodes = simul.G_over_time[0].number_of_nodes()

    unique_incoming_node_array = np.zeros((n_nodes))
    total_incoming_load_array = np.zeros((n_nodes))

    for dst in simul.incoming_load_track_dict:
        dst_idx = simul.nodename_to_idx_mapping[dst]

        n_unique_incoming_nodes = len(simul.incoming_load_track_dict[dst])
        if n_unique_incoming_nodes == 0:
            continue
        unique_incoming_node_array[dst_idx] = n_unique_incoming_nodes
        
        total_incoming_load = 0
        for src in simul.incoming_load_track_dict[dst]:
            total_incoming_load += simul.incoming_load_track_dict[dst][src]
        total_incoming_load_array[dst_idx] = total_incoming_load
    
    return unique_incoming_node_array, total_incoming_load_array

# either node or load
def plot_incoming(x, ylabel, out_folder, filename):
    plt.plot(x, marker='o', linestyle="None", markersize=3)
    plt.xlabel("node index")
    plt.ylabel(ylabel)
    plt.savefig("{}/{}".format(out_folder, filename))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='greedy source detection, missing infection')
    parser.add_argument('-name', '--name', type=str, default="G_Carilion",
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
    parser.add_argument('-n_timesteps', '--n_timesteps', type=int, default=31,
                        help= 'n_timesteps. default it 31. set it to 10 for G_UVA experiments')
    parser.add_argument('-infection_threshold', '--infection_threshold', type=float, default=0.5,
                        help= 'threhsold value to determine if a patient is infected or not in the day.')
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

    ####################################################################
    # Ground truth seeds over time
    number_of_seeds_over_time = np.zeros((n_timesteps)).astype(int)
    for t in range(args.n_t_seeds):
        number_of_seeds_over_time[t] = args.seeds_per_t

    k_total = np.sum(number_of_seeds_over_time)
    print("number_of_seeds_over_time: {}\n".format(number_of_seeds_over_time))

    ####################################################################
    print("Load network...\n")
    flag_increase_area = True # If this is set to True, then increase area of each node based on their max degree over grpahs
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, graph_name = process_data_for_experiments(args, area_people, area_location, flag_increase_area)

    # avg_contact_duration_in_seconds = compute_avg_contact_duration(G_over_time)
    # avg_contact_duration_in_minutes = avg_contact_duration_in_seconds / 60
    # print("avg contact duration in minutes: {}".format(avg_contact_duration_in_minutes))

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

    print("Initiate simul instance")
    ####################################################################
    simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    print("day-level simulation parameters")
    print("rho: {}".format(rho))
    print("d: {}".format(d))
    print("q: {}".format(q))
    print("pi: {}".format(pi))

    rho /= (24*60/5)
    d /= (24*60/5)
    q /= (24*60/5)
    pi /= (24*60/5)

    ####################################################################
    # For minute level simulation, do not increase surface area
    flag_increase_area = False # If this is set to True, then increase area of each node based on their max degree over grpahs
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, graph_name = process_data_for_experiments(args, area_people, area_location, flag_increase_area)
    print("minute-level simulation parameters")
    print("rho: {}".format(rho))
    print("d: {}".format(d))
    print("q: {}".format(q))
    print("pi: {}".format(pi))
    print("contact_area: {}".format(contact_area))

    # simul = non_discrete_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
    simul_minute_level = non_discrete_load_sharing_detailed.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response, args.infection_threshold)
    print("simul. complete.")

    ####################################################################
    # NOTE: For all experiments, run it for n_replicates per seed set
    # n_replicates=1
    # simul_minute_level.set_n_replicates(n_replicates)

    # # Initilaize 2-d seeds_array
    # print("simulate once. Granularity by day")
    # seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    # # Set seeds at multiple timesteps
    # random.seed(123) # ensure same sources are selected for each run (for threshold exploration)
    # list_of_seed_idx_arrays = get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time)
    # print("Seeds: {}".format(list_of_seed_idx_arrays))

    # for t, seed_idx_array in enumerate(list_of_seed_idx_arrays):
    #     seeds_array[t, seed_idx_array] = True

    # simul.set_seeds(seeds_array)
    # simul.simulate()

    # unique_incoming_node_array, total_incoming_load_array = compute_incoming_node_and_load_arrays(simul)
    # plot_incoming(unique_incoming_node_array, "unique_incoming_node_count", "../plots/AAAI", "{}_unique_incoming_node_count.png".format(args.name))
    # plot_incoming(total_incoming_load_array,  "total_incoming_load", "../plots/AAAI", "{}_total_incoming_load_count.png".format(args.name))

    ####################################################################

    # Set random seed, and observe infections
    # 1. Data generation
    print("Generate seed set w/ the best quality. Get ground truth observations...")
    seeds_array, obs_state, I1, list_of_sets_of_P, list_of_sets_of_N \
            = prepare_GT_data(args, simul_minute_level, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, args.GT_quality)
            # = prepare_GT_data(args, simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, args.GT_quality)

    print("list_of_sets_of_P at T: {}".format(list_of_sets_of_P[T]))

    for t, P in enumerate(list_of_sets_of_P):
        print("time: {}, P: {}".format(t, P))

    ###################################################################
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
    # GT_observation_dict["MCC_array"] = MCC_array
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

    path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    if args.dose_response == "exponential":
        outfile = "GT_observation_evalution.pickle"
    elif args.dose_response == "linear":
        outfile = "linear_GT_observation_evalution.pickle"

    with open(path + outfile, "wb") as handle:
        pickle.dump(GT_output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


