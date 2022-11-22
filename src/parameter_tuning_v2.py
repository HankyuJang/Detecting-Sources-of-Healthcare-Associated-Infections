"""
Author: 
Email: 
Last Modified: Dec, 2021 

Description: This script is for parameter tuning only
Goal is to observe 5 - 10 % infections at the end of one-month period

Usage

$ python parameter_tuning_v2.py -name Karate_temporal -seeds_per_t 1
$ python parameter_tuning_v2.py -name Karate_temporal -seeds_per_t 2
$ python parameter_tuning_v2.py -name Karate_temporal -seeds_per_t 3

$ python parameter_tuning_v2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1
$ python parameter_tuning_v2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 2
$ python parameter_tuning_v2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

$ python parameter_tuning_v2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -seeds_per_t 1
$ python parameter_tuning_v2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -seeds_per_t 2
$ python parameter_tuning_v2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -seeds_per_t 3
"""
from utils.load_network import *
from simulator_load_sharing_temporal_v2 import *
from get_seeds import *
from get_people_nodes import *
from prep_GT_observation import get_P_N_over_time

import argparse
import pandas as pd
import math
import random as random
import copy
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
    parser.add_argument('-seeds_per_t', '--seeds_per_t', type=int, default=1,
                        help= 'number of seeds per timestep')
    parser.add_argument('-n_t_seeds', '--n_t_seeds', type=int, default=2,
                        help= 'number of timesteps for seeds')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)
    ####################################################################
    # Parameters for the simulation (not for tuning)
    n_timesteps = 31
    area_people = 2000 # area of patient. 2000cm^2
    area_location = 40000 # area of room. 40000cm^2
    n_replicates = 10 # number of simulations on the same seed
    n_exp = 10 # select 10 different starting seed sets

    # Parameters to tune
    rho = 0.4
    d = 0.1
    q = 8
    pi = 1.0
    contact_area = 150

    # Parameters for experiments
    T = n_timesteps-1 # T is the index of the last timestep
    flag_increase_area = True # If this is set to True, then increase area of each node based on their max degree over grpahs
    ####################################################################
    number_of_seeds_over_time = np.zeros((n_timesteps)).astype(int)
    for t in range(args.n_t_seeds):
        number_of_seeds_over_time[t] = args.seeds_per_t

    k_total = np.sum(number_of_seeds_over_time)
    print("Set number of seeds at various timesteps\ntime 0: 1 seed\ntime 1: 1 seed")
    print("number_of_seeds_over_time: {}\n".format(number_of_seeds_over_time))

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
    simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response, n_replicates)

    rho_list = []
    d_list = []
    q_list = []
    pi_list = []
    contact_area_list = []

    min_avg_inf_cnt_list = []
    avg_avg_inf_cnt_list = []
    max_avg_inf_cnt_list = []

    min_avg_non_inf_cnt_list = []
    avg_avg_non_inf_cnt_list = []
    max_avg_non_inf_cnt_list = []

    avg_recover_event_cnt_list = []

    # NOTE: for each simulation setting, these arrays holds P and N at T at each replicate
    len_P_array = np.zeros((n_replicates)).astype(int)
    len_N_array = np.zeros((n_replicates)).astype(int)

    # parameter_list_contact_area = [150, 500, 1000, 1500, 2000]
    # parameter_list_rho = [0.1, 0.3, 0.5]
    # parameter_list_d = [0.1, 0.3, 0.5]
    # parameter_list_q = [0.01, 0.03, 0.1, 0.3, 1, 3, 9]
    # parameter_list_pi = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

    parameter_list_contact_area = [2000*pow(math.e, -i) for i in range(5)] # [2000.0, 735.7588823428847, 270.6705664732254]
    parameter_list_rho = [pow(math.e, -i) for i in range(5)] # [1.0, 0.36787944117144233, 0.1353352832366127]
    parameter_list_d = [pow(math.e, -i) for i in range(5)] # [1.0, 0.36787944117144233, 0.1353352832366127]
    parameter_list_q = [pow(math.e, i) for i in range(3)] + [pow(math.e, -i) for i in range(1, 3)] # [1.0, 2.718281828459045, 7.3890560989306495] + ..
    parameter_list_pi = [pow(math.e, -i) for i in range(5)] # [1.0, 0.36787944117144233, 0.1353352832366127]

    total_combinations_of_parameters = len(parameter_list_contact_area) * len(parameter_list_rho) * len(parameter_list_d) * \
                                        len(parameter_list_q) * len(parameter_list_pi)
    # NOTE: Need to reconstruct BplusD whenever 'contact_area', 'rho', 'd' change
    loop_idx=0
    for contact_area in parameter_list_contact_area:
        simul.contact_area = contact_area
        for rho in parameter_list_rho:
            simul.rho = rho
            for d in parameter_list_d:
                simul.d = d

                # with these three paremeters, the set of parameters may lead to negative load. Then, skip.
                if not 0 <= simul.d + simul.rho * (simul.contact_area / np.min(simul.area_array)) <= 1:
                    continue

                # NOTE: Need to reconstruct BplusD whenever 'contact_area', 'rho', 'd' change
                simul.BplusD_over_time = simul.constructBplusD_over_time(G_over_time)
                for q in parameter_list_q:
                    simul.q = q
                    for pi in parameter_list_pi:
                        simul.pi = pi

                        print("{}/{}...".format(loop_idx, total_combinations_of_parameters), end='\r', flush=True)
                        loop_idx+=1

                        avg_inf_cnt_array = np.zeros((n_exp))
                        avg_non_inf_cnt_array = np.zeros((n_exp))
                        # NOTE: recover_event_cnt_array the counts here are summation over such events at any time an over 10 replicates
                        recover_event_cnt_array = np.zeros((n_exp))
                        # print("Start simulation")
                        for i in range(n_exp):

                            # Initilaize 2-d seeds_array
                            seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
                            # Set seeds at multiple timesteps
                            list_of_seed_idx_arrays = get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time)
                            for t, seed_idx_array in enumerate(list_of_seed_idx_arrays):
                                seeds_array[t, seed_idx_array] = True

                            simul.set_seeds(seeds_array)
                            simul.simulate()

                            # NOTE: these arrays are 3-d. (num_rep, timestep, nodes)
                            infection_array = simul.infection_array
                            #---------------------------------------------------
                            # 1. Get avg infection count
                            # Get P and N (ground truth infections and non-infections) over time
                            for rep_num in range(n_replicates):
                                list_of_sets_of_P, list_of_sets_of_N = get_P_N_over_time(infection_array, rep_num, list_of_sets_of_V)
                                len_P_T = len(list_of_sets_of_P[T]) # number of infected patients at T
                                len_N_T = len(list_of_sets_of_N[T]) # number of uninfected patients at T
                                len_P_array[rep_num] = len_P_T
                                len_N_array[rep_num] = len_N_T

                            # Get number of infected patients per 10 replicate of simulation
                            avg_inf_cnt = np.mean(len_P_array)
                            avg_non_inf_cnt = np.mean(len_N_array)

                            avg_inf_cnt_array[i] = avg_inf_cnt
                            avg_non_inf_cnt_array[i] = avg_non_inf_cnt
                            #---------------------------------------------------
                            # 2. Get counts of 'RECOVERED' events
                            recover_event_cnt = 0
                            for t in range(n_timesteps-1):
                                recover_event_cnt += ((infection_array[:,t,:].astype(int) - infection_array[:,t+1,:].astype(int)) == 1).sum()
                            # this recover event cnt array holds mean recover event counts over 10 replicates of simulations
                            recover_event_cnt_array[i] = recover_event_cnt / n_replicates

                            # print("Seeds: {}, avg: {:.2f}, len(P) array: {}".format(S_original, avg_inf_cnt, len_P_array))

                        # print()
                        # print("network: {}".format(name))
                        # print("T: {}, rho: {:.2f}, d: {:.2f}, contact_area: {}, q: {:.2f}, pi: {:.2f}".format(n_timesteps, rho, d, contact_area, q, pi))
                        # print("Avg(avg): {:.2f}, Max(avg): {:.2f}".format(np.mean(avg_inf_cnt_array), np.max(avg_inf_cnt_array)))

                        min_avg_inf_cnt = np.min(avg_inf_cnt_array)
                        avg_avg_inf_cnt = np.mean(avg_inf_cnt_array)
                        max_avg_inf_cnt = np.max(avg_inf_cnt_array)

                        min_avg_non_inf_cnt = np.min(avg_non_inf_cnt_array)
                        avg_avg_non_inf_cnt = np.mean(avg_non_inf_cnt_array)
                        max_avg_non_inf_cnt = np.max(avg_non_inf_cnt_array)

                        avg_recover_event_cnt = np.mean(recover_event_cnt_array)

                        # append current parameters, results to lists
                        rho_list.append(rho)
                        d_list.append(d)
                        q_list.append(q)
                        pi_list.append(pi)
                        contact_area_list.append(contact_area)

                        min_avg_inf_cnt_list.append(min_avg_inf_cnt)
                        avg_avg_inf_cnt_list.append(avg_avg_inf_cnt)
                        max_avg_inf_cnt_list.append(max_avg_inf_cnt)

                        min_avg_non_inf_cnt_list.append(min_avg_non_inf_cnt)
                        avg_avg_non_inf_cnt_list.append(avg_avg_non_inf_cnt)
                        max_avg_non_inf_cnt_list.append(max_avg_non_inf_cnt)

                        avg_recover_event_cnt_list.append(avg_recover_event_cnt)

                # NOTE: Save whenever 'contact_area', 'rho', 'd' change
                # Save dataframes every 
                df_results = pd.DataFrame(data={
                    "trans-eff": rho_list,
                    "die-off": d_list,
                    "shedding": q_list,
                    "infectivity": pi_list,
                    "A(contact)": contact_area_list,
                    "P_T(min)": min_avg_inf_cnt_list,
                    "P_T(avg)": avg_avg_inf_cnt_list,
                    "P_T(max)": max_avg_inf_cnt_list,
                    "N_T(min)": min_avg_non_inf_cnt_list,
                    "N_T(avg)": avg_avg_non_inf_cnt_list,
                    "N_T(max)": max_avg_non_inf_cnt_list,
                    "Recover_cnt(avg)": avg_recover_event_cnt_list
                    })
                print(df_results)

                # Save intermediary results
                df_results.to_csv("../tables/parameter_tuning/{}/k{}_parameter_tuning.csv".format(graph_name, k_total), index=False)

    # Save final results
    df_results.to_csv("../tables/parameter_tuning/{}/k{}_parameter_tuning.csv".format(graph_name, k_total), index=False)
