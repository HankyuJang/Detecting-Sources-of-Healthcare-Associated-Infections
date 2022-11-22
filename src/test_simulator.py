"""
Author: 
Email: 
Last Modified: Nov, 2021 

Description: This script is for testing the simulator.

Test 1: A random source on time 0 - check load sharing

"""
from utils.load_network import *
from simulator_load_sharing_temporal_v2 import *

import pandas as pd
import argparse
import math
import random as random
import copy
import timeit
import numpy as np
from tqdm import tqdm

# Get people in day 0 w/ at least 1 neighbor
def get_people_array_in_day_t(G, node_name_to_idx_mapping):
    day_t_people_list = []
    for node_name, degree in G.degree:
        if degree > 0:
            node_idx = node_name_to_idx_mapping[node_name]
            if node_idx in people_nodes_idx:
                day_t_people_list.append(node_idx)
    day_t_people_array = np.array(day_t_people_list)
    return day_t_people_array 

def prepare_dataframes(people_nodes, simul):
    df_seed = pd.DataFrame(data=seeds_array.T.astype(int), index=people_nodes)
    df_infection = pd.DataFrame(data=simul.infection_array[0].T.astype(int), index=people_nodes)
    df_probability = pd.DataFrame(data=simul.probability_array[0].T, index=people_nodes)
    df_load = pd.DataFrame(data=simul.load_array[0].T, index=people_nodes)
    return df_seed, df_infection, df_probability, df_load

def save_dataframes(folder, df_seed, df_infection, df_probability, df_load):
    df_seed.to_csv("{}/seed.csv".format(folder))
    df_infection.to_csv("{}/infection.csv".format(folder))
    df_probability.to_csv("{}/probability.csv".format(folder))
    df_load.to_csv("{}/load.csv".format(folder))

def get_seeds_time_0_and_1(k, day0_people_idx_array, day1_people_idx_array):
    S_0 = np.random.choice(a=day0_people_idx_array, size=k, replace=False)
    set_S_0 = set(S_0)
    while True:
        S_1 = np.random.choice(a=day1_people_idx_array, size=k, replace=False)
        set_S_1 = set(S_1)
        if len(set_S_0.intersection(set_S_1)) == 0:
            break
    return S_0, S_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='greedy source detection, missing infection')
    parser.add_argument('-name', '--name', type=str, default="Karate_temporal",
                        help= 'network to use. Karate_temporal | UIHC_Jan2010_patient_room_temporal | UIHC_HCP_patient_room | UVA_temporal')
    parser.add_argument('-dose_response', '--dose_response', type=str, default="exponential",
                        help= 'dose-response function')
    args = parser.parse_args()

    name = args.name
    dose_response = args.dose_response

    np.set_printoptions(suppress=True)
    # n_timesteps = 31
    # rho = 0.5
    # d = 0
    # q = 0.1
    # pi = 0.0001
    contact_area = 100
    area_people = 100 # area of patient. 2000cm^2
    area_location = 100 # area of room. 40000cm^2

    flag_increase_area = True # If this is set to True, then increase area of each node based on their max degree over grpahs
    # n_replicates = 100 # number of simulations on the same seed
    # n_exp = 5 # select 5 different starting seed sets
    # k=1 # number of seeds

    ####################################################################
    print("Load Karate network")
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_karate_temporal_network(area_people, area_location, flag_increase_area)

    node_name_to_idx_mapping = dict([(node_name, node_idx) for node_idx, node_name in enumerate(G_over_time[0].nodes())])
    node_idx_to_name_mapping = dict([(node_idx, node_name) for node_idx, node_name in enumerate(G_over_time[0].nodes())])

    # These are used for setting seeds
    day0_people_idx_array = get_people_array_in_day_t(G_over_time[0], node_name_to_idx_mapping)
    day1_people_idx_array = get_people_array_in_day_t(G_over_time[1], node_name_to_idx_mapping)

    ####################################################################
    print("Test 1 - seed at time 0")
    folder="../test_simulator/test1"
    n_timesteps, n_replicates, k = 5, 1, 1
    rho, d, q, pi = 1, 0, 1, 1
    simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, dose_response, n_replicates)
    S_original = np.random.choice(a=day0_people_idx_array, size=k, replace=False)

    seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    seeds_array[0, S_original] = True
    simul.set_seeds(seeds_array)
    simul.simulate()

    df_seed, df_infection, df_probability, df_load = prepare_dataframes(people_nodes, simul)
    save_dataframes(folder, df_seed, df_infection, df_probability, df_load)
    
    ####################################################################
    print("Test 2 - seed at time 0 and 1")
    folder="../test_simulator/test2"

    n_timesteps, n_replicates, k = 5, 1, 1
    rho, d, q, pi = 1, 0, 1, 1
    simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, dose_response, n_replicates)

    S_0, S_1 = get_seeds_time_0_and_1(k, day0_people_idx_array, day1_people_idx_array)

    seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    seeds_array[0, S_0] = True
    seeds_array[1, S_1] = True
    simul.set_seeds(seeds_array)
    simul.simulate()

    df_seed, df_infection, df_probability, df_load = prepare_dataframes(people_nodes, simul)
    save_dataframes(folder, df_seed, df_infection, df_probability, df_load)

    ####################################################################

    # 0. Create simulation instance with empty seeds list
    # simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, dose_response, n_replicates)

    # # min_avg_inf_cnt_list = []
    # # avg_avg_inf_cnt_list = []
    # # max_avg_inf_cnt_list = []
    # # avg_recover_event_cnt_list = []
    # avg_inf_cnt_array = np.zeros((n_exp))
    # recover_event_cnt_array = np.zeros((n_exp))

    # for i in range(n_exp):

        # # S_original = np.random.choice(a=people_nodes_idx, size=k, replace=False)
        # S_original = np.random.choice(a=day0_people_idx_array, size=k, replace=False)
        # S_name = node_idx_to_name_mapping[S_original[0]]

        # seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        # seeds_array[0, S_original] = True
        # simul.set_seeds(seeds_array)
        # simul.simulate()

        # infection_array = simul.infection_array
        # #---------------------------------------------------
        # # 1. Get avg infection count
        # # Get number of infected patients per simulation
        # # len_P_array = infection_array[:,-1,:].sum(axis=1)
        # len_P_array = np.sum(np.sum(infection_array, axis=1).astype(bool), axis=1)
        # avg_inf_cnt = np.mean(len_P_array)
        # avg_inf_cnt_array[i] = avg_inf_cnt
        # #---------------------------------------------------
        # # 2. Get counts of 'RECOVERED' events
        # recover_event_cnt = 0
        # for t in range(n_timesteps-1):
            # recover_event_cnt += ((infection_array[:,t,:].astype(int) - infection_array[:,t+1,:].astype(int)) == 1).sum()
        # recover_event_cnt_array[i] = recover_event_cnt




