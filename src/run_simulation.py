"""
Author: 
Email: 
Last Modified: Oct, 2021 

Description: This script is for checking behaviour of the simulator.

Usage

To run it on UIHC original graph,
$ python run_simulation.py -name Karate_temporal
$ python run_simulation.py -name UIHC_HCP_patient_room -year 2011 -dose_response exponential
$ python run_simulation.py -name UIHC_HCP_patient_room -year 2011 -sampled True -dose_response exponential

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
def get_people_array_in_day0(G, node_name_to_idx_mapping):
    day0_people_list = []
    for node_name, degree in G.degree:
        if degree > 0:
            node_idx = node_name_to_idx_mapping[node_name]
            if node_idx in people_nodes_idx:
                day0_people_list.append(node_idx)
    day0_people_array = np.array(day0_people_list)
    return day0_people_array 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='greedy source detection, missing infection')
    parser.add_argument('-name', '--name', type=str, default="Karate_temporal",
                        help= 'network to use. Karate_temporal | UIHC_Jan2010_patient_room_temporal | UIHC_HCP_patient_room | UVA_temporal')
    parser.add_argument('-year', '--year', type=int, default=2011,
                        help= '2007 | 2011')
    parser.add_argument('-sampled', '--sampled', type=bool, default=False,
                        help= 'set it True to use sampled data.')
    parser.add_argument('-dose_response', '--dose_response', type=str, default="exponential",
                        help= 'dose-response function')
    args = parser.parse_args()

    name = args.name
    year = args.year
    sampled = args.sampled
    dose_response = args.dose_response

    np.set_printoptions(suppress=True)
    n_timesteps = 31
    rho = 0.5
    d = 0
    q = 0.1
    pi = 0.0001
    contact_area = 2000
    area_people = 2000 # area of patient. 2000cm^2
    area_location = 40000 # area of room. 40000cm^2

    flag_increase_area = True # If this is set to True, then increase area of each node based on their max degree over grpahs
    n_replicates = 5 # number of simulations on the same seed
    n_replicates = 100 # number of simulations on the same seed
    n_exp = 5 # select 5 different starting seed sets
    k=1 # number of seeds

    ####################################################################
    print("Load network")
    if name == "Karate_temporal":
        # q = 2
        # pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_karate_temporal_network(area_people, area_location, flag_increase_area)
    elif name == "UVA_temporal":
        # q = 2
        # pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UVA_temporal_network(area_people, area_location, flag_increase_area)
    elif name == "UIHC_Jan2010_patient_room_temporal":
        # q = 2
        # pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UIHC_Jan2010_patient_room_temporal_network(area_people, area_location, flag_increase_area)
    elif name == "UIHC_HCP_patient_room":
        # q = 10
        # pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}

        if sampled:
            # contact_area = 10
            name = "{}_{}_sampled".format(name, year)
        else:
            # contact_area = 10
            name = "{}_{}".format(name, year)
        # if year = 2011 # Use non-overlap data.
        # if sampled = True # Use the subgraph. Sampled based on the unit with the most number of CDI cases.
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UIHC_HCP_patient_room_temporal_network(year, sampled, area_people, area_location, flag_increase_area)

    node_name_to_idx_mapping = dict([(node_name, node_idx) for node_idx, node_name in enumerate(G_over_time[0].nodes())])
    node_idx_to_name_mapping = dict([(node_idx, node_name) for node_idx, node_name in enumerate(G_over_time[0].nodes())])

    day0_people_idx_array = get_people_array_in_day0(G_over_time[0], node_name_to_idx_mapping)

    ####################################################################
    # 0. Create simulation instance with empty seeds list
    simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, dose_response, n_replicates)

    # min_avg_inf_cnt_list = []
    # avg_avg_inf_cnt_list = []
    # max_avg_inf_cnt_list = []
    # avg_recover_event_cnt_list = []
    avg_inf_cnt_array = np.zeros((n_exp))
    recover_event_cnt_array = np.zeros((n_exp))

    for i in range(n_exp):

        # S_original = np.random.choice(a=people_nodes_idx, size=k, replace=False)
        S_original = np.random.choice(a=day0_people_idx_array, size=k, replace=False)
        S_name = node_idx_to_name_mapping[S_original[0]]

        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        seeds_array[0, S_original] = True
        simul.set_seeds(seeds_array)
        simul.simulate()

        infection_array = simul.infection_array
        #---------------------------------------------------
        # 1. Get avg infection count
        # Get number of infected patients per simulation
        # len_P_array = infection_array[:,-1,:].sum(axis=1)
        len_P_array = np.sum(np.sum(infection_array, axis=1).astype(bool), axis=1)
        avg_inf_cnt = np.mean(len_P_array)
        avg_inf_cnt_array[i] = avg_inf_cnt
        #---------------------------------------------------
        # 2. Get counts of 'RECOVERED' events
        recover_event_cnt = 0
        for t in range(n_timesteps-1):
            recover_event_cnt += ((infection_array[:,t,:].astype(int) - infection_array[:,t+1,:].astype(int)) == 1).sum()
        recover_event_cnt_array[i] = recover_event_cnt




