"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last Modified: Feb, 2020 

Description: functions for generating random seeds.
"""

from simulator_load_sharing import *
import numpy as np
import random as rd

# S_original_dict. key: k, value: numpy array of index of seeds.
# E.g. {2: array([ 1, 16]), 5: array([ 1, 10, 17, 19,  3]), 8: array([ 9, 19,  8,  2,  4,  5, 16,  0])}
# obs_state_dict: key: k, value: simul.infection_array
# I1_dict: key: k, value: list of sets where each set contains the index of infected nodes at each timestep.
def random_seed_and_observe_infections(simul, k_range, people_nodes_idx):
    seeds_array_dict = dict()
    S_original_dict = dict()
    obs_state_dict = dict()
    I1_dict = dict()
    # Initialize Simulation object
    for k in k_range:
        no_of_S_original = k
        S_original = np.random.choice(a=people_nodes_idx, size=no_of_S_original, replace=False)
        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        # Set nodes in S_original at time 0 as seeds.
        seeds_array[0, S_original] = True
        # simul = Simulation(G, S_original, people_nodes, area_array, contact_area, n_timesteps, rho, d, q)
        simul.set_seeds(seeds_array)
        # simul.set_seeds(S_original)
        simul.simulate()

        obs_state = simul.infection_array
        I1_sets = []
        for t in range(simul.n_timesteps):
            I1 = set(obs_state[t].nonzero()[0])
            I1_sets.append(I1)
        # I2: sample set of uninfected nodes at the last timestep w/o replacement
        S_original_dict[k] = S_original
        seeds_array_dict[k] = seeds_array
        obs_state_dict[k] = obs_state
        I1_dict[k] = I1_sets
        print("k:{}, S_original:{}, |I1|: {}".format(k, S_original, len(I1)))

    # return S_original_dict, obs_state_dict, I1_dict
    return S_original_dict, seeds_array_dict, obs_state_dict, I1_dict


def random_seed1_seed2_and_observe_infections(simul, k_range, people_nodes_idx):
    seeds_array_dict = dict()
    S1_dict = dict()
    S2_dict = dict()
    obs_state_dict = dict()
    I1_dict = dict()
    # Initialize Simulation object
    for k in k_range:
        no_of_seeds = k
        S1 = np.random.choice(a=people_nodes_idx, size=no_of_seeds, replace=False)
        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        # Set nodes in S1 at time 0 as seeds.
        seeds_array[0, S1] = True
        simul.set_seeds(seeds_array)
        simul.simulate()

        # Get nodes in timestep 1 that are not infected
        V = set(simul.people_nodes.nonzero()[0])
        S2_candidate = V - set(S1) - set(simul.infection_array[1,:].nonzero()[0])
        S2 = np.random.choice(a=np.array(list(S2_candidate)), size=no_of_seeds, replace=False)
        seeds_array[1, S2] = True
        simul.set_seeds(seeds_array)
        simul.simulate()

        obs_state = simul.infection_array
        I1_sets = []
        for t in range(simul.n_timesteps):
            I1 = set(obs_state[t].nonzero()[0])
            I1_sets.append(I1)
        # I2: sample set of uninfected nodes at the last timestep w/o replacement
        S1_dict[k] = S1
        S2_dict[k] = S2
        seeds_array_dict[k] = seeds_array
        obs_state_dict[k] = obs_state
        I1_dict[k] = I1_sets
        print("k:{}, S1:{}, S2: {}, |I1|: {}".format(k, S1, S2, len(I1)))

    # return S1_dict, obs_state_dict, I1_dict
    return S1_dict, S2_dict, seeds_array_dict, obs_state_dict, I1_dict

