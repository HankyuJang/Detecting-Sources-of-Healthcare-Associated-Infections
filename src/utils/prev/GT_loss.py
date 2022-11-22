"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last Modified: Mar, 2020 

Description: functions for computing the grount truth loss
"""

from simulator_load_sharing import *
import numpy as np
from utils.greedy_source_detection import compute_loss, get_pos_hit_neg_hit


# GT_loss_1_dict. key: k, value: numpy array of ground truth loss_1 per timestep
# GT_loss_total_dict. key: k, value: numpy array of ground truth loss_total per timestep
# def compute_GT_loss_per_timestep(simul, S_original_dict, obs_state_dict, n_replicates):
def compute_GT_loss_per_timestep(simul, seeds_array_dict, obs_state_dict, n_replicates, P_dict, N_dict):
    GT_loss_1_dict = dict()
    GT_loss_total_dict = dict()

    P_hit_dict = dict()
    N_hit_dict = dict()
    P_hit_frac_dict = dict()
    N_hit_frac_dict = dict()
    P_N_hit_diff_dict = dict()
    P_N_hit_ratio_dict = dict()

    # for k, S_original in S_original_dict.items():
    for k, seeds_array in seeds_array_dict.items():
        # get nodes in P and N
        nodes_in_P = list(P_dict[k])
        nodes_in_N = list(N_dict[k])

        obs_state = obs_state_dict[k].astype(np.float32)
        probability_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
        infection_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)

        # seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        # seeds_array[0, S_original] = True
        simul.set_seeds(seeds_array)
        # Run simulation for n_replicates times.
        for rep in range(n_replicates):
            # simul.set_seeds(S_original)
            simul.simulate()
            probability_array[rep, :, :] = simul.probability_array
            infection_array[rep, :, :] = simul.infection_array

        GT_loss_1_array = np.zeros((simul.n_timesteps)).astype(np.float32)
        GT_loss_total_array = np.zeros((simul.n_timesteps)).astype(np.float32)
        for t in range(simul.n_timesteps):
            # I is the probability array 
            I_prob = np.mean(probability_array[:, t, :], axis=0)
            I1_inf = obs_state[t].astype(np.float32)

            loss_1 = compute_loss("loss_1", I1_inf, I_prob)
            loss_total = compute_loss("loss_total", I1_inf, I_prob)

            GT_loss_1_array[t] = loss_1
            GT_loss_total_array[t] = loss_total
            # print("Ground truth loss. loss_total: {:.3f}, loss_1 = {:.3f}".format(loss_total, loss_1))
        GT_loss_1_dict[k] = GT_loss_1_array
        GT_loss_total_dict[k] = GT_loss_total_array

        # compute hit ratios
        P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio = get_pos_hit_neg_hit(nodes_in_P, nodes_in_N, probability_array)

        P_hit_dict[k] = P_hit
        N_hit_dict[k] = N_hit
        P_hit_frac_dict[k] = P_hit_frac
        N_hit_frac_dict[k] = N_hit_frac
        P_N_hit_diff_dict[k] = P_N_hit_diff
        P_N_hit_ratio_dict[k] = P_N_hit_ratio

    return GT_loss_1_dict, GT_loss_total_dict, \
            P_hit_dict, N_hit_dict, P_hit_frac_dict, N_hit_frac_dict, P_N_hit_diff_dict, P_N_hit_ratio_dict

