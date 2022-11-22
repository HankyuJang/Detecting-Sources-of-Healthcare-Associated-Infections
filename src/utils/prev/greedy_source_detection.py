"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last Modified: Mar, 2021

Description: functions for the greedy source detection algoritm
and some other functions
"""
from simulator_load_sharing import *
from tqdm import tqdm
import numpy as np

from functools import partial
import multiprocessing
import random as rd

def simulate_multiprocessing(simul, seeds_2d_bool, rep):
    np.random.seed(rd.randint(0, 10000000))
    simul.set_seeds(seeds_2d_bool)
    simul.simulate()
    return simul.probability_array, simul.infection_array

# infection_array contains infected node info per timestep
def greedy_source_detection(simul, focus_obs1, k, obs_state, n_replicates, MTP):
    S = set()
    S_candidate_list = list(simul.people_nodes.nonzero()[0])

    # Greedily add a source that yields most of the infections in I_obs
    for i in range(k):
        S_best = -1
        loss_best = 99999.0
        I_best = set()
        for S_candidate in tqdm(S_candidate_list):
            S_temp = {S_candidate}
            S_temp.update(S)
            seeds = np.array(list(S_temp))

            seeds_2d_bool = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            seeds_2d_bool[0, seeds] = True

            probability_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
            infection_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            if MTP[0]:
                n_cpu = MTP[1]
                rep = range(n_replicates)
                pool = multiprocessing.Pool(processes=n_cpu)
                func = partial(simulate_multiprocessing, simul, seeds_2d_bool)
                r = pool.map(func, rep)
                pool.close()
                for i, result in enumerate(r):
                    probability_array[i,:,:] = result[0]
                    infection_array[i,:,:] = result[1]
            else:
                # Run simulation for n_replicates times.
                for idx_replicate in range(n_replicates):
                    # simul = Simulation(G, seeds, people_nodes, area_array, contact_area, n_timesteps, rho, d, q)
                    simul.set_seeds(seeds_2d_bool)
                    simul.simulate()
                    probability_array[idx_replicate, :, :] = simul.probability_array
                    infection_array[idx_replicate, :, :] = simul.infection_array

            # take mean prob for each node to getting infected
            I_prob = np.mean(probability_array[:, simul.n_timesteps-1, :], axis=0)
            I1_inf = obs_state[simul.n_timesteps-1].astype(np.float32)

            if focus_obs1:
                loss = compute_loss("loss_1", I1_inf, I_prob)
                # loss = get_loss_focus_obs1(obs_state, prob_array_mean)
            else:
                loss = compute_loss("loss_total", I1_inf, I_prob)

            if loss < loss_best:
                loss_best = loss
                S_best = S_candidate
                probability_array_best = probability_array

        S.add(S_best)
        S_candidate_list.remove(S_best)

    return np.array(list(S)), probability_array_best, loss_best

# For each of the five dictionaries returned, key is the method used in greedy source detection algorithm
def run_greedy_source_detection_report_loss_per_timestep(simul, focus_obs1_list, k_range, obs_state_dict, seeds_array_dict, n_replicates, MTP, P_dict, N_dict):
    S_detected_dict = dict() # dictionary of dictionaries
    detected_seeds_array_dict = dict()
    n_S_correct_dict = dict()
    probability_array_dict = dict()
    loss_1_dict = dict()
    loss_total_dict = dict()

    P_hit_dict = dict()
    N_hit_dict = dict()
    P_hit_frac_dict = dict()
    N_hit_frac_dict = dict()
    P_N_hit_diff_dict = dict()
    P_N_hit_ratio_dict = dict()

    for focus_obs1 in focus_obs1_list:
        S_detected_dict[focus_obs1] = dict()
        detected_seeds_array_dict[focus_obs1] = dict()
        n_S_correct_dict[focus_obs1] = dict()
        probability_array_dict[focus_obs1] = dict()
        loss_1_dict[focus_obs1] = dict()
        loss_total_dict[focus_obs1] = dict()

        P_hit_dict[focus_obs1] = dict()
        N_hit_dict[focus_obs1] = dict()
        P_hit_frac_dict[focus_obs1] = dict()
        N_hit_frac_dict[focus_obs1] = dict()
        P_N_hit_diff_dict[focus_obs1] = dict()
        P_N_hit_ratio_dict[focus_obs1] = dict()
        print("\nGreedy source detection. Focus obs1: {}".format(focus_obs1))

        for k in k_range:
            # get nodes in P and N
            nodes_in_P = list(P_dict[k])
            nodes_in_N = list(N_dict[k])

            obs_state = obs_state_dict[k].astype(np.float32)
            S_detected, probability_array, loss = greedy_source_detection(simul, focus_obs1, k, obs_state, n_replicates, MTP)

            # S = S_original_dict[k]
            S = seeds_array_dict[k][0,:].nonzero()[0] # get original seeds at time 0.
            n_S_correct = len(set(S_detected).intersection(set(S)))
            n_S_correct_dict[focus_obs1][k] = n_S_correct

            S_detected_dict[focus_obs1][k] = S_detected
            detected_seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            print(S_detected)
            detected_seeds_array[0, S_detected] = True
            detected_seeds_array_dict[focus_obs1][k] = detected_seeds_array

            probability_array_dict[focus_obs1][k] = np.mean(probability_array[:, :, :], axis=0)

            loss_1_array = np.zeros((simul.n_timesteps)).astype(np.float32)
            loss_total_array = np.zeros((simul.n_timesteps)).astype(np.float32)

            for t in range(simul.n_timesteps):
                I_prob = np.mean(probability_array[:, t, :], axis=0)
                I1_inf = obs_state[t].astype(np.float32)

                loss_1 = compute_loss("loss_1", I1_inf, I_prob)
                loss_total = compute_loss("loss_total", I1_inf, I_prob)

                loss_1_array[t] = loss_1
                loss_total_array[t] = loss_total
                # print("k:{}, S_detected: {}, |S_correct|: {}, loss_total: {:.3f}, loss_1: {:.3f}".format(k, S_detected, len(set(S_detected).intersection(set(S))), loss_total, loss_1))

            loss_1_dict[focus_obs1][k] = loss_1_array
            loss_total_dict[focus_obs1][k] = loss_total_array

            # compute hit ratios
            P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio = get_pos_hit_neg_hit(nodes_in_P, nodes_in_N, probability_array)

            P_hit_dict[focus_obs1][k] = P_hit
            N_hit_dict[focus_obs1][k] = N_hit
            P_hit_frac_dict[focus_obs1][k] = P_hit_frac
            N_hit_frac_dict[focus_obs1][k] = N_hit_frac
            P_N_hit_diff_dict[focus_obs1][k] = P_N_hit_diff
            P_N_hit_ratio_dict[focus_obs1][k] = P_N_hit_ratio

    return S_detected_dict, detected_seeds_array_dict, n_S_correct_dict, probability_array_dict, loss_1_dict, loss_total_dict, \
            P_hit_dict, N_hit_dict, P_hit_frac_dict, N_hit_frac_dict, P_N_hit_diff_dict, P_N_hit_ratio_dict
            

# infection_array contains infected node info per timestep
def greedy_source_detection_t0t1(simul, focus_obs1, k, obs_state, n_replicates, MTP):
    S1 = set()
    S1_candidate_list = list(simul.people_nodes.nonzero()[0])

    # Greedily add a source that yields most of the infections in I_obs
    for i in range(k):
        S1_best = -1
        loss_best = 99999.0
        I_best = set()
        for S1_candidate in tqdm(S1_candidate_list):
            S1_temp = {S1_candidate}
            S1_temp.update(S1)
            seeds = np.array(list(S1_temp))

            seeds_2d_bool = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            seeds_2d_bool[0, seeds] = True

            probability_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
            infection_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            # Run simulation for n_replicates times.
            for idx_replicate in range(n_replicates):
                # simul = Simulation(G, seeds, people_nodes, area_array, contact_area, n_timesteps, rho, d, q)
                simul.set_seeds(seeds_2d_bool)
                simul.simulate()
                probability_array[idx_replicate, :, :] = simul.probability_array
                infection_array[idx_replicate, :, :] = simul.infection_array

            # take mean prob for each node to getting infected
            I_prob = np.mean(probability_array[:, simul.n_timesteps-1, :], axis=0)
            I1_inf = obs_state[simul.n_timesteps-1].astype(np.float32)

            if focus_obs1:
                loss = compute_loss("loss_1", I1_inf, I_prob)
                # loss = get_loss_focus_obs1(obs_state, prob_array_mean)
            else:
                loss = compute_loss("loss_total", I1_inf, I_prob)

            if loss < loss_best:
                loss_best = loss
                S1_best = S1_candidate
                probability_array_best = probability_array

        S1.add(S1_best)
        S1_candidate_list.remove(S1_best)

    # Now for S2
    seeds1 = np.array(list(S1))
    S2 = set()
    S2_candidate_list = list(set(list(simul.people_nodes.nonzero()[0])) - S1)
    for i in range(k):
        S2_best = -1
        loss_best = 99999.0
        I_best = set()
        for S2_candidate in tqdm(S2_candidate_list):
            S2_temp = {S2_candidate}
            S2_temp.update(S2)
            seeds2 = np.array(list(S2_temp))

            seeds_2d_bool = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            seeds_2d_bool[0, seeds1] = True # seeds at time 0
            seeds_2d_bool[1, seeds2] = True # seeds at time 1

            probability_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
            infection_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            # Run simulation for n_replicates times.
            for idx_replicate in range(n_replicates):
                # simul = Simulation(G, seeds, people_nodes, area_array, contact_area, n_timesteps, rho, d, q)
                simul.set_seeds(seeds_2d_bool)
                simul.simulate()
                probability_array[idx_replicate, :, :] = simul.probability_array
                infection_array[idx_replicate, :, :] = simul.infection_array

            # take mean prob for each node to getting infected
            I_prob = np.mean(probability_array[:, simul.n_timesteps-1, :], axis=0)
            I1_inf = obs_state[simul.n_timesteps-1].astype(np.float32)

            if focus_obs1:
                loss = compute_loss("loss_1", I1_inf, I_prob)
                # loss = get_loss_focus_obs1(obs_state, prob_array_mean)
            else:
                loss = compute_loss("loss_total", I1_inf, I_prob)

            if loss < loss_best:
                loss_best = loss
                S2_best = S2_candidate
                probability_array_best = probability_array

        S2.add(S2_best)
        S2_candidate_list.remove(S2_best)

    return np.array(list(S1)), np.array(list(S2)), probability_array_best, loss_best

def run_greedy_source_detection_t0t1_report_loss_per_timestep(simul, focus_obs1_list, k_range, obs_state_dict, seeds_array_dict, n_replicates, MTP):
    S1_detected_dict = dict() # dictionary of dictionaries
    S2_detected_dict = dict() # dictionary of dictionaries
    detected_seeds_array_dict = dict()
    n_S_correct_dict = dict()
    probability_array_dict = dict()
    loss_1_dict = dict()
    loss_total_dict = dict()

    for focus_obs1 in focus_obs1_list:
        S1_detected_dict[focus_obs1] = dict()
        S2_detected_dict[focus_obs1] = dict()
        detected_seeds_array_dict[focus_obs1] = dict()
        n_S_correct_dict[focus_obs1] = dict()
        probability_array_dict[focus_obs1] = dict()
        loss_1_dict[focus_obs1] = dict()
        loss_total_dict[focus_obs1] = dict()
        print("\nGreedy source detection. Focus obs1: {}".format(focus_obs1))

        for k in k_range:
            obs_state = obs_state_dict[k].astype(np.float32)
            S1_detected, S2_detected, probability_array, loss = greedy_source_detection_t0t1(simul, focus_obs1, k, obs_state, n_replicates, MTP)

            # S = S_original_dict[k]
            S1 = seeds_array_dict[k][0,:].nonzero()[0] # get original seeds at time 0.
            S2 = seeds_array_dict[k][1,:].nonzero()[0] # get original seeds at time 1.
            S = set(S1).union(set(S2))
            S_detected = set(S1_detected).union(set(S2_detected))
            n_S_correct = len(S.intersection(S_detected))
            n_S_correct_dict[focus_obs1][k] = n_S_correct

            S1_detected_dict[focus_obs1][k] = S1_detected
            S2_detected_dict[focus_obs1][k] = S2_detected

            detected_seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            detected_seeds_array[0, S1_detected] = True
            detected_seeds_array[1, S2_detected] = True
            detected_seeds_array_dict[focus_obs1][k] = detected_seeds_array

            probability_array_dict[focus_obs1][k] = np.mean(probability_array[:, :, :], axis=0)

            loss_1_array = np.zeros((simul.n_timesteps)).astype(np.float32)
            loss_total_array = np.zeros((simul.n_timesteps)).astype(np.float32)

            for t in range(simul.n_timesteps):
                I_prob = np.mean(probability_array[:, t, :], axis=0)
                I1_inf = obs_state[t].astype(np.float32)

                loss_1 = compute_loss("loss_1", I1_inf, I_prob)
                loss_total = compute_loss("loss_total", I1_inf, I_prob)

                loss_1_array[t] = loss_1
                loss_total_array[t] = loss_total
                # print("k:{}, S1: {}, S2: {}, S1_detected: {}, S2_detected: {}, |S_correct|: {}, loss_total: {:.3f}, loss_1: {:.3f}".format(k, S1, S2, S1_detected, S2_detected, n_S_correct, loss_total, loss_1))

            loss_1_dict[focus_obs1][k] = loss_1_array
            loss_total_dict[focus_obs1][k] = loss_total_array

    return S1_detected_dict, S2_detected_dict, detected_seeds_array_dict, n_S_correct_dict, probability_array_dict, loss_1_dict, loss_total_dict

def compute_loss(loss_type, I1_inf, I_prob):
    if loss_type == "loss_1":
        idx_True = I1_inf.nonzero()[0]
        loss = np.sum(I1_inf[idx_True].astype(float) - I_prob[idx_True])
    elif loss_type == "loss_total":
        loss = np.sum(np.abs(I1_inf - I_prob))
    return loss

# nodes_in_P, nodes_in_N. These are lists
# probability_array is a 3-d numpy array. (#rep, #timesteps, #nodes)
def get_pos_hit_neg_hit(nodes_in_P, nodes_in_N, probability_array):
    len_P = len(nodes_in_P)
    len_N = len(nodes_in_N)
    # mean infection prob at the last timestep
    mean_infection_prob = np.mean(probability_array, axis=0)[-1, :]
    # print()
    # print(mean_infection_prob.shape)
    # print(mean_infection_prob)

    P_hit = np.sum(mean_infection_prob[nodes_in_P])
    N_hit = np.sum(mean_infection_prob[nodes_in_N])
    P_hit_frac = P_hit / len_P
    N_hit_frac = N_hit / len_N

    P_N_hit_diff = P_hit_frac - N_hit_frac
    P_N_hit_ratio = P_hit_frac / N_hit_frac
    
    return P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio
