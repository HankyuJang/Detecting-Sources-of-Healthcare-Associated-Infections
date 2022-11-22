"""
Author: Bijaya Adhikari, Hankyu Jang
Email: {bijaya-adhikari, hankyu-jang}@uiowa.edu
Last Modified: Apr, 2021 

Description: ICSK. Sources at time 0 only. Observe infections over time
Temporal graph.

Usage

To run it on Karate graph,
$ python ISCK_temporal.py -dose_response linear
$ python ISCK_temporal.py -dose_response exponential

To run it on UIHC sampled graph,
$ python ISCK_temporal.py -name UIHC_HCP_patient_room -year 2011 -sampled True -dose_response linear
$ python ISCK_temporal.py -name UIHC_HCP_patient_room -year 2011 -sampled True -dose_response exponential

To run it on UIHC original graph,
$ python ISCK_temporal.py -name UIHC_HCP_patient_room -year 2011 -dose_response linear
$ python ISCK_temporal.py -name UIHC_HCP_patient_room -year 2011 -dose_response exponential

"""
from utils.load_network import *
from utils.greedy_source_detection import *
from utils.random_seed import *
from utils.GT_loss import *
from simulator_load_sharing_temporal import *

import pandas as pd
import argparse
import math
import random as random
import copy
import timeit
import numpy as np

def get_MCC(TP, TN, FP, FN):
    numerator = (TP*TN) - (FP*FN)
    denominator = np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )
    return numerator / denominator

def prepare_df_exp_GS(focus_obs1, k_range, obs_state_dict, S_original_dict, GS_S_detected_dict, GS_n_S_correct_dict,\
        GS_P_hit_dict, P_dict, GS_P_hit_frac_dict, GS_N_hit_dict, N_dict, GS_N_hit_frac_dict,  GS_P_N_hit_diff_dict,  GS_P_N_hit_ratio_dict):
    df_exp = pd.DataFrame(
            {"k":[k for k in k_range],
                "n_obs_state": [obs_state[n_timesteps-1].nonzero()[0].shape[0] for obs_state in obs_state_dict.values()],
                "S": [sorted(list(S_original)) for S_original in S_original_dict.values()],
                "GS_S_detected": [sorted(list(GS_S_detected)) for GS_S_detected in GS_S_detected_dict[focus_obs1].values()],
                "GS_n_S_correct": list(GS_n_S_correct_dict[focus_obs1].values()),
                "P_hit": list(GS_P_hit_dict[focus_obs1].values()),
                "len(P)": [len(P_set) for P_set in P_dict.values()],
                "P_hit_frac": list(GS_P_hit_frac_dict[focus_obs1].values()),
                "N_hit": list(GS_N_hit_dict[focus_obs1].values()),
                "len(N)": [len(N_set) for N_set in N_dict.values()],
                "N_hit_frac": list(GS_N_hit_frac_dict[focus_obs1].values()),
                "P_N_hit_diff": list(GS_P_N_hit_diff_dict[focus_obs1].values()),
                "P_N_hit_ratio": list(GS_P_N_hit_ratio_dict[focus_obs1].values()),
                "obs_state": [list(obs_state[n_timesteps-1].nonzero()[0]) for obs_state in obs_state_dict.values()]
                })
    # Compute expected F1 and expected MCC
    df_exp.insert(value=df_exp["P_hit"], loc=df_exp.shape[1]-1, column="E_TP")
    df_exp.insert(value=df_exp["len(N)"] - df_exp["N_hit"], loc=df_exp.shape[1]-1, column="E_TN")
    df_exp.insert(value=df_exp["N_hit"], loc=df_exp.shape[1]-1, column="E_FP")
    df_exp.insert(value=df_exp["len(P)"] - df_exp["P_hit"], loc=df_exp.shape[1]-1, column="E_FN")
    df_exp.insert(value=df_exp["E_TP"] / (df_exp["E_TP"] + df_exp["E_FP"]), loc=df_exp.shape[1]-1, column="E_precision")
    df_exp.insert(value=df_exp["E_TP"] / (df_exp["E_TP"] + df_exp["E_FN"]), loc=df_exp.shape[1]-1, column="E_recall")
    df_exp.insert(value=2*df_exp["E_precision"]*df_exp["E_recall"] / (df_exp["E_precision"] + df_exp["E_recall"]), loc=df_exp.shape[1]-1, column="E_F1")
    df_exp.insert(value=get_MCC(df_exp["E_TP"], df_exp["E_TN"], df_exp["E_FP"], df_exp["E_FN"]), loc=df_exp.shape[1]-1, column="E_MCC")

    return df_exp

def prepare_df_exp(k_range, obs_state_dict, S_original_dict, S_detected_dict, n_S_correct_dict, \
        P_hit_dict, P_dict, P_hit_frac_dict, N_hit_dict, N_dict, N_hit_frac_dict,  P_N_hit_diff_dict,  P_N_hit_ratio_dict):
    df_exp = pd.DataFrame(
            {"k":[k for k in k_range],
                "n_obs_state": [obs_state[n_timesteps-1].nonzero()[0].shape[0] for obs_state in obs_state_dict.values()],
                "S": [sorted(list(S_original)) for S_original in S_original_dict.values()],
                "S_detected": [sorted(list(S_detected)) for S_detected in S_detected_dict.values()],
                "n_S_correct": list(n_S_correct_dict.values()),
                "P_hit": list(P_hit_dict.values()),
                "len(P)": [len(P_set) for P_set in P_dict.values()],
                "P_hit_frac": list(P_hit_frac_dict.values()),
                "N_hit": list(N_hit_dict.values()),
                "len(N)": [len(N_set) for N_set in N_dict.values()],
                "N_hit_frac": list(N_hit_frac_dict.values()),
                "P_N_hit_diff": list(P_N_hit_diff_dict.values()),
                "P_N_hit_ratio": list(P_N_hit_ratio_dict.values()),
                "obs_state": [list(obs_state[n_timesteps-1].nonzero()[0]) for obs_state in obs_state_dict.values()]
                })
    # Compute expected F1 and expected MCC
    df_exp.insert(value=df_exp["P_hit"], loc=df_exp.shape[1]-1, column="E_TP")
    df_exp.insert(value=df_exp["len(N)"] - df_exp["N_hit"], loc=df_exp.shape[1]-1, column="E_TN")
    df_exp.insert(value=df_exp["N_hit"], loc=df_exp.shape[1]-1, column="E_FP")
    df_exp.insert(value=df_exp["len(P)"] - df_exp["P_hit"], loc=df_exp.shape[1]-1, column="E_FN")
    df_exp.insert(value=df_exp["E_TP"] / (df_exp["E_TP"] + df_exp["E_FP"]), loc=df_exp.shape[1]-1, column="E_precision")
    df_exp.insert(value=df_exp["E_TP"] / (df_exp["E_TP"] + df_exp["E_FN"]), loc=df_exp.shape[1]-1, column="E_recall")
    df_exp.insert(value=2*df_exp["E_precision"]*df_exp["E_recall"] / (df_exp["E_precision"] + df_exp["E_recall"]), loc=df_exp.shape[1]-1, column="E_F1")
    df_exp.insert(value=get_MCC(df_exp["E_TP"], df_exp["E_TN"], df_exp["E_FP"], df_exp["E_FN"]), loc=df_exp.shape[1]-1, column="E_MCC")

    return df_exp

def prepare_result_dataframes():
    df_GT = prepare_df_exp(k_range, obs_state_dict, S_original_dict, S_original_dict, dict(list(zip(k_range, k_range))), \
            GT_P_hit_dict, P_dict, GT_P_hit_frac_dict, GT_N_hit_dict, N_dict, GT_N_hit_frac_dict,  GT_P_N_hit_diff_dict,  GT_P_N_hit_ratio_dict)
    df_B_random = prepare_df_exp(k_range, obs_state_dict, S_original_dict, B_random_S_dict, B_random_n_S_correct_dict, \
            B_random_P_hit_dict, P_dict, B_random_P_hit_frac_dict, B_random_N_hit_dict, N_dict, B_random_N_hit_frac_dict,  B_random_P_N_hit_diff_dict,  B_random_P_N_hit_ratio_dict)
    df_B_degree = prepare_df_exp(k_range, obs_state_dict, S_original_dict, B_degree_S_dict, B_degree_n_S_correct_dict, \
            B_degree_P_hit_dict, P_dict, B_degree_P_hit_frac_dict, B_degree_N_hit_dict, N_dict, B_degree_N_hit_frac_dict,  B_degree_P_N_hit_diff_dict,  B_degree_P_N_hit_ratio_dict)
    df_exp1 = prepare_df_exp_GS(True, k_range, obs_state_dict, S_original_dict, GS_S_detected_dict, GS_n_S_correct_dict, \
            GS_P_hit_dict, P_dict, GS_P_hit_frac_dict, GS_N_hit_dict, N_dict, GS_N_hit_frac_dict,  GS_P_N_hit_diff_dict,  GS_P_N_hit_ratio_dict)
    df_exp2 = prepare_df_exp_GS(False, k_range, obs_state_dict, S_original_dict, GS_S_detected_dict, GS_n_S_correct_dict, \
            GS_P_hit_dict, P_dict, GS_P_hit_frac_dict, GS_N_hit_dict, N_dict, GS_N_hit_frac_dict,  GS_P_N_hit_diff_dict,  GS_P_N_hit_ratio_dict)
    df_ISCK = prepare_df_exp(k_range, obs_state_dict, S_original_dict, ISCK_S_detected_dict, ISCK_n_S_correct_dict, \
            ISCK_P_hit_dict, P_dict, ISCK_P_hit_frac_dict, ISCK_N_hit_dict, N_dict, ISCK_N_hit_frac_dict,  ISCK_P_N_hit_diff_dict,  ISCK_P_N_hit_ratio_dict)
    df_ISCK_diff = prepare_df_exp(k_range, obs_state_dict, S_original_dict, ISCK_diff_S_detected_dict, ISCK_diff_n_S_correct_dict, \
            ISCK_diff_P_hit_dict, P_dict, ISCK_diff_P_hit_frac_dict, ISCK_diff_N_hit_dict, N_dict, ISCK_diff_N_hit_frac_dict,  ISCK_diff_P_N_hit_diff_dict,  ISCK_diff_P_N_hit_ratio_dict)

    # The dataframes below are no longer used.
    df_GT_loss_total = pd.DataFrame(data=GT_loss_total_dict)
    df_GS_loss_total_exp1 = pd.DataFrame(data=GS_loss_total_dict[True])
    df_GS_loss_total_exp2 = pd.DataFrame(data=GS_loss_total_dict[False])
    df_ISCK_loss_total = pd.DataFrame(data=ISCK_loss_total_dict)
    df_ISCK_diff_loss_total = pd.DataFrame(data=ISCK_diff_loss_total_dict)

    df_GT_loss_1 = pd.DataFrame(data=GT_loss_1_dict)
    df_GS_loss_1_exp1 = pd.DataFrame(data=GS_loss_1_dict[True])
    df_GS_loss_1_exp2 = pd.DataFrame(data=GS_loss_1_dict[False])
    df_ISCK_loss_1 = pd.DataFrame(data=ISCK_loss_1_dict)
    df_ISCK_diff_loss_1 = pd.DataFrame(data=ISCK_diff_loss_1_dict)

    return df_GT, df_B_random, df_B_degree, df_exp1, df_exp2, df_ISCK, df_ISCK_diff, \
            df_GT_loss_total, df_GT_loss_1, df_GS_loss_total_exp1, df_GS_loss_total_exp2, df_ISCK_loss_total, df_ISCK_diff_loss_total, \
            df_GS_loss_1_exp1, df_GS_loss_1_exp2, df_ISCK_loss_1, df_ISCK_diff_loss_1

def save_result_dataframes(folder, name, dose_response):
    # Save datasets
    df_GT.to_csv("../tables/{}/{}/{}/GT.csv".format(folder, name, dose_response), index=False)
    df_B_random.to_csv("../tables/{}/{}/{}/B_random.csv".format(folder, name, dose_response), index=False)
    df_B_degree.to_csv("../tables/{}/{}/{}/B_degree.csv".format(folder, name, dose_response), index=False)
    df_exp1.to_csv("../tables/{}/{}/{}/exp1.csv".format(folder, name, dose_response), index=False)
    df_exp2.to_csv("../tables/{}/{}/{}/exp2.csv".format(folder, name, dose_response), index=False)
    df_ISCK.to_csv("../tables/{}/{}/{}/ISCK.csv".format(folder, name, dose_response), index=False)
    df_ISCK_diff.to_csv("../tables/{}/{}/{}/ISCK_diff.csv".format(folder, name, dose_response), index=False)

    # df_GT_loss_total.to_csv("../tables/{}/{}/{}_GT_loss_total.csv".format(folder, name, folder), index=False)
    # df_GT_loss_1.to_csv("../tables/{}/{}/{}_GT_loss_1.csv".format(folder, name, folder), index=False)
    # df_GS_loss_total_exp1.to_csv("../tables/{}/{}/{}_GS_loss_total_exp1.csv".format(folder, name, folder), index=False)
    # df_GS_loss_total_exp2.to_csv("../tables/{}/{}/{}_GS_loss_total_exp2.csv".format(folder, name, folder), index=False)
    # df_ISCK_loss_total.to_csv("../tables/{}/{}/{}_ISCK_loss_total.csv".format(folder, name, folder), index=False)
    # df_GS_loss_1_exp1.to_csv("../tables/{}/{}/{}_GS_loss_1_exp1.csv".format(folder, name, folder), index=False)
    # df_GS_loss_1_exp2.to_csv("../tables/{}/{}/{}_GS_loss_1_exp2.csv".format(folder, name, folder), index=False)
    # df_ISCK_loss_1.to_csv("../tables/{}/{}/{}_ISCK_loss_1.csv".format(folder, name, folder), index=False)

# Return infected node set at the last timestep
# people nodes only?
def get_P_N(people_nodes_idx, k_range, I1_dict):
    # Compute P and N
    P_dict = dict()
    N_dict = dict()
    V = set(people_nodes_idx)
    for k in k_range:
        P = I1_dict[k][-1]
        N = V - P
        P_dict[k] = P
        N_dict[k] = N
    return P_dict, N_dict

# expected number of infections in N_t given X is the seed set
# Run simulations for some number of replicates, then compute avg infection of each node at the end of the timestep
# add parameters: some other parameters needed to run simulation
def f(simul, X, N):
    if X==None or len(X) == 0:
        return 0

    seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    seeds_array[0, list(X)] = True

    probability_array = np.zeros((simul.n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
    infection_array = np.zeros((simul.n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)

    for idx_replicate in range(simul.n_replicates):
        simul.set_seeds(seeds_array)
        simul.simulate()

        probability_array[idx_replicate, :, :] = simul.probability_array
        infection_array[idx_replicate, :, :] = simul.infection_array

    expected_infection_in_N = np.sum(np.mean(probability_array[:, simul.n_timesteps-1, list(N)], axis=0))
    return expected_infection_in_N

# expected number of infections in P_t given X is the seed set
# Run simulations for some number of replicates, then compute avg infection of each node at the end of the timestep
# add parameters: some other parameters needed to run simulation
def g(simul, X, P):
    if len(X) == 0:
        return 0
    seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    seeds_array[0, list(X)] = True

    probability_array = np.zeros((simul.n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
    infection_array = np.zeros((simul.n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)

    for idx_replicate in range(simul.n_replicates):
        simul.set_seeds(seeds_array)
        simul.simulate()

        probability_array[idx_replicate, :, :] = simul.probability_array
        infection_array[idx_replicate, :, :] = simul.infection_array

    expected_infection_in_P = np.sum(np.mean(probability_array[:, simul.n_timesteps-1, list(P)], axis=0))

    return expected_infection_in_P


# for j (node that is in X \ S), compute f(X) - f(X \ j)
# def f_gain_of_adding_j(X, S):
def f_2(simul, X, N, S):
    X_copied = copy.deepcopy(X)
    f_of_X = f(simul, X_copied, N)
    total = 0
    for j in X_copied - S:
        f_of_X_minus_j = f(simul, X_copied.remove(j), N)
        total += (f_of_X - f_of_X_minus_j)
        X_copied.add(j)
    return total

# expected number of infections with seeds in S\X
def f_3(f_gain_to_empty_dict, simul, X, N, S):
    total = 0
    for j in S - X:
        # total += f(simul, set([j]), N) # this term should be a summation of values in the look-up table.
        total += f_gain_to_empty_dict[j]
    return total

# For all the nodes j in people nodes, compute f(j|empty), save it as a dictionary
def precompute_f_gain_to_empty_set(simul, N, V):
    f_gain_to_empty_dict = {}
    for j in V:
        f_gain_to_empty_dict[j] = f(simul, set([j]), N)
    return f_gain_to_empty_dict

def f_hat(f_gain_to_empty_dict, simul, X, N, S):
    if len(S) == 0:
        return 0
    val1 = f(simul, X, N) - f_2(simul, X, N, S) 
    val1 = max(0, val1) + 0.00000001
    
    val = val1 + f_3(f_gain_to_empty_dict, simul, X, N, S)
    if val <0:
        print(" f_hat is negative")
    return val
    # return f(simul, X, N) - f_2(simul, X, N, S) + f_3(simul, X, N, S)

# def g_hat(V, X, S):
    # total = 0
    # for j in S:
        # S_pi_j = get_S_pi_j(V, X, j)
        # S_pi_j_minus1 = S_pi_j - set(j)
        # total += 1
    # return total

# TODO: keep adding nodes in pi until k1 nodes are in pi
def get_pi(f_gain_to_empty_dict, simul, V, S, P, N, k1, how):
    # print(S)
    V_copied = set(copy.deepcopy(V))
    # print("V_copied: {}".format(V_copied))
    # pi_V = set()#[]
    pi_V = []
    while len(V_copied) > 0:
        max_gain = -math.inf
        best_node = None
        current_footperint_in_P = g(simul, pi_V, P)
        for v in V_copied:
            pi_V.append(v)
            temp_footprint_in_P = g(simul, pi_V, P)
            # print("S:{}".format(S))
            # print("N:{}".format(N))
            # print("X:{}".format(X))
            # print("v:{}".format(v))
            temp_f_hat = f_hat(f_gain_to_empty_dict, simul, X=S, N=N, S=set([v]))

            # temp_gain = (temp_footprint_in_P - current_footperint_in_P) / temp_f_hat
            temp_gain = (temp_footprint_in_P - current_footperint_in_P) 
            #if temp_gain < 0:
                #print("temp gain:", temp_gain)
            
            if how == "ratio":
                temp_gain =  temp_gain / temp_f_hat
            elif how == "diff":
                temp_gain =  temp_gain - temp_f_hat

            if temp_gain >= max_gain:
                max_gain = temp_gain
                best_node = v
            pi_V.remove(v)

        # [TODO] best_nodes are sometimes None. If this occurs, add all to pi_V then break
        #--------------------------
        if best_node == None:
            for v in V_copied:
                pi_V.append(v)
            break
        #--------------------------

        V_copied.remove(best_node) 
        pi_V.append(best_node)
        # print("len(pi): {}, pi_V: {}".format(len(pi_V), pi_V))
        # BREAK if k1 nodes are added in pi_V. When doing this, simply add all the remaining nodes in V_copied to pi_V
        if len(pi_V) == k1:
            pi_V.extend(list(V_copied))
            break
        # print("V_copied: {}".format(V_copied))
    return pi_V

def ISCK(simul, V, P, N, k1, k2, how):
    # precompute f(j|empty)
    f_gain_to_empty_dict = precompute_f_gain_to_empty_set(simul, N, V)
    S = set()
    while True:
        pi = get_pi(f_gain_to_empty_dict, simul, V, S, P, N, k1, how)
        print("pi: {}".format(pi))

        S_new = set()
        i=0
        #--------------------------
        # while len(S_new) < k1 and f_hat(simul, X=set(), N=N, S=S_new) < k2:
        while len(S_new) < k1 and f_hat(f_gain_to_empty_dict, simul, X=S, N=N, S=S_new) < k2: # [TODO] modified
        #--------------------------
            S_new.add(pi[i])
            i+=1
        if S_new == S:
            break
        S = S_new
        print("G(): ", g(simul, S, P))
        print("f(): ", f(simul, S, N))
    return S

def compute_loss_per_timestep(simul, seeds_array, obs_state):

    probability_array = np.zeros((simul.n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
    infection_array = np.zeros((simul.n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)

    # seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    # seeds_array[0, S_original] = True
    simul.set_seeds(seeds_array)
    # Run simulation for n_replicates times.
    for rep in range(simul.n_replicates):
        # simul.set_seeds(S_original)
        simul.simulate()
        probability_array[rep, :, :] = simul.probability_array
        infection_array[rep, :, :] = simul.infection_array

    loss_1_array = np.zeros((simul.n_timesteps)).astype(np.float32)
    loss_total_array = np.zeros((simul.n_timesteps)).astype(np.float32)
    for t in range(simul.n_timesteps):
        # I is the probability array 
        I_prob = np.mean(probability_array[:, t, :], axis=0)
        I1_inf = obs_state[t].astype(np.float32)

        loss_1 = compute_loss("loss_1", I1_inf, I_prob)
        loss_total = compute_loss("loss_total", I1_inf, I_prob)

        loss_1_array[t] = loss_1
        loss_total_array[t] = loss_total
        # print("Ground truth loss. loss_total: {:.3f}, loss_1 = {:.3f}".format(loss_total, loss_1))
    return loss_1_array, loss_total_array, probability_array

def run_ISCK_report_loss_per_timestep(how):
    ISCK_S_detected_dict = dict()
    ISCK_n_S_correct_dict = dict()
    ISCK_loss_1_dict = dict()
    ISCK_loss_total_dict = dict()

    P_hit_dict = dict()
    N_hit_dict = dict()
    P_hit_frac_dict = dict()
    N_hit_frac_dict = dict()
    P_N_hit_diff_dict = dict()
    P_N_hit_ratio_dict = dict()

    for k in k_range:
        print("\n ISCK. k: {}".format(k))
        P = P_dict[k]
        N = N_dict[k]

        # get nodes in P and N
        nodes_in_P = list(P_dict[k])
        nodes_in_N = list(N_dict[k])

        k1 = k
        k2 = len(N)
        V = set(people_nodes_idx)
        simul.set_n_replicates(n_replicates)
        # X = {1,2,4,5}
        # expected_infections_in_N = f(simul, X, N)
        # expected_infections_in_P = g(simul, X, P)

        # Run ISCK and detect seeds
        start = timeit.default_timer()
        S_detected = ISCK(simul, V, P, N, k1, k2, how)
        stop = timeit.default_timer()
        print("Time elapsed: {:.2f}s".format(stop - start))
        print("S_detected: {}\n".format(S_detected))

        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        seeds_array[0, list(S_detected)] = True

        # Get the observed values for the ground truth
        obs_state_array = obs_state_dict[k]
        S = S_original_dict[k]

        # Computing loss for ISCK
        ISCK_loss_1_array, ISCK_loss_total_array, probability_array = compute_loss_per_timestep(simul, seeds_array, obs_state_array)
        ISCK_n_S_correct = len(set(S_detected).intersection(set(S)))

        ISCK_S_detected_dict[k] = np.array(list(S_detected))
        ISCK_n_S_correct_dict[k] = ISCK_n_S_correct
        ISCK_loss_1_dict[k] = ISCK_loss_1_array
        ISCK_loss_total_dict[k] = ISCK_loss_total_array

        # compute hit ratios
        P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio = get_pos_hit_neg_hit(nodes_in_P, nodes_in_N, probability_array)

        P_hit_dict[k] = P_hit
        N_hit_dict[k] = N_hit
        P_hit_frac_dict[k] = P_hit_frac
        N_hit_frac_dict[k] = N_hit_frac
        P_N_hit_diff_dict[k] = P_N_hit_diff
        P_N_hit_ratio_dict[k] = P_N_hit_ratio

    return ISCK_S_detected_dict, ISCK_n_S_correct_dict, ISCK_loss_1_dict, ISCK_loss_total_dict, \
             P_hit_dict, N_hit_dict, P_hit_frac_dict, N_hit_frac_dict, P_N_hit_diff_dict, P_N_hit_ratio_dict

def run_B_random_report_loss_per_timestep():

    B_random_S_dict = dict()
    B_random_n_S_correct_dict = dict()
    B_random_loss_1_dict = dict()
    B_random_loss_total_dict = dict()

    P_hit_dict = dict()
    N_hit_dict = dict()
    P_hit_frac_dict = dict()
    N_hit_frac_dict = dict()
    P_N_hit_diff_dict = dict()
    P_N_hit_ratio_dict = dict()

    for k in k_range:
        print("\n Baseline - Random. k: {}".format(k))
        P = P_dict[k]
        N = N_dict[k]
        S_random = np.random.choice(a=people_nodes_idx, size=k, replace=False)
        print("S_random: {}\n".format(S_random))

        # get nodes in P and N
        nodes_in_P = list(P_dict[k])
        nodes_in_N = list(N_dict[k])

        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        seeds_array[0, list(S_random)] = True

        # Get the observed values for the ground truth
        obs_state_array = obs_state_dict[k]
        S = S_original_dict[k]

        # Computing loss
        B_random_loss_1_array, B_random_loss_total_array, probability_array = compute_loss_per_timestep(simul, seeds_array, obs_state_array)
        B_random_n_S_correct = len(set(S_random).intersection(set(S)))

        B_random_S_dict[k] = np.array(list(S_random))
        B_random_n_S_correct_dict[k] = B_random_n_S_correct
        B_random_loss_1_dict[k] = B_random_loss_1_array
        B_random_loss_total_dict[k] = B_random_loss_total_array

        # compute hit ratios
        P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio = get_pos_hit_neg_hit(nodes_in_P, nodes_in_N, probability_array)

        P_hit_dict[k] = P_hit
        N_hit_dict[k] = N_hit
        P_hit_frac_dict[k] = P_hit_frac
        N_hit_frac_dict[k] = N_hit_frac
        P_N_hit_diff_dict[k] = P_N_hit_diff
        P_N_hit_ratio_dict[k] = P_N_hit_ratio

    return B_random_S_dict, B_random_n_S_correct_dict, B_random_loss_1_dict, B_random_loss_total_dict, \
             P_hit_dict, N_hit_dict, P_hit_frac_dict, N_hit_frac_dict, P_N_hit_diff_dict, P_N_hit_ratio_dict

def run_B_degree_report_loss_per_timestep():

    B_degree_S_dict = dict()
    B_degree_n_S_correct_dict = dict()
    B_degree_loss_1_dict = dict()
    B_degree_loss_total_dict = dict()

    P_hit_dict = dict()
    N_hit_dict = dict()
    P_hit_frac_dict = dict()
    N_hit_frac_dict = dict()
    P_N_hit_diff_dict = dict()
    P_N_hit_ratio_dict = dict()

    for k in k_range:
        print("\n Baseline - Random. k: {}".format(k))
        P = P_dict[k]
        N = N_dict[k]

        # S_degree = np.random.choice(a=people_nodes_idx, size=k, replace=False)
        # people_nodes_by_type = [v for v in G_over_time[0].nodes() if G_over_time[0].nodes[v]["type"] == "patient"]
        # Indexing is somewhat complicated, because for UIHC graphs, node id is not index.
        people_nodes_degree_array = np.array(list(G_over_time[0].degree()))[people_nodes_idx]
        people_nodes_degree_list = [(v, int(deg), idx) for (v, deg), idx in zip(people_nodes_degree_array, people_nodes_idx)]
        # people_nodes_degree_list = list(G_over_time[0].degree(people_nodes_idx))
        people_nodes_degree_list.sort(key = lambda x: x[1], reverse = True)
        S_degree = [idx for v, deg, idx in people_nodes_degree_list[:k]]

        print("S_degree: {}\n".format(S_degree))

        # get nodes in P and N
        nodes_in_P = list(P_dict[k])
        nodes_in_N = list(N_dict[k])

        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        seeds_array[0, list(S_degree)] = True

        # Get the observed values for the ground truth
        obs_state_array = obs_state_dict[k]
        S = S_original_dict[k]

        # Computing loss
        B_degree_loss_1_array, B_degree_loss_total_array, probability_array = compute_loss_per_timestep(simul, seeds_array, obs_state_array)
        B_degree_n_S_correct = len(set(S_degree).intersection(set(S)))

        B_degree_S_dict[k] = np.array(list(S_degree))
        B_degree_n_S_correct_dict[k] = B_degree_n_S_correct
        B_degree_loss_1_dict[k] = B_degree_loss_1_array
        B_degree_loss_total_dict[k] = B_degree_loss_total_array

        # compute hit ratios
        P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio = get_pos_hit_neg_hit(nodes_in_P, nodes_in_N, probability_array)

        P_hit_dict[k] = P_hit
        N_hit_dict[k] = N_hit
        P_hit_frac_dict[k] = P_hit_frac
        N_hit_frac_dict[k] = N_hit_frac
        P_N_hit_diff_dict[k] = P_N_hit_diff
        P_N_hit_ratio_dict[k] = P_N_hit_ratio

    return B_degree_S_dict, B_degree_n_S_correct_dict, B_degree_loss_1_dict, B_degree_loss_total_dict, \
             P_hit_dict, N_hit_dict, P_hit_frac_dict, N_hit_frac_dict, P_N_hit_diff_dict, P_N_hit_ratio_dict

def prepare_GT_data(simul, k_range, people_nodes_idx):
    # Keep sampling until at least two additional infection is observed at the last timestep
    # while True:
        # S_original_dict, seeds_array_dict, obs_state_dict, I1_dict = random_seed_and_observe_infections(simul, k_range, people_nodes_idx)
        # S_original_dict, seeds_array_dict, obs_state_dict, I1_dict = random_seed_and_observe_infections(simul, k_range, people_nodes_idx)
        # temp = [len(I1_dict[k][-1]) > k+1 for k in k_range]
        # if sum(temp) == len(k_range):
            # break

    # Using the original seed set, run simulation for 100 times. For each run, compute P, N, and the score per run (score is computed by itself vs all remaining)
    seeds_array_dict = dict()
    S_original_dict = dict()
    P_dict = dict()
    N_dict = dict()
    obs_state_dict = dict() # replace this with the best GT.
    I1_dict = dict() # replace this with the best GT.

    n_replicates = 100
    for k in k_range:
    # for k, seeds_array in seeds_array_dict.items():
        while True: # Ensure that |P| is large enough. Maybe at least 3+k ?
            no_of_S_original = k
            S_original = np.random.choice(a=people_nodes_idx, size=k, replace=False)
            seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            # Set nodes in S_original at time 0 as seeds.
            seeds_array[0, S_original] = True
            # S_original_dict, seeds_array_dict, _, _ = random_seed_and_observe_infections_on_k(simul, k, people_nodes_idx)
            S_original_dict[k] = S_original
            seeds_array_dict[k] = seeds_array

            probability_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(np.float32)
            infection_array = np.zeros((n_replicates, simul.n_timesteps, simul.number_of_nodes)).astype(bool)
            simul.set_seeds(seeds_array)
            for rep in range(n_replicates):
                simul.simulate()
                probability_array[rep, :, :] = simul.probability_array
                infection_array[rep, :, :] = simul.infection_array

            F1_array = np.zeros((n_replicates))
            MCC_array = np.zeros((n_replicates))
            V = set(people_nodes_idx)

            for rep_num in range(n_replicates):
                P = set(infection_array[rep_num,-1,:].nonzero()[0])
                N = V - P
                idx_except_itself = [idx for idx in range(n_replicates) if idx != rep_num]
                probability_array_except_itself = probability_array[idx_except_itself, :, :]
                P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio = get_pos_hit_neg_hit(list(P), list(N), probability_array_except_itself)
                TP = P_hit
                TN = len(N) - N_hit
                FP = N_hit
                FN = len(P) - P_hit
                F1 = TP / (TP + 0.5*(FP+FN))
                MCC = get_MCC(TP, TN, FP, FN)
                F1_array[rep_num] = F1
                MCC_array[rep_num] = MCC

            # Selecting the simulation that is the best 
            rep_idx_best = MCC_array.argmax()
            P = set(infection_array[rep_idx_best,-1,:].nonzero()[0])
            print("GT. k={}, |P|={}".format(k, len(P)))
            N = V - P
            P_dict[k] = P
            N_dict[k] = N
            obs_state = infection_array[rep_idx_best,:,:]


            obs_state_dict[k] = obs_state
            I1_sets = []
            for t in range(simul.n_timesteps):
                I1 = set(obs_state[t].nonzero()[0])
                I1_sets.append(I1)
            I1_dict[k] = I1_sets
            # if len(P) >= 3+k: # Ensure that |P| is large enough. Maybe at least 3+k ?
            if len(P) >= 1+k: # Ensure that |P| is large enough. Maybe at least one additional infection?
                print("----------------------")
                break

    return S_original_dict, seeds_array_dict, obs_state_dict, I1_dict, P_dict, N_dict 

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
    dose_response = args.dose_response

    np.set_printoptions(suppress=True)
    n_timesteps = 10
    n_replicates = 1
    rho = 0.4
    d = 0.1
    contact_area = 30
    area_people = 2000 # area of patient. 2000cm^2
    area_location = 40000 # area of room. 40000cm^2

    ####################################################################
    print("Load network")
    if name == "Karate_temporal":
        q = 2
        pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_karate_temporal_network(area_people, area_location)
    elif name == "UVA_temporal":
        q = 2
        pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UVA_temporal_network(area_people, area_location)
    elif name == "UIHC_Jan2010_patient_room_temporal":
        q = 2
        pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UIHC_Jan2010_patient_room_temporal_network(area_people, area_location)
    elif name == "UIHC_HCP_patient_room":
        q = 2
        pi = 1.0 # pi is the infectivity. f(x) = 1 - e ^ {- pi * load}

        year = args.year
        sampled = args.sampled
        if sampled:
            name = "{}_{}_sampled".format(name, year)
        else:
            contact_area = 10
            name = "{}_{}".format(name, year)
        # if year = 2011 # Use non-overlap data.
        # if sampled = True # Use the subgraph. Sampled based on the unit with the most number of CDI cases.
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UIHC_HCP_patient_room_temporal_network(year, sampled, area_people, area_location)
    ####################################################################
    # 0. Create simulation instance with empty seeds list
    simul = Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, dose_response)

    ####################################################################
    # Set random seed, and observe infections
    # 1. Data generation
    print("Generating data for ISCK...")
    # k_range = [2, 4, 6]
    k_range = [1,2,3]
    S_original_dict, seeds_array_dict, obs_state_dict, I1_dict, P_dict, N_dict = prepare_GT_data(simul, k_range, people_nodes_idx)

    ####################################################################
    # 1.1 Get the nodes in P and the nodes in N
    # Evaluation metrics will be computed based on the number of hits in people nodes in these sets
    # P_dict, N_dict = get_P_N(people_nodes_idx, k_range, I1_dict)

    ####################################################################
    # 2. Compute ground truth loss per timestep
    # We're not interested in loss over timestep (e.g. missing infection) in this project, so just take the loss at the last timestep.
    print("Compute GT losses")
    n_replicates = 10
    GT_loss_1_dict, GT_loss_total_dict, \
            GT_P_hit_dict, GT_N_hit_dict, GT_P_hit_frac_dict, GT_N_hit_frac_dict, GT_P_N_hit_diff_dict, GT_P_N_hit_ratio_dict = \
            compute_GT_loss_per_timestep(simul, seeds_array_dict, obs_state_dict, n_replicates, P_dict, N_dict)

    ####################################################################
    # Baselines
    # Randomly selected seed out of people nodes
    B_random_S_dict, B_random_n_S_correct_dict, B_random_loss_1_dict, B_random_loss_total_dict, \
            B_random_P_hit_dict, B_random_N_hit_dict, B_random_P_hit_frac_dict, B_random_N_hit_frac_dict, B_random_P_N_hit_diff_dict, B_random_P_N_hit_ratio_dict = \
            run_B_random_report_loss_per_timestep()

    # Degree centrality out of people nodes
    B_degree_S_dict, B_degree_n_S_correct_dict, B_degree_loss_1_dict, B_degree_loss_total_dict, \
            B_degree_P_hit_dict, B_degree_N_hit_dict, B_degree_P_hit_frac_dict, B_degree_N_hit_frac_dict, B_degree_P_N_hit_diff_dict, B_degree_P_N_hit_ratio_dict = \
            run_B_degree_report_loss_per_timestep()


    ####################################################################
    # 3. Greedy source detection
    MTP = (False, -1) # Do not use multicores in replicates. Not implemented yet.
    print("Run greedy source detection, compute loss per timestep for the best nodeset")
    focus_obs1_list = [True, False] # focus_obs1 = True is using loss_1 in the optimization. focus_obs1 = False is using loss_total
    GS_S_detected_dict, GS_detected_seeds_array_dict, GS_n_S_correct_dict, GS_probability_array_dict, GS_loss_1_dict, GS_loss_total_dict, \
        GS_P_hit_dict, GS_N_hit_dict, GS_P_hit_frac_dict, GS_N_hit_frac_dict, GS_P_N_hit_diff_dict, GS_P_N_hit_ratio_dict = \
        run_greedy_source_detection_report_loss_per_timestep(simul, focus_obs1_list, k_range, obs_state_dict, seeds_array_dict, n_replicates, MTP, P_dict, N_dict)

    ####################################################################
    # 4. ISCK
    print("\n ISCK on ratio")
    ISCK_S_detected_dict, ISCK_n_S_correct_dict, ISCK_loss_1_dict, ISCK_loss_total_dict, \
        ISCK_P_hit_dict, ISCK_N_hit_dict, ISCK_P_hit_frac_dict, ISCK_N_hit_frac_dict, ISCK_P_N_hit_diff_dict, ISCK_P_N_hit_ratio_dict \
        = run_ISCK_report_loss_per_timestep(how="ratio")

    ####################################################################
    # 4.1 ISCK variation (diff)
    print("\n ISCK on diff")
    ISCK_diff_S_detected_dict, ISCK_diff_n_S_correct_dict, ISCK_diff_loss_1_dict, ISCK_diff_loss_total_dict, \
        ISCK_diff_P_hit_dict, ISCK_diff_N_hit_dict, ISCK_diff_P_hit_frac_dict, ISCK_diff_N_hit_frac_dict, ISCK_diff_P_N_hit_diff_dict, ISCK_diff_P_N_hit_ratio_dict \
        = run_ISCK_report_loss_per_timestep(how="diff")

    ####################################################################
    # 5. Save results 
    folder = "ISCK_temporal"
    df_GT, df_B_random, df_B_degree, df_exp1, df_exp2, df_ISCK, df_ISCK_diff, \
            df_GT_loss_total, df_GT_loss_1, df_GS_loss_total_exp1, df_GS_loss_total_exp2, df_ISCK_loss_total, df_ISCK_diff_loss_total, \
            df_GS_loss_1_exp1, df_GS_loss_1_exp2, df_ISCK_loss_1, df_ISCK_diff_loss_1 = prepare_result_dataframes()

    print("Ground truth")
    print(df_GT[["k", "len(P)", "n_S_correct", "P_hit", "P_hit_frac", "N_hit", "N_hit_frac", "P_N_hit_diff", "P_N_hit_ratio", "E_F1", "E_MCC"]].round(2))
    print("Baseline. Random seed")
    print(df_B_random[["k", "n_S_correct", "P_hit", "P_hit_frac", "N_hit", "N_hit_frac", "P_N_hit_diff", "P_N_hit_ratio", "E_F1", "E_MCC"]].round(2))
    print("Baseline. Seed based on the degree centrality")
    print(df_B_degree[["k", "n_S_correct", "P_hit", "P_hit_frac", "N_hit", "N_hit_frac", "P_N_hit_diff", "P_N_hit_ratio", "E_F1", "E_MCC"]].round(2))
    print("Greedy source detection. exp1")
    print(df_exp1[["k", "GS_n_S_correct", "P_hit", "P_hit_frac", "N_hit", "N_hit_frac", "P_N_hit_diff", "P_N_hit_ratio", "E_F1", "E_MCC"]].round(2))
    print("Greedy source detection. exp2")
    print(df_exp2[["k", "GS_n_S_correct", "P_hit", "P_hit_frac", "N_hit", "N_hit_frac", "P_N_hit_diff", "P_N_hit_ratio", "E_F1", "E_MCC"]].round(2))
    print("ISCK")
    print(df_ISCK[["k", "n_S_correct", "P_hit", "P_hit_frac", "N_hit", "N_hit_frac", "P_N_hit_diff", "P_N_hit_ratio", "E_F1", "E_MCC"]].round(2))
    print("ISCK_diff")
    print(df_ISCK_diff[["k", "n_S_correct", "P_hit", "P_hit_frac", "N_hit", "N_hit_frac", "P_N_hit_diff", "P_N_hit_ratio", "E_F1", "E_MCC"]].round(2))

    save_result_dataframes(folder, name, dose_response)

