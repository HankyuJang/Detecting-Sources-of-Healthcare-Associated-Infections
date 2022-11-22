"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last Modified: Feb, 2020 

Description: Greedy source detection. Sources at time 0 and time 1. Observe infections over time
"""
from utils.load_network import *
from utils.greedy_source_detection import *
from utils.random_seed import *
from utils.GT_loss import *
from simulator_load_sharing import *

import pandas as pd
import argparse
import math
import random as random
import copy

def prepare_df_exp_t0t1(focus_obs1, k_range, obs_state_dict, S1_dict, S2_dict, S1_detected_dict, S2_detected_dict, n_S_correct_dict):
    df_exp = pd.DataFrame(
            {"k":[k for k in k_range],
                "n_obs_state": [obs_state[n_timesteps-1].nonzero()[0].shape[0] for obs_state in obs_state_dict.values()],
                "S1": [sorted(list(S1)) for S1 in S1_dict.values()],
                "S2": [sorted(list(S2)) for S2 in S2_dict.values()],
                "S1_detected": [sorted(list(S1_detected)) for S1_detected in S1_detected_dict[focus_obs1].values()],
                "S2_detected": [sorted(list(S2_detected)) for S2_detected in S2_detected_dict[focus_obs1].values()],
                "n_S_correct": list(n_S_correct_dict[focus_obs1].values()),
                "obs_state": [list(obs_state[n_timesteps-1].nonzero()[0]) for obs_state in obs_state_dict.values()]
                })
    return df_exp

def prepare_result_dataframes_t0t1():
    df_exp1 = prepare_df_exp_t0t1(True, k_range, obs_state_dict, S1_dict, S2_dict, S1_detected_dict, S2_detected_dict, n_S_correct_dict)
    df_exp2 = prepare_df_exp_t0t1(False, k_range, obs_state_dict, S1_dict, S2_dict, S1_detected_dict, S2_detected_dict, n_S_correct_dict)
    df_GT_loss_total = pd.DataFrame(data=GT_loss_total_dict)
    df_GT_loss_1 = pd.DataFrame(data=GT_loss_1_dict)
    df_loss_total_exp1 = pd.DataFrame(data=loss_total_dict[True])
    df_loss_total_exp2 = pd.DataFrame(data=loss_total_dict[False])
    df_loss_1_exp1 = pd.DataFrame(data=loss_1_dict[True])
    df_loss_1_exp2 = pd.DataFrame(data=loss_1_dict[False])
    return df_exp1, df_exp2, df_GT_loss_total, df_GT_loss_1, df_loss_total_exp1, df_loss_total_exp2, df_loss_1_exp1, df_loss_1_exp2

def save_result_dataframes_t0t1(folder, name):
    # Save datasets
    df_exp1.to_csv("../tables/{}/{}/{}_exp1.csv".format(folder, name, folder), index=False)
    df_exp2.to_csv("../tables/{}/{}/{}_exp2.csv".format(folder, name, folder), index=False)
    df_GT_loss_total.to_csv("../tables/{}/{}/{}_GT_loss_total.csv".format(folder, name, folder), index=False)
    df_GT_loss_1.to_csv("../tables/{}/{}/{}_GT_loss_1.csv".format(folder, name, folder), index=False)
    df_loss_total_exp1.to_csv("../tables/{}/{}/{}_loss_total_exp1.csv".format(folder, name, folder), index=False)
    df_loss_total_exp2.to_csv("../tables/{}/{}/{}_loss_total_exp2.csv".format(folder, name, folder), index=False)
    df_loss_1_exp1.to_csv("../tables/{}/{}/{}_loss_1_exp1.csv".format(folder, name, folder), index=False)
    df_loss_1_exp2.to_csv("../tables/{}/{}/{}_loss_1_exp2.csv".format(folder, name, folder), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='greedy source detection, missing infection')
    parser.add_argument('-name', '--name', type=str, default="Karate",
                        help= 'network to use. Karate | UIHC_Jan2010_patient_room')
    args = parser.parse_args()
    name = args.name

    np.set_printoptions(suppress=True)
    n_timesteps = 30
    n_replicates = 1
    rho = 0.4
    d = 0.1
    q = 1
    # contact_area = 150 # contact area. 150cm^2
    contact_area = 10
    area_people = 2000 # area of patient. 2000cm^2
    area_location = 40000 # area of room. 40000cm^2

    ####################################################################
    print("Load network")
    if name == "Karate":
        G, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_karate_network(area_people, area_location)
    elif name == "UIHC_Jan2010_patient_room":
        G, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UIHC_Jan2010_patient_room_network(area_people, area_location)
    print(nx.info(G))
    ####################################################################
    # 0. Create simulation instance with empty seeds list
    simul = Simulation(G, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q)

    ####################################################################
    # Set random seed, and observe infections
    # 1. Data generation
    print("Generating data for greedy source detection")
    k_range = [1, 2, 3]
    S1_dict, S2_dict, seeds_array_dict, obs_state_dict, I1_dict = random_seed1_seed2_and_observe_infections(simul, k_range, people_nodes_idx)
    ####################################################################
    # 2. Compute ground truth loss per timestep
    print("Compute GT losses")
    n_replicates = 10
    GT_loss_1_dict, GT_loss_total_dict = compute_GT_loss_per_timestep(simul, seeds_array_dict, obs_state_dict, n_replicates)

    ####################################################################
    # 3. Experiment
    MTP = (False, -1) # Do not use multicores in replicates.
    # MTP = (True, 10)
    print("Run greedy source detection, compute loss per timestep for the best nodeset")
    focus_obs1_list = [True, False]
    S1_detected_dict, S2_detected_dict, detected_seeds_array_dict, n_S_correct_dict, probability_array_dict, loss_1_dict, loss_total_dict = run_greedy_source_detection_t0t1_report_loss_per_timestep(simul, focus_obs1_list, k_range, obs_state_dict, seeds_array_dict, n_replicates, MTP)

    ####################################################################
    # 4. Save results 
    df_exp1, df_exp2, df_GT_loss_total, df_GT_loss_1, df_loss_total_exp1, df_loss_total_exp2, df_loss_1_exp1, df_loss_1_exp2 = prepare_result_dataframes_t0t1()
    folder = "gs2mi"
    save_result_dataframes_t0t1(folder, name)
