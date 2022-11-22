"""
Author: -
Email: -
Last Modified: Feb 2022

Description: 

This script loads pickle objects and prepare result tables on the case study

Usage

$ python case_study_result_table.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T

"""

import argparse
from tqdm import tqdm

from utils.load_network import *
from prep_result_dataframes import *
import pandas as pd
import pickle

def load_cult(path, infile):
    column_list=['S_detected', 'S_timesteps', 'n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']

    df_cult = pd.read_csv(path+infile)
    df_cult.reset_index(drop=True, inplace=False)

    MCC_array = df_cult["MCC"].values
    i_best = np.nanargmax(MCC_array)

    return df_cult.iloc[[i_best]][column_list]

def load_GT_results(GT_output_dict):

    GT_observation_dict = GT_output_dict["GT_observation_dict"]
    seeds_array = GT_observation_dict["seeds_array"]
    number_of_seeds_over_time = GT_observation_dict["number_of_seeds_over_time"]

    GT_n_S = np.sum(number_of_seeds_over_time)

    GT_evaluation_dict = GT_output_dict["GT_evaluation_dict"]
    # GT_evaluation_dict["loss_1"] = GT_loss_1
    # GT_evaluation_dict["loss_total"] = GT_loss_total
    # GT_evaluation_dict["list_of_P_hit"] = GT_list_of_P_hit
    # GT_evaluation_dict["list_of_N_hit"] = GT_list_of_N_hit
    GT_TP = GT_evaluation_dict["TP"]
    GT_TN = GT_evaluation_dict["TN"]
    GT_FP = GT_evaluation_dict["FP"]
    GT_FN = GT_evaluation_dict["FN"]
    GT_F1 = GT_evaluation_dict["F1"]
    GT_MCC = GT_evaluation_dict["MCC"]
    GT_time_elapsed = GT_evaluation_dict["time_elapsed"]

    df_GT = prepare_df_exp(seeds_array, GT_n_S, GT_n_S, \
            GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC, GT_time_elapsed)

    return df_GT

def load_B_random_results(B_random_evaluation_rep_dict):

    L_n_S, L_n_S_correct, L_TP, L_TN, L_FP, L_FN, L_F1, L_MCC, L_time_elapsed = initilize_n_empty_lists(9)
    L_reps = B_random_evaluation_rep_dict.keys()
    L_S_detected, L_S_timesteps = [], []
    for rep in L_reps:
        B_random_evaluation_dict = B_random_evaluation_rep_dict[rep]
        L_n_S.append(B_random_evaluation_dict["n_S"])
        L_n_S_correct.append(B_random_evaluation_dict["n_S_correct"])
        L_TP.append(B_random_evaluation_dict["TP"])
        L_TN.append(B_random_evaluation_dict["TN"])
        L_FP.append(B_random_evaluation_dict["FP"])
        L_FN.append(B_random_evaluation_dict["FN"])
        L_F1.append(B_random_evaluation_dict["F1"])
        L_MCC.append(B_random_evaluation_dict["MCC"])
        L_time_elapsed.append(B_random_evaluation_dict["time_elapsed"])

        # Prep this for hop evaluation
        detected_seeds_array = B_random_evaluation_dict["seeds_array"]
        S_detected = list(detected_seeds_array.nonzero()[1])
        S_timesteps = list(detected_seeds_array.nonzero()[0])
        L_S_detected.append(S_detected)
        L_S_timesteps.append(S_timesteps)

    # NOTE: Since it's taking avg of 30 repetitions, return empty seed
    B_random_seeds_array = B_random_evaluation_dict["seeds_array"]
    B_random_seeds_array[:] = 0

    # MCC can be nan if div by 0.
    array_MCC = np.array(L_MCC)
    MCC_mean = np.nanmean(array_MCC)

    df_B_random = prepare_df_exp(B_random_seeds_array, L_avg(L_n_S), L_avg(L_n_S_correct), \
            L_avg(L_TP), L_avg(L_TN), L_avg(L_FP), L_avg(L_FN), L_avg(L_F1), MCC_mean, L_avg(L_time_elapsed))

    return df_B_random, L_S_detected, L_S_timesteps

def load_Px_results(Px_evaluation_dict):

    column_list=['S_detected', 'S_timesteps', 'n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']

    # For P1
    if "df_greedy" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_greedy"]
    # For P2
    elif "df_ISCK_results" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_ISCK_results"]
    # For P3
    elif "df_greedy_ratio" in Px_evaluation_dict:
        df_algo_results = Px_evaluation_dict["df_greedy_ratio"]

    df_algo_results = df_algo_results.reset_index(drop=True, inplace=False)
    MCC_array = df_algo_results["MCC"].values
    i_best = np.nanargmax(MCC_array)

    return df_algo_results.iloc[[i_best]][column_list]
    # seeds_array = Px_greedy_evaluation_dict["seeds_array"]
    # n_S = Px_greedy_evaluation_dict["n_S"]
    # n_S_correct = Px_greedy_evaluation_dict["n_S_correct"]
    # TP = Px_greedy_evaluation_dict["TP"]
    # TN = Px_greedy_evaluation_dict["TN"]
    # FP = Px_greedy_evaluation_dict["FP"]
    # FN = Px_greedy_evaluation_dict["FN"]
    # F1 = Px_greedy_evaluation_dict["F1"]
    # MCC = Px_greedy_evaluation_dict["MCC"]
    # time_elapsed = Px_greedy_evaluation_dict["time_elapsed"]

    # df_Px = prepare_df_exp(seeds_array, n_S, n_S_correct, \
            # TP, TN, FP, FN, F1, MCC, time_elapsed)
    # return df_Px

def initilize_n_empty_lists(n):
    list_to_return = []
    for i in range(n):
        list_to_return.append([])
    return list_to_return

def L_avg(a):
    return sum(a) / len(a)

def get_infile_name_and_name_list(args, graph_name):
    if args.dose_response=="exponential":
        txt_dose_response = ""
    elif args.dose_response=="linear":
        txt_dose_response = "linear_"

    P1_infile_name_list = [
            "{}P1_greedy_evalution_lazyT_expectedT.pickle".format(txt_dose_response)
            ]
    P1_name_list = [
            "{}P1_E_LazyGreedy".format(txt_dose_response),
            ]
    P2_infile_name_list = [
            "{}P2_ISCK_greedy_evalution_lazyT_expectedT.pickle".format(txt_dose_response),
            "{}P2_ISCK_multiplicative_update_evalution_lazyT_expectedT.pickle".format(txt_dose_response)
            ]
    P2_name_list = [
            "{}P2_E_LazyISCK_greedy".format(txt_dose_response),
            "{}P2_E_LazyISCK_multiplicative_update".format(txt_dose_response)
            ]
    P3_infile_name_list = [
            "{}P3_GR_gconstraintF_evalution_lazyF_expectedT.pickle".format(txt_dose_response),
            "{}P3_GR_gconstraintT_evalution_lazyF_expectedT.pickle".format(txt_dose_response),
            ]
    P3_name_list = [
            "{}P3_E_GreedyRatio".format(txt_dose_response),
            "{}P3_E_GreedyRatio_ghit>50".format(txt_dose_response),
            ]
    return P1_infile_name_list, P1_name_list, P2_infile_name_list, P2_name_list, P3_infile_name_list, P3_name_list

def recompute_n_S_correct(df_result, start_idx_of_algo):
    n_S_correct_list = list(df_result["n_S_correct"].values[:start_idx_of_algo]) # get those till random baseline

    GT_S_detected = eval(df_result.loc["GT"]["S_detected"])
    set_GT_S_detected = set(GT_S_detected)

    S_detected_list_algo = list(df_result["S_detected"].values[start_idx_of_algo:])
    for S_detected in S_detected_list_algo:
        set_S_detected = set(eval(S_detected))
        n_S_correct = len(set_GT_S_detected.intersection(set_S_detected))
        n_S_correct_list.append(n_S_correct)
    df_result["n_S_correct"] = n_S_correct_list

def compute_hop_v1(G_over_time, df_result, node_idx_to_name_mapping, P1_name_list, P2_name_list, P3_name_list):
    # Compute avg hop distance of detected seeds to GT source
    # NOTE: this implementation assumes that GT seeds are at time 0 and time 1
    GT_S = eval(df_result.loc["GT"]["S_detected"])
    GT_S_timesteps = eval(df_result.loc["GT"]["S_timesteps"])

    GT_S_at_t0 = []
    GT_S_at_t1 = []
    for seed, time in zip(GT_S, GT_S_timesteps):
        seed = node_idx_to_name_mapping[seed]
        if time == 0:
            GT_S_at_t0.append(seed)
        elif time == 1:
            GT_S_at_t1.append(seed)

    n_GT_S_at_t0 = len(GT_S_at_t0)
    n_GT_S_at_t1 = len(GT_S_at_t1)

    G0 = G_over_time[0]
    G1 = G_over_time[1]
    P123_name_list = P1_name_list + P2_name_list + P3_name_list

    print("computing hop distances...")
    avg_of_min_hop_list = [0, np.nan] # GT and random.
    for algo_name in P123_name_list:
    # for algo_name in ["linear_P1_Greedy"]:
        algo_S_detected = eval(df_result.loc[algo_name]["S_detected"])
        algo_S_detected = np.unique(np.array(algo_S_detected)) # Remove duplicates (there could be same seed in time 0 and time 1)

        n_algo_S_detected = algo_S_detected.shape[0]

        hop_array_seed_to_GT_seed_at_t0 = np.zeros((n_algo_S_detected, n_GT_S_at_t0))
        hop_array_seed_to_GT_seed_at_t1 = np.zeros((n_algo_S_detected, n_GT_S_at_t1))

        for idx_seed, seed in enumerate(algo_S_detected):
            seed = node_idx_to_name_mapping[seed]
            # Compute min hop at time 0
            for idx_GT_seed, GT_seed in enumerate(GT_S_at_t0):
                try:
                    seed_to_GT_seed_hop = nx.shortest_path_length(G0, source=seed, target=GT_seed)
                except Exception as e:
                    print("{}\n".format(e))
                    seed_to_GT_seed_hop = np.nan
                hop_array_seed_to_GT_seed_at_t0[idx_seed, idx_GT_seed] = seed_to_GT_seed_hop

            # Compute min hop at time 1
            for idx_GT_seed, GT_seed in enumerate(GT_S_at_t1):
                try:
                    seed_to_GT_seed_hop = nx.shortest_path_length(G1, source=seed, target=GT_seed)
                except Exception as e:
                    print("{}\n".format(e))
                    seed_to_GT_seed_hop = np.nan
                hop_array_seed_to_GT_seed_at_t1[idx_seed, idx_GT_seed] = seed_to_GT_seed_hop

        # Now, for each detected seed, get the closest distance to GT seeds. 
        # Then take average of such distances over detected seeds
        hop_array_seed_to_GT_seed = np.concatenate((hop_array_seed_to_GT_seed_at_t0, hop_array_seed_to_GT_seed_at_t1), axis=1)
        print("hop_array_seed_to_GT_seed.shape: {}".format(hop_array_seed_to_GT_seed.shape))

        min_hop_array = np.nanmin(hop_array_seed_to_GT_seed, axis=1)
        print("min_hop_array: {}".format(min_hop_array))

        avg_of_min_hop = np.nanmean(min_hop_array)

        avg_of_min_hop_list.append(avg_of_min_hop)

    print("Complete: computing hop distances...")

    df_result.insert(loc=4, column="hops", value=avg_of_min_hop_list)

def get_S_list_of_tuples(df_result, algo_name):
    S_list_of_tuples = []

    S = eval(df_result.loc[algo_name]["S_detected"])
    S_timesteps = eval(df_result.loc[algo_name]["S_timesteps"])
    for seed, time in zip(S, S_timesteps):
        seed = node_idx_to_name_mapping[seed]
        S_list_of_tuples.append((seed, time))
    print("S_list_of_tuples: {}".format(S_list_of_tuples))
    return S_list_of_tuples

def get_GT_S_list_of_tuples(df_result):
    GT_S_list_of_tuples = []

    GT_S = eval(df_result.loc["GT"]["S_detected"])
    GT_S_timesteps = eval(df_result.loc["GT"]["S_timesteps"])
    for seed, time in zip(GT_S, GT_S_timesteps):
        seed = node_idx_to_name_mapping[seed]
        GT_S_list_of_tuples.append((seed, time))
    print("GT_S_list_of_tuples: {}".format(GT_S_list_of_tuples))
    return GT_S_list_of_tuples

def get_L_B_random_list_of_tuples(L_B_random_S_detected, L_B_random_S_timesteps):
    print("prepare node tuples for baselines")
    n_rep_B_random = len(L_B_random_S_detected)
    L_B_random_list_of_tuples = []

    for B_random_S_detected, B_random_S_timesteps in zip(L_B_random_S_detected, L_B_random_S_timesteps):
        B_random_list_of_tuples = []

        for seed, time in zip(B_random_S_detected, B_random_S_timesteps):
            seed = node_idx_to_name_mapping[seed]
            B_random_list_of_tuples.append((seed, time))
        L_B_random_list_of_tuples.append(B_random_list_of_tuples)
    print("L_B_random_list_of_tuples: {}".format(L_B_random_list_of_tuples))
    return L_B_random_list_of_tuples

def get_algo_P123_list_of_tuples(df_result, P123_name_list):
    print("prepare node tuples for our algorithms")
    algo_P123_list_of_tuples = []
    for algo_name in P123_name_list:
        algo_list_of_tuples = []

        algo_S_detected = eval(df_result.loc[algo_name]["S_detected"])
        algo_S_timesteps = eval(df_result.loc[algo_name]["S_timesteps"])
        for seed, time in zip(algo_S_detected, algo_S_timesteps):
            seed = node_idx_to_name_mapping[seed]
            algo_list_of_tuples.append((seed, time))
        algo_P123_list_of_tuples.append(algo_list_of_tuples)
    # print("algo_P123_list_of_tuples: {}".format(algo_P123_list_of_tuples))
    return algo_P123_list_of_tuples

def get_H_large(G_over_time):
    H_over_time = []

    print("Relabeling nodes...")
    G_node_list = [v for v in G_over_time[0].nodes]
    for t, G in enumerate(G_over_time):
        mapping = dict([(v, (v, t)) for v in G.nodes])
        H = nx.relabel_nodes(G, mapping)
        H_over_time.append(H)

    # Connect nodes in time t to t+1
    print("Copy nodes and edges to H_large...")
    H_large = nx.Graph()
    for t, H in enumerate(H_over_time):
        H_nodes = [v for v in H.nodes]
        H_large.add_nodes_from(H_nodes)
        print(nx.info(H_large))

        H_edges = [e for e in H.edges]
        H_large.add_edges_from(H_edges)
        print(nx.info(H_large))

    print("Add edges over time on same node")
    for t in range(len(H_over_time)-1):
        H_prev = H_over_time[t]
        H_next = H_over_time[t+1]

        H_prev_nodes = [v for v in H_prev]
        H_next_nodes = [v for v in H_next]

        H_edges_over_time = [(v1, v2) for v1, v2 in zip(H_prev_nodes, H_next_nodes)]
        H_large.add_edges_from(H_edges_over_time)
        print(nx.info(H_large))
    return H_large

def get_shortest_path_fromGT_source_nodes(H_large, GT_S_list_of_tuples):
    print("Precompute shortest path from GT source nodes to all the remaining nodes")
    sp_from_GT_S_tuples = dict()
    for GT_S_tuple in tqdm(GT_S_list_of_tuples):
        p = nx.shortest_path_length(H_large, source=GT_S_tuple)
        sp_from_GT_S_tuples[GT_S_tuple] = p
    print("Complete - shortest path from GT source nodes to all the remaining nodes")
    return sp_from_GT_S_tuples

def get_avg_of_min_hop(detected_S_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples):
    n_detected_S_S_detected = len(detected_S_list_of_tuples)
    hop_array_GT_seed_to_detected_S_seed = np.zeros((n_detected_S_S_detected, n_GT_S))
    for idx_detected_S_S, detected_S_S_tuple in enumerate(detected_S_list_of_tuples):
        for idx_GT_S, GT_S_tuple in enumerate(GT_S_list_of_tuples):
            try:
                distance = sp_from_GT_S_tuples[GT_S_tuple][detected_S_S_tuple]
            except Exception as e:
                print("{}\n".format(e))
                distance = np.nan
            hop_array_GT_seed_to_detected_S_seed[idx_detected_S_S][idx_GT_S] = distance
    print("hop_array_GT_seed_to_detected_S_seed.shape: {}".format(hop_array_GT_seed_to_detected_S_seed.shape))
    # NOTE: set this to zero gives min hop from each GT seed.
    # NOTE: set this to one gives min hop from each seed from our algo
    min_hop_array = np.nanmin(hop_array_GT_seed_to_detected_S_seed, axis=0) 
    print("min_hop_array: {}".format(min_hop_array))
    avg_of_min_hop = np.nanmean(min_hop_array)
    return avg_of_min_hop

# def compute_avg_of_min_hop_list(GT_S_list_of_tuples, L_B_random_list_of_tuples, cult_list_of_tuples, \
        # algo_P123_list_of_tuples):
def compute_avg_of_min_hop_list(GT_S_list_of_tuples, algo_P123_list_of_tuples):

    avg_of_min_hop_list = [0] # GT

    # n_our_algorithms = len(P123_name_list)
    n_GT_S = len(GT_S_list_of_tuples)

    n_our_algorithms = len(algo_P123_list_of_tuples)

    # n_rep_B_random = len(L_B_random_list_of_tuples)

    # ------------------------------------------
    # print("Get the hops Baseline random")
    # B_random_avg_of_min_hop = np.zeros((n_rep_B_random))
    # for idx_rep, B_random_list_of_tuples in enumerate(L_B_random_list_of_tuples):
        # avg_of_min_hop = get_avg_of_min_hop(B_random_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples)
        # B_random_avg_of_min_hop[idx_rep] = avg_of_min_hop
    # print("B_random_avg_of_min_hop: \n{}".format(B_random_avg_of_min_hop))
    # avg_of_min_hop_list.append(np.nanmean(B_random_avg_of_min_hop))

    # ------------------------------------------
    # print("Get the hops cult")
    # avg_of_min_hop = get_avg_of_min_hop(cult_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples)
    # avg_of_min_hop_list.append(avg_of_min_hop)

    # ------------------------------------------
    print("Get the hops our algo")
    for algo_list_of_tuples in algo_P123_list_of_tuples:
        avg_of_min_hop = get_avg_of_min_hop(algo_list_of_tuples, GT_S_list_of_tuples, n_GT_S, sp_from_GT_S_tuples)
        avg_of_min_hop_list.append(avg_of_min_hop)

    return avg_of_min_hop_list

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
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    args = parser.parse_args()

    n_t_for_eval = args.n_t_for_eval
    graph_name = get_graph_name(args)

    P1_infile_name_list, P1_name_list, P2_infile_name_list, P2_name_list, P3_infile_name_list, P3_name_list = get_infile_name_and_name_list(args, graph_name)

    np.set_printoptions(suppress=True)

    # -------------------------------------------
    path = "../tables/case_study/{}/".format(graph_name)

    list_of_df = []
    list_of_algorithm = []

    print("GT ...")
    infile = "GT_observation_evalution.pickle"
    with open(path + infile, 'rb') as handle:
        GT_output_dict = pickle.load(handle)

    n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

    df_GT = load_GT_results(GT_output_dict)
    list_of_df.append(df_GT)
    list_of_algorithm.append("GT")

    # print("Baseline random ...")
    # infile = "B_random_evalution_30rep.pickle"
    # with open(path + infile, 'rb') as handle:
        # B_random_evaluation_rep_dict = pickle.load(handle)
    # df_B_random, L_B_random_S_detected, L_B_random_S_timesteps = load_B_random_results(B_random_evaluation_rep_dict)
    # list_of_df.append(df_B_random)
    # list_of_algorithm.append("Random(30rep)")

    # print("Baseline cult ...")
    # if args.dose_response=="exponential":
        # infile = "cult.csv"
    # elif args.dose_response=="linear":
        # infile = "linear_cult.csv"
    # df_cult = load_cult(path, infile)
    # list_of_df.append(df_cult)
    # list_of_algorithm.append("cult")

    print("P1 ...")
    for infile in P1_infile_name_list:
        with open(path + infile, 'rb') as handle:
            P1_greedy_evaluation_dict = pickle.load(handle)
        df_P1_greedy = load_Px_results(P1_greedy_evaluation_dict)
        list_of_df.append(df_P1_greedy)
    list_of_algorithm.extend(P1_name_list)

    print("P2 ...")
    for infile in P2_infile_name_list:
        with open(path + infile, 'rb') as handle:
            P2_ISCK_evaluation_dict = pickle.load(handle)
        df_P2_ISCK = load_Px_results(P2_ISCK_evaluation_dict)
        list_of_df.append(df_P2_ISCK)
    list_of_algorithm.extend(P2_name_list)

    print("P3 ...")
    for infile in P3_infile_name_list:
        with open(path + infile, 'rb') as handle:
            P3_greedy_ratio_evaluation_dict = pickle.load(handle)
        df_P3_greedy_ratio = load_Px_results(P3_greedy_ratio_evaluation_dict)
        list_of_df.append(df_P3_greedy_ratio)
    list_of_algorithm.extend(P3_name_list)

    df_result = concat_result_dataframes(list_of_algorithm, list_of_df)

    #NOTE: n_S_correct seem to have computed incorrectly. Recompute this.
    recompute_n_S_correct(df_result, 2)

    print(df_result[["S_detected", 'n_S', 'n_S_correct', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']].round(3))
    
    # -------------------------------------------
    # These loops saves ground truth observations
    print("Saving GT_observations.csv ...")
    infile = "GT_observation_evalution.pickle"
    with open(path + infile, 'rb') as handle:
        GT_output_dict = pickle.load(handle)

    n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

    df_GT_observations = prepare_ground_truth_table(seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
    df_list_of_P = pd.DataFrame(data={"P_t":list_of_sets_of_P})

    if args.dose_response=="exponential":
        outfile = "GT_observations.csv"
    elif args.dose_response=="linear":
        outfile = "linear_GT_observations.csv"
    df_GT_observations.to_csv(path + outfile, index=False)

    if args.dose_response=="exponential":
        outfile = "list_of_P.csv"
    elif args.dose_response=="linear":
        outfile = "linear_list_of_P.csv"
    df_list_of_P.to_csv(path + outfile, index=True)
    
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, _ = process_data_for_experiments(args, area_people, area_location, flag_increase_area)

    # NOTE: compute_hop_v2
    GT_S_list_of_tuples = get_GT_S_list_of_tuples(df_result)
    # L_B_random_list_of_tuples = get_L_B_random_list_of_tuples(L_B_random_S_detected, L_B_random_S_timesteps)
    # cult_list_of_tuples = get_S_list_of_tuples(df_result, "cult")

    P123_name_list = P1_name_list + P2_name_list + P3_name_list
    algo_P123_list_of_tuples = get_algo_P123_list_of_tuples(df_result, P123_name_list)

    H_large = get_H_large(G_over_time)
    sp_from_GT_S_tuples = get_shortest_path_fromGT_source_nodes(H_large, GT_S_list_of_tuples)

    # avg_of_min_hop_list = compute_avg_of_min_hop_list(GT_S_list_of_tuples, L_B_random_list_of_tuples, cult_list_of_tuples, algo_P123_list_of_tuples)
    avg_of_min_hop_list = compute_avg_of_min_hop_list(GT_S_list_of_tuples, algo_P123_list_of_tuples)

    df_result.insert(loc=4, column="hops", value=avg_of_min_hop_list)
    print(df_result[['S_detected', 'n_S', 'n_S_correct', 'hops', 'TP', 'TN', 'FP', 'FN', 'F1', 'MCC', 'Time(s)']].round(3))

    if args.dose_response=="exponential":
        outfile = "result_concat.csv"
    elif args.dose_response=="linear":
        outfile = "linear_result_concat.csv"
    df_result.to_csv(path + outfile, index=True)

    # -------------------------------------------
    # These loops saves intermediary results for P1 - greedy
    list_of_df = []

    print("P1 ...")
    for name, infile in zip(P1_name_list, P1_infile_name_list):
        with open(path + infile, 'rb') as handle:
            P1_greedy_evaluation_dict = pickle.load(handle)
        df_greedy = P1_greedy_evaluation_dict["df_greedy"]
        df_greedy.insert(loc=0, value=name, column="Algorithm")
        list_of_df.append(df_greedy)

    df_greedy_concat = pd.concat(list_of_df)
    df_greedy_concat = df_greedy_concat.rename(columns={'k': "cardinality_constraint"})

    if args.dose_response=="exponential":
        outfile = "P1_greedy_concat.csv"
    elif args.dose_response=="linear":
        outfile = "P1_greedy_concat.csv"
    df_greedy_concat.to_csv(path + outfile, index=False)

    # -------------------------------------------
    # These loops saves intermediary results for P2 - ISCK
    print("P2...")
    list_of_df = []
    list_of_df_best_constraint = []

    print("P2 ...")
    for name, infile in zip(P2_name_list, P2_infile_name_list):
        with open(path + infile, 'rb') as handle:
            P2_ISCK_evaluation_dict = pickle.load(handle)
        df_ISCK = P2_ISCK_evaluation_dict["df_ISCK_results"]
        df_ISCK.insert(loc=0, value=name, column="Algorithm")
        list_of_df.append(df_ISCK)

        best_constraint_obj = df_ISCK.sort_values(by= "MCC", ascending=False).iloc[0][["Algorithm", "knapsack_constraint"]]
        df_best_constraint = pd.DataFrame(best_constraint_obj).T
        list_of_df_best_constraint.append(df_best_constraint)

    df_ISCK_concat = pd.concat(list_of_df)

    df_ISCK_best_constraint = pd.concat(list_of_df_best_constraint)

    if args.dose_response=="exponential":
        outfile = "P2_ISCK_concat.csv"
    elif args.dose_response=="linear":
        outfile = "linear_P2_ISCK_concat.csv"
    df_ISCK_concat.to_csv(path + outfile, index=False)

    if args.dose_response=="exponential":
        outfile = "P2_ISCK_best_constraint.csv"
    elif args.dose_response=="linear":
        outfile = "linear_P2_ISCK_best_constraint.csv"
    df_ISCK_best_constraint.to_csv(path + outfile, index=False)

    # -------------------------------------------
    # These loops saves intermediary results for greedy ratio
    print("Saving P3_greedy_ratio_concat.csv and f_penalty_arrayx.csv ... in P3_details/")
    list_of_df = []
    list_of_df_best_penalty = []

    print("P3 ...")
    for name, infile in zip(P3_name_list, P3_infile_name_list):
        with open(path + infile, 'rb') as handle:
            P3_greedy_ratio_evaluation_dict = pickle.load(handle)
        df_greedy_ratio = P3_greedy_ratio_evaluation_dict["df_greedy_ratio"]
        df_greedy_ratio.insert(loc=0, value=name, column="Algorithm")
        list_of_df.append(df_greedy_ratio)

        best_penalty_obj = df_greedy_ratio.sort_values(by= "MCC", ascending=False).iloc[0][["Algorithm", "L_penalty_array"]]
        df_best_penalty = pd.DataFrame(best_penalty_obj).T
        list_of_df_best_penalty.append(df_best_penalty)

        dict_of_intermediary_results_dict = P3_greedy_ratio_evaluation_dict["dict_of_intermediary_results_dict"]
        for result_dict_idx in dict_of_intermediary_results_dict.keys():
            df_intermediary_results = pd.DataFrame(data=dict_of_intermediary_results_dict[result_dict_idx])

            if args.dose_response=="exponential":
                outfile = name + "f_penalty_array{}.csv".format(result_dict_idx)
            elif args.dose_response=="linear":
                outfile = "linear_" + name + "f_penalty_array{}.csv".format(result_dict_idx)
            df_intermediary_results.to_csv(path + "P3_details/" + outfile, index=False)

    df_greedy_ratio_concat = pd.concat(list_of_df)

    df_greedy_ratio_best_penalty = pd.concat(list_of_df_best_penalty)

    if args.dose_response=="exponential":
        outfile = "P3_greedy_ratio_concat.csv"
    elif args.dose_response=="linear":
        outfile = "linear_P3_greedy_ratio_concat.csv"
    df_greedy_ratio_concat.to_csv(path + "P3_details/" + outfile, index=False)

    if args.dose_response=="exponential":
        outfile = "P3_greedy_ratio_best_penalty.csv"
    elif args.dose_response=="linear":
        outfile = "linear_P3_greedy_ratio_best_penalty.csv"
    df_greedy_ratio_best_penalty.to_csv(path + outfile, index=False)

    print("GT: idx 214 {}, detected seeds: 753 {}, 6 {}, 409 {}, 35 {}, 664 {}".format(node_idx_to_name_mapping[214],node_idx_to_name_mapping[753], node_idx_to_name_mapping[6],node_idx_to_name_mapping[409], node_idx_to_name_mapping[35], node_idx_to_name_mapping[664]))
    print("P[T-1]: {},    P[T]: {}".format(list_of_sets_of_P[T-1], list_of_sets_of_P[T]))
