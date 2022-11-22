"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads pickle objects and prepare result tables

Usage

To run it on Karate graph,
$ python final_exp_result_tables.py

tables are saved in the folder where pickled objects are at
"""

from utils.load_network import *
from prep_result_dataframes import *
import pandas as pd
import pickle

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

    # NOTE: Since it's taking avg of 30 repetitions, return empty seed
    B_random_seeds_array = B_random_evaluation_dict["seeds_array"]
    B_random_seeds_array[:] = 0

    # MCC can be nan if div by 0.
    array_MCC = np.array(L_MCC)
    MCC_mean = np.nanmean(array_MCC)

    df_B_random = prepare_df_exp(B_random_seeds_array, L_avg(L_n_S), L_avg(L_n_S_correct), \
            L_avg(L_TP), L_avg(L_TN), L_avg(L_FP), L_avg(L_FN), L_avg(L_F1), MCC_mean, L_avg(L_time_elapsed))

    return df_B_random

def load_Px_results(Px_greedy_evaluation_dict):
    seeds_array = Px_greedy_evaluation_dict["seeds_array"]
    n_S = Px_greedy_evaluation_dict["n_S"]
    n_S_correct = Px_greedy_evaluation_dict["n_S_correct"]
    # Px_greedy_evaluation_dict["loss_1"] = Px_greedy_loss_1
    # Px_greedy_evaluation_dict["loss_total"] = Px_greedy_loss_total
    # Px_greedy_evaluation_dict["list_of_P_hit"] = Px_greedy_list_of_P_hit
    # Px_greedy_evaluation_dict["list_of_N_hit"] = Px_greedy_list_of_N_hit
    TP = Px_greedy_evaluation_dict["TP"]
    TN = Px_greedy_evaluation_dict["TN"]
    FP = Px_greedy_evaluation_dict["FP"]
    FN = Px_greedy_evaluation_dict["FN"]
    F1 = Px_greedy_evaluation_dict["F1"]
    MCC = Px_greedy_evaluation_dict["MCC"]
    time_elapsed = Px_greedy_evaluation_dict["time_elapsed"]

    df_Px = prepare_df_exp(seeds_array, n_S, n_S_correct, \
            TP, TN, FP, FN, F1, MCC, time_elapsed)

    return df_Px

def initilize_n_empty_lists(n):
    list_to_return = []
    for i in range(n):
        list_to_return.append([])
    return list_to_return

def L_avg(a):
    return sum(a) / len(a)

if __name__ == "__main__":

    seeds_per_t_list = [1, 2, 3]
    n_t_for_eval_list = [1, 2]
    graph_name_list = ["Karate_temporal", "UIHC_HCP_patient_room_withinHCPxPx_2011_sampled"]

    # seeds_per_t_list = [1]
    # n_t_for_eval_list = [1]
    # graph_name_list = ["Karate_temporal"]
    P1_infile_name_list = [
            "P1_greedy_evalution_lazyF_expectedF.pickle",
            "P1_greedy_evalution_lazyT_expectedF.pickle",
            "P1_greedy_evalution_lazyT_expectedT.pickle"
            ]
    P1_name_list = [
            "P1_Greedy",
            "P1_LazyGreedy",
            "P1_E_LazyGreedy",
            ]
    P2_infile_name_list = [
            "P2_ISCK_greedy_evalution_lazyF_expectedF.pickle",
            "P2_ISCK_greedy_evalution_lazyT_expectedF.pickle",
            "P2_ISCK_greedy_evalution_lazyT_expectedT.pickle",
            "P2_ISCK_multiplicative_update_evalution_lazyF_expectedF.pickle",
            "P2_ISCK_multiplicative_update_evalution_lazyT_expectedF.pickle",
            "P2_ISCK_multiplicative_update_evalution_lazyT_expectedT.pickle"
            ]
    P2_name_list = [
            "P2_ISCK_greedy",
            "P2_LazyISCK_greedy",
            "P2_E_LazyISCK_greedy",
            "P2_ISCK_multiplicative_update",
            "P2_LazyISCK_multiplicative_update",
            "P2_E_LazyISCK_multiplicative_update"
            ]
    # P2_infile_name_list = [
            # "P2_ISCK_greedy_evalution_lazyF_expectedF.pickle",
            # "P2_ISCK_greedy_evalution_lazyT_expectedF.pickle",
            # "P2_ISCK_greedy_evalution_lazyT_expectedT.pickle",
            # "P2_ISCK_multiplicative_update_evalution_lazyF_expectedF.pickle",
            # "P2_ISCK_multiplicative_update_evalution_lazyT_expectedF.pickle",
            # "P2_ISCK_multiplicative_update_evalution_lazyT_expectedT.pickle"
            # ]
    # P2_name_list = [
            # "P2_ISCK_greedy",
            # "P2_LazyISCK_greedy",
            # "P2_E_LazyISCK_greedy",
            # "P2_ISCK_multiplicative_update",
            # "P2_LazyISCK_multiplicative_update",
            # "P2_E_LazyISCK_multiplicative_update"
            # ]
    P3_infile_name_list = [
            "P3_GR_gconstraintF_evalution_lazyF_expectedF.pickle",
            # "P3_GR_gconstraintF_evalution_lazyT_expectedF.pickle",
            # "P3_GR_gconstraintF_evalution_lazyT_expectedT.pickle",
            "P3_GR_gconstraintT_evalution_lazyF_expectedF.pickle",
            # "P3_GR_gconstraintT_evalution_lazyT_expectedF.pickle",
            # "P3_GR_gconstraintT_evalution_lazyT_expectedT.pickle"
            ]
    P3_name_list = [
            "P3_GreedyRatio",
            # "P3_LazyGreedyRatio",
            # "P3_E_LazyGreedyRatio",
            "P3_GreedyRatio_ghit>50",
            # "P3_LazyGreedyRatio_ghit>50",
            # "P3_E_LazyGreedyRatio_ghit>50"
            ]

    n_t_seeds = 2

    np.set_printoptions(suppress=True)

    # index_list = [
            # "GT", "Random(30rep)", 
            # "P1-Greedy", "P1-LazyGreedy", "P1-E_LazyGreedy",
            # "P2-ISCK-greedy", "P2-LazyISCK-greedy", "P2-E_LazyISCK-greedy",
            # "P2-ISCK-multiplicative_update", "P2-LazyISCK-multiplicative_update", "P2-E_LazyISCK-multiplicative_update",
            # "P3-GreedyRatio", "P3-LazyGreedyRatio", "P3-E_LazyGreedyRatio",
            # "P3-GreedyRatio-ghit>50", "P3-LazyGreedyRatio-ghit>50", "P3-E_LazyGreedyRatio-ghit>50"
            # ]

    # -------------------------------------------
    # These loops computes result tables.
    for seeds_per_t in seeds_per_t_list:
        for n_t_for_eval in n_t_for_eval_list:
            for graph_name in graph_name_list:

                path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

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

                print("Baselines ...")
                infile = "B_random_evalution_30rep.pickle"
                with open(path + infile, 'rb') as handle:
                    B_random_evaluation_rep_dict = pickle.load(handle)
                df_B_random = load_B_random_results(B_random_evaluation_rep_dict)
                list_of_df.append(df_B_random)
                list_of_algorithm.append("Random(30rep)")

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
                print(df_result)
                
                outfile = "result_concat.csv"
                df_result.to_csv(path + outfile, index=True)

    # -------------------------------------------
    # These loops saves ground truth observations
    print("Saving GT_observations.csv ...")
    for seeds_per_t in seeds_per_t_list:
        for n_t_for_eval in n_t_for_eval_list:
            for graph_name in graph_name_list:

                path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

                print("GT ...")
                infile = "GT_observation_evalution.pickle"
                with open(path + infile, 'rb') as handle:
                    GT_output_dict = pickle.load(handle)

                n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
                        node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
                        I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

                df_GT_observations = prepare_ground_truth_table(seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
                df_list_of_P = pd.DataFrame(data={"P_t":list_of_sets_of_P})

                outfile = "GT_observations.csv"
                df_GT_observations.to_csv(path + outfile, index=False)

                outfile = "list_of_P.csv"
                df_list_of_P.to_csv(path + outfile, index=True)
                
    # -------------------------------------------
    # These loops saves intermediary results for P1 - greedy
    for seeds_per_t in seeds_per_t_list:
        for n_t_for_eval in n_t_for_eval_list:
            for graph_name in graph_name_list:

                path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

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

                outfile = "P1_greedy_concat.csv"
                df_greedy_concat.to_csv(path + outfile, index=False)

    # -------------------------------------------
    # These loops saves intermediary results for P2 - ISCK
    print("Saving P2...")
    for seeds_per_t in seeds_per_t_list:
        for n_t_for_eval in n_t_for_eval_list:
            for graph_name in graph_name_list:

                path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

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

                outfile = "P2_ISCK_concat.csv"
                df_ISCK_concat.to_csv(path + outfile, index=False)

                outfile = "P2_ISCK_best_constraint.csv"
                df_ISCK_best_constraint.to_csv(path + outfile, index=False)

    # -------------------------------------------
    # These loops saves intermediary results for greedy ratio
    print("Saving P3_greedy_ratio_concat.csv and f_penalty_arrayx.csv ... in P3_details/")
    for seeds_per_t in seeds_per_t_list:
        for n_t_for_eval in n_t_for_eval_list:
            for graph_name in graph_name_list:

                path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, seeds_per_t, n_t_seeds, n_t_for_eval)

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

                        outfile = name + "f_penalty_array{}.csv".format(result_dict_idx)
                        df_intermediary_results.to_csv(path + "P3_details/" + outfile, index=False)

                df_greedy_ratio_concat = pd.concat(list_of_df)

                df_greedy_ratio_best_penalty = pd.concat(list_of_df_best_penalty)

                outfile = "P3_greedy_ratio_concat.csv"
                df_greedy_ratio_concat.to_csv(path + "P3_details/" + outfile, index=False)

                outfile = "P3_greedy_ratio_best_penalty.csv"
                df_greedy_ratio_best_penalty.to_csv(path + outfile, index=False)
