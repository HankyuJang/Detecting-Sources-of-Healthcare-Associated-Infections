"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads ground truth observations
and runs P3 - GreedyRatio

NOTE: additional arguments: 
    - flag_g_constraint

Usage

To run it on UIHC sampled graph,
$ python bad_quality_P3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1
$ python bad_quality_P3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

$ python bad_quality_P3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1 -flag_g_constraint T
$ python bad_quality_P3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3 -flag_g_constraint T

"""

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
# import simulator_expected_load_sharing_temporal as expected_load_sharing
import simulator_truncated_expected_load_sharing_temporal as truncated_expected_load_sharing
from prep_GT_observation import *
from get_people_nodes import *
from approx_algorithms import *
from prep_result_dataframes import *

import argparse
import pandas as pd
import random as random
import timeit
import pickle

def initilize_n_empty_lists(n):
    list_to_return = []
    for i in range(n):
        list_to_return.append([])
    return list_to_return

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
    parser.add_argument('-GT_quality', '--GT_quality', type=str, default="best",
                        help= 'Quality of the ground truth simulation. best | median. Always use best')
    parser.add_argument('-seeds_per_t', '--seeds_per_t', type=int, default=1,
                        help= 'number of seeds per timestep')
    parser.add_argument('-n_t_seeds', '--n_t_seeds', type=int, default=2,
                        help= 'number of timesteps for seeds')
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    parser.add_argument('-flag_lazy', '--flag_lazy', type=bool, default=False,
                        help= '')
    parser.add_argument('-flag_expected_simulation', '--flag_expected_simulation', type=bool, default=False,
                        help= '')
    parser.add_argument('-flag_g_constraint', '--flag_g_constraint', type=bool, default=False,
                        help= '')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    print("Load GT observations...\n")
    graph_name = get_graph_name(args)
    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    if args.dose_response == "exponential":
        infile = "GT_observation_evalution.pickle"
    if args.dose_response == "linear":
        infile = "linear_GT_observation_evalution.pickle"
    with open(path + infile, 'rb') as handle:
        GT_output_dict = pickle.load(handle)

    n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

    ####################################################################
    # Additional input for problem 3
    print("Additional input for problem 3")
    L_penalty_array = []
    L_penalty_array_last_n_t_for_eval = [] # keep this array for printing
    for i in range(-10, 1, 5): # [-10, -5, 0]
        array_of_penalty_on_f = np.zeros((n_timesteps))
        array_of_penalty_on_f[-1] = pow(2, i)
        if args.n_t_for_eval == 2:
            array_of_penalty_on_f[-2] = pow(2, i)*2
        L_penalty_array.append(array_of_penalty_on_f)
        L_penalty_array_last_n_t_for_eval.append(array_of_penalty_on_f[-args.n_t_for_eval:])

    ####################################################################
    print("Load network...\n")
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, _ = process_data_for_experiments(args, area_people, area_location, flag_increase_area)

    ####################################################################
    # 0. Create simulation instance with empty seeds list
    rho, d, q, pi, contact_area = set_simulation_parameters(args, k_total)
    print("rho: {}".format(rho))
    print("d: {}".format(d))
    print("q: {}".format(q))
    print("pi: {}".format(pi))
    print("contact_area: {}".format(contact_area))

    if args.flag_expected_simulation:
        # simul = expected_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
        truncate_probability = 0.05
        simul = truncated_expected_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response, n_replicates=1, n_t_for_eval=args.n_t_for_eval, truncate_threshold=truncate_probability)
    else:
        simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
        simul.set_n_replicates(n_replicates)

    ####################################################################
    # P3_Greedy_Ratio
    L_GR_S_detected, L_GR_S_timesteps, \
    L_GR_seeds_array, L_GR_n_S, L_GR_n_S_correct, L_GR_loss_1, L_GR_loss_total, \
            L_GR_list_of_P_hit, L_GR_list_of_N_hit, \
            L_GR_TP, L_GR_TN, L_GR_FP, L_GR_FN, L_GR_F1, L_GR_MCC, L_GR_time_elapsed = initilize_n_empty_lists(16)

    dict_of_intermediary_results_dict = dict()
    for penalty_array_idx, array_of_penalty_on_f in enumerate(L_penalty_array):
        start = timeit.default_timer()
        print("-"*20)
        print("Penalty array: {}".format(array_of_penalty_on_f))
        print("P3 Greedy Ratio")
        # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
        intermediary_results_dict, [GR_seeds_array, GR_n_S, GR_n_S_correct, GR_loss_1, GR_loss_total, \
            GR_list_of_P_hit, GR_list_of_N_hit, \
            GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC] = \
                run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                    seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval, array_of_penalty_on_f, flag_lazy=args.flag_lazy,
                    flag_g_constraint=args.flag_g_constraint)

        stop = timeit.default_timer()
        GR_time_elapsed = stop - start

        # Given the seedset, simply re-do the evaluation from the detected seeds
        if args.flag_expected_simulation:
            original_simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
            original_simul.set_n_replicates(n_replicates)

            GR_seeds_array, GR_n_S, GR_n_S_correct, GR_loss_1, GR_loss_total, \
                GR_list_of_P_hit, GR_list_of_N_hit, \
                GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC = \
                    evaluate_solution_seeds(original_simul, list_of_people_idx_arrays, seeds_array, GR_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval)

        dict_of_intermediary_results_dict[penalty_array_idx] = intermediary_results_dict

        L_GR_S_detected.append(str(list(GR_seeds_array.nonzero()[1])))
        L_GR_S_timesteps.append(str(list(GR_seeds_array.nonzero()[0])))

        L_GR_seeds_array.append(GR_seeds_array)
        L_GR_n_S.append(GR_n_S)
        L_GR_n_S_correct.append(GR_n_S_correct)
        L_GR_loss_1.append(GR_loss_1)
        L_GR_loss_total.append(GR_loss_total)
        L_GR_list_of_P_hit.append(GR_list_of_P_hit)
        L_GR_list_of_N_hit.append(GR_list_of_N_hit)
        L_GR_TP.append(GR_TP)
        L_GR_TN.append(GR_TN)
        L_GR_FP.append(GR_FP)
        L_GR_FN.append(GR_FN)
        L_GR_F1.append(GR_F1)
        L_GR_MCC.append(GR_MCC)
        L_GR_time_elapsed.append(GR_time_elapsed)

    df_greedy_ratio = pd.DataFrame({
        "L_penalty_array": L_penalty_array_last_n_t_for_eval,
        "S_detected": L_GR_S_detected,
        "S_timesteps": L_GR_S_timesteps,
        "n_S": L_GR_n_S,
        "n_S_correct": L_GR_n_S_correct,
        "TP": L_GR_TP,
        "TN": L_GR_TN,
        "FP": L_GR_FP,
        "FN": L_GR_FN,
        "F1": L_GR_F1,
        "MCC": L_GR_MCC,
        "Time(s)": L_GR_time_elapsed
        })
    print("\nGR results per iteration")
    print(df_greedy_ratio.round(2))

    # This returns the best MCC score over various penalty arrays
    i_GR = np.nanargmax(np.array(L_GR_MCC))

    P3_greedy_ratio_evaluation_dict = dict()
    P3_greedy_ratio_evaluation_dict["array_of_penalty_on_f"] = L_penalty_array_last_n_t_for_eval[i_GR]
    P3_greedy_ratio_evaluation_dict["seeds_array"] = L_GR_seeds_array[i_GR]
    P3_greedy_ratio_evaluation_dict["n_S"] = L_GR_n_S[i_GR]
    P3_greedy_ratio_evaluation_dict["n_S_correct"] = L_GR_n_S_correct[i_GR]
    P3_greedy_ratio_evaluation_dict["loss_1"] = L_GR_loss_1[i_GR]
    P3_greedy_ratio_evaluation_dict["loss_total"] = L_GR_loss_total[i_GR]
    P3_greedy_ratio_evaluation_dict["list_of_P_hit"] = L_GR_list_of_P_hit[i_GR]
    P3_greedy_ratio_evaluation_dict["list_of_N_hit"] = L_GR_list_of_N_hit[i_GR]
    P3_greedy_ratio_evaluation_dict["TP"] = L_GR_TP[i_GR]
    P3_greedy_ratio_evaluation_dict["TN"] = L_GR_TN[i_GR]
    P3_greedy_ratio_evaluation_dict["FP"] = L_GR_FP[i_GR]
    P3_greedy_ratio_evaluation_dict["FN"] = L_GR_FN[i_GR]
    P3_greedy_ratio_evaluation_dict["F1"] = L_GR_F1[i_GR]
    P3_greedy_ratio_evaluation_dict["MCC"] = L_GR_MCC[i_GR]
    P3_greedy_ratio_evaluation_dict["time_elapsed"] = L_GR_time_elapsed[i_GR]
    #NOTE extra set of keys
    P3_greedy_ratio_evaluation_dict["dict_of_intermediary_results_dict"] = dict_of_intermediary_results_dict
    P3_greedy_ratio_evaluation_dict["df_greedy_ratio"] = df_greedy_ratio

    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    if args.dose_response == "exponential":
        outfile = get_outfile_name_for_P3_GR_pickle("P3_GR", args)
    elif args.dose_response == "linear":
        outfile = get_outfile_name_for_P3_GR_pickle("linear_P3_GR", args)
    print("Result pickle saved in {}".format(path + outfile))

    with open(path + outfile, "wb") as handle:
        pickle.dump(P3_greedy_ratio_evaluation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # outfile_overtime = get_outfile_name_for_P3_GR_penalty_arrays("P3_GR", args)
    # df_greedy_ratio.to_csv(path + outfile_overtime, index=False)

