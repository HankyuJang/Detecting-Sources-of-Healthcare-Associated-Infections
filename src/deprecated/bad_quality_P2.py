"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads ground truth observations
and runs P2 - ISCK

NOTE: additional arguments: 
    - n_ISCK_iter
    - compute_pi

Usage

To run it on UIHC sampled graph,
$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1
$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1 -flag_lazy T
$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3 -flag_lazy T

$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1 -compute_pi greedy
$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3 -compute_pi greedy

$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1 -flag_lazy T -compute_pi greedy
$ python bad_quality_P2.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3 -flag_lazy T -compute_pi greedy


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

def prepare_ISCK_dataframes_per_iteration():
    # ISCK
    list_of_df_ISCK = []
    for i in range(len(L_ISCK_seeds_array)):
        list_of_df_ISCK.append(prepare_df_exp(L_ISCK_seeds_array[i], L_ISCK_n_S[i], L_ISCK_n_S_correct[i], \
                L_ISCK_TP[i], L_ISCK_TN[i], L_ISCK_FP[i], L_ISCK_FN[i], L_ISCK_F1[i], L_ISCK_MCC[i], ISCK_time_elapsed))

    return pd.concat(list_of_df_ISCK)

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
    parser.add_argument('-n_ISCK_iter', '--n_ISCK_iter', type=int, default=10,
                        help= 'Number of iterations for ISCK')
    parser.add_argument('-compute_pi', '--compute_pi', type=str, default="multiplicative_update",
                        help= 'greedy | multiplicative_update   -> method for computing pi')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    print("Load GT observations...\n")
    graph_name = get_graph_name(args)
    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    if args.dose_response == "exponential":
        infile = "GT_observation_evalution.pickle"
    elif args.dose_response == "linear":
        infile = "linear_GT_observation_evalution.pickle"
    with open(path + infile, 'rb') as handle:
        GT_output_dict = pickle.load(handle)

    n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

    ####################################################################
    # Additional input for problem 2
    print("Additional input for problem 2")
    N_T = len(list_of_sets_of_N[-1])
    print("Number of uninfected people nodes at T: {}".format(N_T))
    L_knapsack_constraint_array = []
    L_knapsack_constraint_array_last_n_t_for_eval = [] # keep this array for printing
    for i in range(0, 10, 2):
        array_of_knapsack_constraints_on_f = np.zeros((n_timesteps))
        array_of_knapsack_constraints_on_f[-1] = pow(2, -i) * N_T
        if args.n_t_for_eval == 2:
            array_of_knapsack_constraints_on_f[-2] = 0.5 * array_of_knapsack_constraints_on_f[-1]
        L_knapsack_constraint_array.append(array_of_knapsack_constraints_on_f)
        L_knapsack_constraint_array_last_n_t_for_eval.append(array_of_knapsack_constraints_on_f[-args.n_t_for_eval:])
        print("knapsack constraints at last n_t_for_eval: {}".format(array_of_knapsack_constraints_on_f[-args.n_t_for_eval:]))

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
    # P2_ISCK
    LoL_ISCK_S_detected, LoL_ISCK_S_timesteps, \
    LoL_ISCK_seeds_array, LoL_ISCK_n_S, LoL_ISCK_n_S_correct, LoL_ISCK_loss_1, LoL_ISCK_loss_total, \
            LoL_ISCK_list_of_P_hit, LoL_ISCK_list_of_N_hit, \
            LoL_ISCK_TP, LoL_ISCK_TN, LoL_ISCK_FP, LoL_ISCK_FN, LoL_ISCK_F1, LoL_ISCK_MCC, LoL_ISCK_time_elapsed = initilize_n_empty_lists(16)

    list_of_df = []
    list_of_W = []
    for knapsack_constraint_array_idx, array_of_knapsack_constraints_on_f in enumerate(L_knapsack_constraint_array):

        start = timeit.default_timer()
        print("-"*20)
        print("knapsack_constraint array: {}".format(L_knapsack_constraint_array_last_n_t_for_eval[knapsack_constraint_array_idx]))
        print("P2 ISCK - compute pi {}".format(args.compute_pi))
        # NOTE: L_ denotes list of 
        # NOTE: compute_pi in ["greedy", "multiplicative_update"]
        L_ISCK_seeds_array, L_ISCK_n_S, L_ISCK_n_S_correct, L_ISCK_loss_1, L_ISCK_loss_total, \
            L_ISCK_list_of_P_hit, L_ISCK_list_of_N_hit, \
            L_ISCK_TP, L_ISCK_TN, L_ISCK_FP, L_ISCK_FN, L_ISCK_F1, L_ISCK_MCC, W = \
                run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                    seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval, \
                    array_of_knapsack_constraints_on_f, flag_lazy=args.flag_lazy, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi=args.compute_pi)
        stop = timeit.default_timer()
        ISCK_time_elapsed = stop - start
        list_of_W.append(W)

        ##############################################################################
        # Given the seedset, simply re-do the evaluation from the detected seeds
        if args.flag_expected_simulation:
            original_simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
            original_simul.set_n_replicates(n_replicates)

            L_ISCK_n_S, L_ISCK_n_S_correct, L_ISCK_loss_1, L_ISCK_loss_total,\
                L_ISCK_list_of_P_hit, L_ISCK_list_of_N_hit, \
                L_ISCK_TP, L_ISCK_TN, L_ISCK_FP, L_ISCK_FN, L_ISCK_F1, L_ISCK_MCC = initilize_n_empty_lists(12)

            for ISCK_seeds_array in L_ISCK_seeds_array:
                _, ISCK_n_S, ISCK_n_S_correct, ISCK_loss_1, ISCK_loss_total, \
                    ISCK_list_of_P_hit, ISCK_list_of_N_hit, \
                    ISCK_TP, ISCK_TN, ISCK_FP, ISCK_FN, ISCK_F1, ISCK_MCC = \
                        evaluate_solution_seeds(original_simul, list_of_people_idx_arrays, seeds_array, ISCK_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval)

                L_ISCK_n_S.append(ISCK_n_S)
                L_ISCK_n_S_correct.append(ISCK_n_S_correct)
                L_ISCK_loss_1.append(ISCK_loss_1)
                L_ISCK_loss_total.append(ISCK_loss_total)
                L_ISCK_list_of_P_hit.append(ISCK_list_of_P_hit)
                L_ISCK_list_of_N_hit.append(ISCK_list_of_N_hit)
                L_ISCK_TP.append(ISCK_TP)
                L_ISCK_TN.append(ISCK_TN)
                L_ISCK_FP.append(ISCK_FP)
                L_ISCK_FN.append(ISCK_FN)
                L_ISCK_F1.append(ISCK_F1)
                L_ISCK_MCC.append(ISCK_MCC)
        ##############################################################################

        df_ISCK_results = prepare_ISCK_dataframes_per_iteration()
        n_iterations = df_ISCK_results.shape[0]

        df_ISCK_results.insert(loc=0, column="iteration", value=range(n_iterations))
        df_ISCK_results.insert(loc=0, column="knapsack_constraint", value=str(L_knapsack_constraint_array_last_n_t_for_eval[knapsack_constraint_array_idx]))

        list_of_df.append(df_ISCK_results)
        print("\nISCK results per iteration")
        print(df_ISCK_results.round(2))

        i_ISCK = np.nanargmax(np.array(L_ISCK_MCC))

        LoL_ISCK_seeds_array.append(L_ISCK_seeds_array[i_ISCK])
        LoL_ISCK_n_S.append(L_ISCK_n_S[i_ISCK])
        LoL_ISCK_n_S_correct.append(L_ISCK_n_S_correct[i_ISCK])
        LoL_ISCK_loss_1.append(L_ISCK_loss_1[i_ISCK])
        LoL_ISCK_loss_total.append(L_ISCK_loss_total[i_ISCK])
        LoL_ISCK_list_of_P_hit.append(L_ISCK_list_of_P_hit[i_ISCK])
        LoL_ISCK_list_of_N_hit.append(L_ISCK_list_of_N_hit[i_ISCK])
        LoL_ISCK_TP.append(L_ISCK_TP[i_ISCK])
        LoL_ISCK_TN.append(L_ISCK_TN[i_ISCK])
        LoL_ISCK_FP.append(L_ISCK_FP[i_ISCK])
        LoL_ISCK_FN.append(L_ISCK_FN[i_ISCK])
        LoL_ISCK_F1.append(L_ISCK_F1[i_ISCK])
        LoL_ISCK_MCC.append(L_ISCK_MCC[i_ISCK])
        LoL_ISCK_time_elapsed.append(ISCK_time_elapsed)

    df_ISCK_results_concat = pd.concat(list_of_df)

    i_best_ISCK = np.nanargmax(np.array(LoL_ISCK_MCC))

    P2_ISCK_evaluation_dict = dict()
    P2_ISCK_evaluation_dict["seeds_array"] = LoL_ISCK_seeds_array[i_best_ISCK]
    P2_ISCK_evaluation_dict["n_S"] = LoL_ISCK_n_S[i_best_ISCK]
    P2_ISCK_evaluation_dict["n_S_correct"] = LoL_ISCK_n_S_correct[i_best_ISCK]
    P2_ISCK_evaluation_dict["loss_1"] = LoL_ISCK_loss_1[i_best_ISCK]
    P2_ISCK_evaluation_dict["loss_total"] = LoL_ISCK_loss_total[i_best_ISCK]
    P2_ISCK_evaluation_dict["list_of_P_hit"] = LoL_ISCK_list_of_P_hit[i_best_ISCK]
    P2_ISCK_evaluation_dict["list_of_N_hit"] = LoL_ISCK_list_of_N_hit[i_best_ISCK]
    P2_ISCK_evaluation_dict["TP"] = LoL_ISCK_TP[i_best_ISCK]
    P2_ISCK_evaluation_dict["TN"] = LoL_ISCK_TN[i_best_ISCK]
    P2_ISCK_evaluation_dict["FP"] = LoL_ISCK_FP[i_best_ISCK]
    P2_ISCK_evaluation_dict["FN"] = LoL_ISCK_FN[i_best_ISCK]
    P2_ISCK_evaluation_dict["F1"] = LoL_ISCK_F1[i_best_ISCK]
    P2_ISCK_evaluation_dict["MCC"] = LoL_ISCK_MCC[i_best_ISCK]
    P2_ISCK_evaluation_dict["time_elapsed"] = LoL_ISCK_time_elapsed[i_best_ISCK]
    #NOTE: ISCK result over time and over knapsack constraints
    P2_ISCK_evaluation_dict["df_ISCK_results"] = df_ISCK_results_concat
    P2_ISCK_evaluation_dict["W"] = list_of_W

    path = "../tables/GT_bad/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    if args.dose_response == "exponential":
        outfile = get_outfile_name_for_P2_ISCK_pickle("P2_ISCK", args)
    elif args.dose_response == "linear":
        outfile = get_outfile_name_for_P2_ISCK_pickle("linear_P2_ISCK", args)
    print("Result pickle saved in {}".format(path + outfile))

    with open(path + outfile, "wb") as handle:
        pickle.dump(P2_ISCK_evaluation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # outfile_overtime = get_outfile_name_for_P2_ISCK_overtime("P2_ISCK", args)
    # df_ISCK_results.to_csv(path + outfile_overtime, index=False)
