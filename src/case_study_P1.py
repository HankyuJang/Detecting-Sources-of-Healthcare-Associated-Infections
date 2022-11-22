"""
Author: -
Email: -
Last Modified: Feb 2022

Description: 

This script loads ground truth CDI observations 
and runs P1

To run it on UIHC sampled graph,
$ python case_study_P1.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled T -flag_lazy T -flag_expected_simulation T

"""

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
# import simulator_expected_load_sharing_temporal as expected_load_sharing
import simulator_truncated_expected_load_sharing_temporal as truncated_expected_load_sharing
from prep_GT_observation import *
from get_people_nodes import *
from approx_algorithms import *
from prep_result_dataframes import get_outfile_name_for_pickle

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
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    parser.add_argument('-flag_lazy', '--flag_lazy', type=bool, default=False,
                        help= '')
    parser.add_argument('-flag_expected_simulation', '--flag_expected_simulation', type=bool, default=False,
                        help= '')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    print("Load GT observations...\n")
    graph_name = get_graph_name(args)
    path = "../tables/case_study/{}/".format(graph_name)
    if args.dose_response == "exponential":
        infile = "GT_observation_evalution.pickle"
    elif args.dose_response == "linear":
        infile = "linear_GT_observation_evalution.pickle"
    with open(path + infile, 'rb') as handle:
        GT_output_dict = pickle.load(handle)

    n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N = unravel_GT_observaion_pickle(GT_output_dict)

    print("list_of_sets_of_P at T: {}".format(list_of_sets_of_P[T]))

    ####################################################################
    # Additional input for problem 1
    print("Additional input for problem 1")
    epsilon = k_total * 0.5 # k_total is the ground truth number of seeds
    cardinality_constraint_min = int(k_total - epsilon)
    cardinality_constraint_max = int(k_total + epsilon)
    cardinality_constraint_list = [k for k in range(cardinality_constraint_min, cardinality_constraint_max+1) if k>=1]
    print("Cardinality constraint: {}".format(cardinality_constraint_list))

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
        # if graph_name=="UIHC_HCP_patient_room_withinHCPxPx_2011":
            # truncate_probability = 0.00
        # else:
        truncate_probability = 0.05
        print("Truncate_prob: {}".format(truncate_probability))
        simul = truncated_expected_load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response, n_replicates=1, n_t_for_eval=args.n_t_for_eval, truncate_threshold=truncate_probability)
    else:
        simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
        simul.set_n_replicates(n_replicates)

    ####################################################################
    # Greedy source detection
    # P1_greedy
    L_P1_greedy_S_detected, L_P1_greedy_S_timesteps, \
    L_P1_greedy_seeds_array, L_P1_greedy_n_S, L_P1_greedy_n_S_correct, L_P1_greedy_loss_1, L_P1_greedy_loss_total, \
            L_P1_greedy_list_of_P_hit, L_P1_greedy_list_of_N_hit, \
            L_P1_greedy_TP, L_P1_greedy_TN, L_P1_greedy_FP, L_P1_greedy_FN, L_P1_greedy_F1, L_P1_greedy_MCC, L_P1_greedy_time_elapsed = initilize_n_empty_lists(16)

    for cardinality_constraint_idx, cardinality_constraint in enumerate(cardinality_constraint_list):

        start = timeit.default_timer()
        print("-"*20)
        print("Cardinality constraint: {}".format(cardinality_constraint))
        print("P1 Greedy")
        focus_obs1 = True
        P1_greedy_seeds_array, P1_greedy_n_S, P1_greedy_n_S_correct, P1_greedy_loss_1, P1_greedy_loss_total, \
            P1_greedy_list_of_P_hit, P1_greedy_list_of_N_hit, \
            P1_greedy_TP, P1_greedy_TN, P1_greedy_FP, P1_greedy_FN, P1_greedy_F1, P1_greedy_MCC = \
                run_greedy_source_detection_report_loss_per_timestep(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                    seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval, flag_lazy=args.flag_lazy)
        stop = timeit.default_timer()
        P1_greedy_time_elapsed = stop - start

        # Given the seedset, simply re-do the evaluation from the detected seeds
        if args.flag_expected_simulation:
            original_simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)
            original_simul.set_n_replicates(n_replicates)

            P1_greedy_seeds_array, P1_greedy_n_S, P1_greedy_n_S_correct, P1_greedy_loss_1, P1_greedy_loss_total, \
                P1_greedy_list_of_P_hit, P1_greedy_list_of_N_hit, \
                P1_greedy_TP, P1_greedy_TN, P1_greedy_FP, P1_greedy_FN, P1_greedy_F1, P1_greedy_MCC = \
                    evaluate_solution_seeds(original_simul, list_of_people_idx_arrays, seeds_array, P1_greedy_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, args.n_t_for_eval)

        L_P1_greedy_S_detected.append(str(list(P1_greedy_seeds_array.nonzero()[1])))
        L_P1_greedy_S_timesteps.append(str(list(P1_greedy_seeds_array.nonzero()[0])))

        L_P1_greedy_seeds_array.append(P1_greedy_seeds_array)
        L_P1_greedy_n_S.append(P1_greedy_n_S)
        L_P1_greedy_n_S_correct.append(P1_greedy_n_S_correct)
        L_P1_greedy_loss_1.append(P1_greedy_loss_1)
        L_P1_greedy_loss_total.append(P1_greedy_loss_total)
        L_P1_greedy_list_of_P_hit.append(P1_greedy_list_of_P_hit)
        L_P1_greedy_list_of_N_hit.append(P1_greedy_list_of_N_hit)
        L_P1_greedy_TP.append(P1_greedy_TP)
        L_P1_greedy_TN.append(P1_greedy_TN)
        L_P1_greedy_FP.append(P1_greedy_FP)
        L_P1_greedy_FN.append(P1_greedy_FN)
        L_P1_greedy_F1.append(P1_greedy_F1)
        L_P1_greedy_MCC.append(P1_greedy_MCC)
        L_P1_greedy_time_elapsed.append(P1_greedy_time_elapsed)

    df_greedy= pd.DataFrame({
        "k": cardinality_constraint_list,
        "S_detected": L_P1_greedy_S_detected,
        "S_timesteps": L_P1_greedy_S_timesteps,
        "n_S": L_P1_greedy_n_S,
        "n_S_correct": L_P1_greedy_n_S_correct,
        "TP": L_P1_greedy_TP,
        "TN": L_P1_greedy_TN,
        "FP": L_P1_greedy_FP,
        "FN": L_P1_greedy_FN,
        "F1": L_P1_greedy_F1,
        "MCC": L_P1_greedy_MCC,
        "Time(s)": L_P1_greedy_time_elapsed
        })
    print("\ngreedy results")
    print(df_greedy.round(2))

    # This returns the best MCC score over various solutions
    i_P1_greedy = np.argmax(np.array(L_P1_greedy_MCC))

    P1_greedy_evaluation_dict = dict()

    P1_greedy_evaluation_dict["k"] = cardinality_constraint_list[i_P1_greedy]
    P1_greedy_evaluation_dict["seeds_array"] = L_P1_greedy_seeds_array[i_P1_greedy]
    P1_greedy_evaluation_dict["n_S"] = L_P1_greedy_n_S[i_P1_greedy]
    P1_greedy_evaluation_dict["n_S_correct"] = L_P1_greedy_n_S_correct[i_P1_greedy]
    P1_greedy_evaluation_dict["loss_1"] = L_P1_greedy_loss_1[i_P1_greedy]
    P1_greedy_evaluation_dict["loss_total"] = L_P1_greedy_loss_total[i_P1_greedy]
    P1_greedy_evaluation_dict["list_of_P_hit"] = L_P1_greedy_list_of_P_hit[i_P1_greedy]
    P1_greedy_evaluation_dict["list_of_N_hit"] = L_P1_greedy_list_of_N_hit[i_P1_greedy]
    P1_greedy_evaluation_dict["TP"] = L_P1_greedy_TP[i_P1_greedy]
    P1_greedy_evaluation_dict["TN"] = L_P1_greedy_TN[i_P1_greedy]
    P1_greedy_evaluation_dict["FP"] = L_P1_greedy_FP[i_P1_greedy]
    P1_greedy_evaluation_dict["FN"] = L_P1_greedy_FN[i_P1_greedy]
    P1_greedy_evaluation_dict["F1"] = L_P1_greedy_F1[i_P1_greedy]
    P1_greedy_evaluation_dict["MCC"] = L_P1_greedy_MCC[i_P1_greedy]
    P1_greedy_evaluation_dict["time_elapsed"] = L_P1_greedy_time_elapsed[i_P1_greedy]
    #NOTE extra set of keys
    P1_greedy_evaluation_dict["df_greedy"] = df_greedy

    path = "../tables/case_study/{}/".format(graph_name)
    if args.dose_response == "exponential":
        outfile = get_outfile_name_for_pickle("P1_greedy", args)
    elif args.dose_response == "linear":
        outfile = get_outfile_name_for_pickle("linear_P1_greedy", args)
    print("Result pickle saved in {}".format(path + outfile))

    with open(path + outfile, "wb") as handle:
        pickle.dump(P1_greedy_evaluation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

