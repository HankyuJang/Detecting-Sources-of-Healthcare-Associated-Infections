"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

    Lazy greedy vs Lazy ISCK vs GreedyRatio

Usage

#-------------------------------------
To run it on Karate graph,
$ python exp_p1_p2_p3.py -seeds_per_t 1
$ python exp_p1_p2_p3.py -seeds_per_t 3

To run it on UIHC sampled graph,
$ python exp_p1_p2_p3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1
$ python exp_p1_p2_p3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3

#-------------------------------------
Experiment2 - GT quality any
To run it on Karate graph,
$ python exp_p1_p2_p3.py -seeds_per_t 1 -GT_quality any
$ python exp_p1_p2_p3.py -seeds_per_t 3 -GT_quality any

To run it on UIHC sampled graph,
$ python exp_p1_p2_p3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1 -GT_quality any
$ python exp_p1_p2_p3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3 -GT_quality any

#-------------------------------------
Experiment3 - Dose response = linear
To run it on Karate graph,
$ python exp_p1_p2_p3.py -seeds_per_t 1 -dose_response linear
$ python exp_p1_p2_p3.py -seeds_per_t 3 -dose_response linear

To run it on UIHC sampled graph,
$ python exp_p1_p2_p3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 1 -dose_response linear
$ python exp_p1_p2_p3.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -seeds_per_t 3 -dose_response linear


"""

from utils.load_network import *
from utils.set_parameters import *
import simulator_load_sharing_temporal_v2 as load_sharing
import simulator_expected_load_sharing_temporal as expected_load_sharing
from approx_algorithms import *
from prep_GT_observation import *
from get_people_nodes import *
from prep_result_dataframes import *

import argparse
import pandas as pd
import random as random
import timeit

# NOTE: setting the penalty array to all ones does not do anything.
def prep_array_of_knapsack_constraints_on_f(args, compute_pi, n_timesteps, list_of_sets_of_N):
    if compute_pi == "greedy":
        if args.name=="Karate_temporal":
            i = -2
        elif args.name=="UIHC_HCP_patient_room_withinHCPxPx":
            # i = -6
            i = -4 # Loosen this
    elif compute_pi == "multiplicative_update":
        if args.name=="Karate_temporal":
            i = -2
        elif args.name=="UIHC_HCP_patient_room_withinHCPxPx":
            i = -4
    N_T = len(list_of_sets_of_N[-1])
    print("Number of uninfected people nodes at T: {}".format(N_T))
    array_of_knapsack_constraints_on_f = np.zeros((n_timesteps))
    array_of_knapsack_constraints_on_f[-1] = pow(2, i) * N_T
    if args.n_t_for_eval == 2:
        array_of_knapsack_constraints_on_f[-2] = 0.5 * array_of_knapsack_constraints_on_f[-1]
    print("P2. Knapsack constraints: {}".format(array_of_knapsack_constraints_on_f))
    return array_of_knapsack_constraints_on_f

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
                        help= 'Quality of the ground truth simulation. best | median | any')
    parser.add_argument('-seeds_per_t', '--seeds_per_t', type=int, default=1,
                        help= 'number of seeds per timestep')
    parser.add_argument('-n_t_seeds', '--n_t_seeds', type=int, default=2,
                        help= 'number of timesteps for seeds')
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    parser.add_argument('-n_ISCK_iter', '--n_ISCK_iter', type=int, default=10,
                        help= 'Number of iterations for ISCK')
    args = parser.parse_args()

    np.set_printoptions(suppress=True)

    ####################################################################
    n_experiment_repeat = 30

    ####################################################################
    # Parameters for the simulation. These are same regardless of the graph
    n_timesteps = 31
    n_replicates = 100 
    area_people = 2000 # area of patient. 2000cm^2
    area_location = 40000 # area of room. 40000cm^2
    ####################################################################
    # Parameters for experiments
    # NOTE: treat T as a global variable. Used anywhere in this script.
    T = n_timesteps-1 # T is the index of the last timestep
    n_t_for_eval = args.n_t_for_eval # Use the latest n timesteps for evaluation. e.g., T and T-1

    flag_increase_area = True # If this is set to True, then increase area of each node based on their max degree over grpahs

    ####################################################################
    # Ground truth seeds over time
    number_of_seeds_over_time = np.zeros((n_timesteps)).astype(int)
    for t in range(args.n_t_seeds):
        number_of_seeds_over_time[t] = args.seeds_per_t

    k_total = np.sum(number_of_seeds_over_time)
    print("Set number of seeds at various timesteps\ntime 0: 1 seed\ntime 1: 1 seed")
    print("number_of_seeds_over_time: {}\n".format(number_of_seeds_over_time))

    ####################################################################
    # Additional input for problem 1
    cardinality_constraint = k_total
    ####################################################################
    # Additional input for problem 3
    print("Additional input for problem 3")
    if args.name=="Karate_temporal":
        i = -5
    elif args.name=="UIHC_HCP_patient_room_withinHCPxPx":
        i = -5
    array_of_penalty_on_f = np.zeros((n_timesteps))
    array_of_penalty_on_f[-1] = pow(2, i)
    if args.n_t_for_eval == 2:
        array_of_penalty_on_f[-2] = pow(2, i)*2
    print("P3. Penalty on f: {}".format(array_of_penalty_on_f))

    ####################################################################
    print("Load network...\n")
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, graph_name = process_data_for_experiments(args, area_people, area_location, flag_increase_area)

    # NOTE: Make sure all the graphs in different time snapshots have same set of nodes
    node_name_to_idx_mapping = dict([(node_name, node_idx) for node_idx, node_name in enumerate(G_over_time[0].nodes())])
    node_idx_to_name_mapping = dict([(node_idx, node_name) for node_idx, node_name in enumerate(G_over_time[0].nodes())])

    # Get a list of people index arrays. Array at each idx correspond to the indicies of people at that day
    list_of_people_idx_arrays = get_people_idx_array_over_time(G_over_time, node_name_to_idx_mapping, people_nodes_idx)
    # List of sets of people indicies
    list_of_sets_of_V = [set(arr) for arr in list_of_people_idx_arrays]

    ####################################################################
    # 0. Create simulation instance with empty seeds list
    rho, d, q, pi, contact_area = set_simulation_parameters(args, k_total)
    print("rho: {}".format(rho))
    print("d: {}".format(d))
    print("q: {}".format(q))
    print("pi: {}".format(pi))
    print("contact_area: {}".format(contact_area))
    simul = load_sharing.Simulation(G_over_time, [], people_nodes, area_array, contact_area, n_timesteps, rho, d, q, pi, args.dose_response)

    ####################################################################
    # NOTE: For all experiments, run it for n_replicates per seed set
    simul.set_n_replicates(n_replicates)
    ####################################################################

    # F1
    GT_F1_array = np.zeros((n_experiment_repeat))
    BR_F1_array = np.zeros((n_experiment_repeat))
    lazy_G1_F1_array = np.zeros((n_experiment_repeat))
    lazy_ISCK_F1_array = np.zeros((n_experiment_repeat))
    lazy_ISCK_MU_F1_array = np.zeros((n_experiment_repeat))
    GR_F1_array = np.zeros((n_experiment_repeat))
    GR_ghit50_F1_array = np.zeros((n_experiment_repeat))
    # MCC
    GT_MCC_array = np.zeros((n_experiment_repeat))
    BR_MCC_array = np.zeros((n_experiment_repeat))
    lazy_G1_MCC_array = np.zeros((n_experiment_repeat))
    lazy_ISCK_MCC_array = np.zeros((n_experiment_repeat))
    lazy_ISCK_MU_MCC_array = np.zeros((n_experiment_repeat))
    GR_MCC_array = np.zeros((n_experiment_repeat))
    GR_ghit50_MCC_array = np.zeros((n_experiment_repeat))
    # TP
    GT_TP_array = np.zeros((n_experiment_repeat))
    BR_TP_array = np.zeros((n_experiment_repeat))
    lazy_G1_TP_array = np.zeros((n_experiment_repeat))
    lazy_ISCK_TP_array = np.zeros((n_experiment_repeat))
    lazy_ISCK_MU_TP_array = np.zeros((n_experiment_repeat))
    GR_TP_array = np.zeros((n_experiment_repeat))
    GR_ghit50_TP_array = np.zeros((n_experiment_repeat))

    for idx_experiment_repeat in range(n_experiment_repeat):
        try:

            print("{}/{}...".format(1+idx_experiment_repeat, n_experiment_repeat))
            ####################################################################
            # Set random seed, and observe infections
            # 1. Data generation
            if args.GT_quality in ["best", "median"]:
                print("Generate seed set w/ the best quality. Get ground truth observations...")
                seeds_array, obs_state, I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N \
                        = prepare_GT_data(args, simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, args.GT_quality)
            elif args.GT_quality == "any":
                print("Generate seed set. Do not care about the quality. Get ground truth observations...")
                seeds_array, obs_state, I1, list_of_sets_of_P, list_of_sets_of_N \
                        = prepare_GT_data_quality_x(args, simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval)

                # NOTE: For all experiments, run it for n_replicates per seed set
                simul.set_n_replicates(n_replicates)


            ####################################################################
            # 2. Compute ground truth loss per timestep
            # We're not interested in loss over timestep (e.g. missing infection) in this project, so just take the loss at the last timestep.
            start = timeit.default_timer()
            print("Compute GT losses")
            GT_loss_1, GT_loss_total, \
                GT_list_of_P_hit, GT_list_of_N_hit, \
                GT_TP, GT_TN, GT_FP, GT_FN, GT_F1, GT_MCC = \
                compute_GT_loss_per_timestep(simul, list_of_people_idx_arrays, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
            stop = timeit.default_timer()
            GT_time_elapsed = stop - start

            GT_F1_array[idx_experiment_repeat] = GT_F1
            GT_MCC_array[idx_experiment_repeat] = GT_MCC
            GT_TP_array[idx_experiment_repeat] = GT_TP

            print("GT_TP: {:.2f}, GT_F1: {:.2f}, GT_MCC: {:.2f}".format(GT_TP, GT_F1, GT_MCC))

            ####################################################################
            # Baselines
            # Randomly selected seed out of people nodes
            start = timeit.default_timer()
            print("-"*20)
            print("Compute random baseline")
            BR_seeds_array, BR_n_S, BR_n_S_correct, BR_loss_1, BR_loss_total, \
                BR_list_of_P_hit, BR_list_of_N_hit, \
                BR_TP, BR_TN, BR_FP, BR_FN, BR_F1, BR_MCC = \
                    run_BR_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                                                    seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
            stop = timeit.default_timer()
            BR_time_elapsed = stop - start

            BR_F1_array[idx_experiment_repeat] = BR_F1
            BR_MCC_array[idx_experiment_repeat] = BR_MCC
            BR_TP_array[idx_experiment_repeat] = BR_TP

            ####################################################################
            # Greedy ratio
            start = timeit.default_timer()
            print("-"*20)
            print("P3 Greedy Ratio")
            # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
            _, [GR_seeds_array, GR_n_S, GR_n_S_correct, GR_loss_1, GR_loss_total, \
                GR_list_of_P_hit, GR_list_of_N_hit, \
                GR_TP, GR_TN, GR_FP, GR_FN, GR_F1, GR_MCC] = \
                    run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                        seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=False,
                        flag_g_constraint=False)

            stop = timeit.default_timer()
            GR_time_elapsed = stop - start

            GR_F1_array[idx_experiment_repeat] = GR_F1
            GR_MCC_array[idx_experiment_repeat] = GR_MCC
            GR_TP_array[idx_experiment_repeat] = GR_TP

            ####################################################################
            # Greedy ratio
            start = timeit.default_timer()
            print("-"*20)
            print("P3 Greedy Ratio g hit > 50 %")
            # NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
            _, [GR_ghit50_seeds_array, GR_ghit50_n_S, GR_ghit50_n_S_correct, GR_ghit50_loss_1, GR_ghit50_loss_total, \
                GR_ghit50_list_of_P_hit, GR_ghit50_list_of_N_hit, \
                GR_ghit50_TP, GR_ghit50_TN, GR_ghit50_FP, GR_ghit50_FN, GR_ghit50_F1, GR_ghit50_MCC] = \
                    run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                        seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=False,
                        flag_g_constraint=True)

            stop = timeit.default_timer()
            GR_ghit50_time_elapsed = stop - start

            GR_ghit50_F1_array[idx_experiment_repeat] = GR_ghit50_F1
            GR_ghit50_MCC_array[idx_experiment_repeat] = GR_ghit50_MCC
            GR_ghit50_TP_array[idx_experiment_repeat] = GR_ghit50_TP

            ####################################################################
            # Lazy implementations
            ####################################################################
            # Greedy source detection
            start = timeit.default_timer()
            print("-"*20)
            print("P1 Lazy Greedy")
            focus_obs1 = True
            lazy_G1_seeds_array, lazy_G1_n_S, lazy_G1_n_S_correct, lazy_G1_loss_1, lazy_G1_loss_total, \
                lazy_G1_list_of_P_hit, lazy_G1_list_of_N_hit, \
                lazy_G1_TP, lazy_G1_TN, lazy_G1_FP, lazy_G1_FN, lazy_G1_F1, lazy_G1_MCC = \
                    run_greedy_source_detection_report_loss_per_timestep(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                        seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=True)
            stop = timeit.default_timer()
            lazy_G1_time_elapsed = stop - start

            lazy_G1_F1_array[idx_experiment_repeat] = lazy_G1_F1
            lazy_G1_MCC_array[idx_experiment_repeat] = lazy_G1_MCC
            lazy_G1_TP_array[idx_experiment_repeat] = lazy_G1_TP

            ####################################################################
            # ISCK
            ####################################################################
            # Additional input for problem 2
            array_of_knapsack_constraints_on_f = prep_array_of_knapsack_constraints_on_f(args, "greedy", n_timesteps, list_of_sets_of_N)

            start = timeit.default_timer()
            print("-"*20)
            print("P2 Lazy ISCK")
            L_lazy_ISCK_seeds_array, L_lazy_ISCK_n_S, L_lazy_ISCK_n_S_correct, L_lazy_ISCK_loss_1, L_lazy_ISCK_loss_total, \
                L_lazy_ISCK_list_of_P_hit, L_lazy_ISCK_list_of_N_hit, \
                L_lazy_ISCK_TP, L_lazy_ISCK_TN, L_lazy_ISCK_FP, L_lazy_ISCK_FN, L_lazy_ISCK_F1, L_lazy_ISCK_MCC, W = \
                    run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                        seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                        array_of_knapsack_constraints_on_f, flag_lazy=True, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="greedy")
            stop = timeit.default_timer()
            lazy_ISCK_time_elapsed = stop - start

            i_ISCK = np.argmax(np.array(L_lazy_ISCK_MCC))

            lazy_ISCK_F1_array[idx_experiment_repeat] = L_lazy_ISCK_F1[i_ISCK]
            lazy_ISCK_MCC_array[idx_experiment_repeat] = L_lazy_ISCK_MCC[i_ISCK]
            lazy_ISCK_TP_array[idx_experiment_repeat] = L_lazy_ISCK_TP[i_ISCK]

            # ISCK MU
            # NOTE: setting the penalty array to all ones does not do anything.
            array_of_knapsack_constraints_on_f = prep_array_of_knapsack_constraints_on_f(args, "multiplicative_update", n_timesteps, list_of_sets_of_N)

            start = timeit.default_timer()
            print("-"*20)
            print("P2 Lazy ISCK MU")
            L_lazy_ISCK_MU_seeds_array, L_lazy_ISCK_MU_n_S, L_lazy_ISCK_MU_n_S_correct, L_lazy_ISCK_MU_loss_1, L_lazy_ISCK_MU_loss_total, \
                L_lazy_ISCK_MU_list_of_P_hit, L_lazy_ISCK_MU_list_of_N_hit, \
                L_lazy_ISCK_MU_TP, L_lazy_ISCK_MU_TN, L_lazy_ISCK_MU_FP, L_lazy_ISCK_MU_FN, L_lazy_ISCK_MU_F1, L_lazy_ISCK_MU_MCC, W = \
                    run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                        seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                        array_of_knapsack_constraints_on_f, flag_lazy=True, flag_knapsack_in_pi=True, n_ISCK_iter=args.n_ISCK_iter, compute_pi="multiplicative_update")
            stop = timeit.default_timer()
            lazy_ISCK_MU_time_elapsed = stop - start

            i_ISCK_MU = np.argmax(np.array(L_lazy_ISCK_MU_MCC))

            lazy_ISCK_MU_F1_array[idx_experiment_repeat] = L_lazy_ISCK_MU_F1[i_ISCK_MU]
            lazy_ISCK_MU_MCC_array[idx_experiment_repeat] = L_lazy_ISCK_MU_MCC[i_ISCK_MU]
            lazy_ISCK_MU_TP_array[idx_experiment_repeat] = L_lazy_ISCK_MU_TP[i_ISCK_MU]
            
        except Exception as e:
            print("\nError in iteration {}".format(idx_experiment_repeat))
            print("{}\n".format(e))

    ####################################################################
    df_result = pd.DataFrame(data={
        "GT_F1": GT_F1_array,
        "BR_F1": BR_F1_array,
        "lazy_G1_F1": lazy_G1_F1_array,
        "lazy_ISCK_F1": lazy_ISCK_F1_array,
        "lazy_ISCK_MU_F1": lazy_ISCK_MU_F1_array,
        "GR_F1": GR_F1_array,
        "GR_ghit50_F1": GR_ghit50_F1_array,
        "GT_MCC": GT_MCC_array,
        "BR_MCC": BR_MCC_array,
        "lazy_G1_MCC": lazy_G1_MCC_array,
        "lazy_ISCK_MCC": lazy_ISCK_MCC_array,
        "lazy_ISCK_MU_MCC": lazy_ISCK_MU_MCC_array,
        "GR_MCC": GR_MCC_array,
        "GR_ghit50_MCC": GR_ghit50_MCC_array,
        "GT_TP": GT_TP_array,
        "BR_TP": BR_TP_array,
        "lazy_G1_TP": lazy_G1_TP_array,
        "lazy_ISCK_TP": lazy_ISCK_TP_array,
        "lazy_ISCK_MU_TP": lazy_ISCK_MU_TP_array,
        "GR_TP": GR_TP_array,
        "GR_ghit50_TP": GR_ghit50_TP_array})
    
    print("Results")
    print(df_result.mean())
    
    path = "../tables/exp_p1_p2_p3/"
    if args.dose_response == "exponential":
        if args.GT_quality in ["best", "median"]:
            outfile = "{}_k{}.csv".format(graph_name, k_total)
        elif args.GT_quality == "any":
            outfile = "GT_quality_x_{}_k{}.csv".format(graph_name, k_total)
    elif args.dose_response == "linear":
        if args.GT_quality in ["best", "median"]:
            outfile = "linear_{}_k{}.csv".format(graph_name, k_total)
        elif args.GT_quality == "any":
            outfile = "linear_GT_quality_x_{}_k{}.csv".format(graph_name, k_total)

    df_result.to_csv(path + outfile, index=False)
