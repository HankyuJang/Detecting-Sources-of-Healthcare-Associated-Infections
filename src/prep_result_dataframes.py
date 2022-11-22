import pandas as pd
import numpy as np

def get_outfile_name_for_pickle(algorithm_name, args):
    if args.flag_lazy:
        lazy = 'T'
    else:
        lazy = 'F'

    if args.flag_expected_simulation:
        expected_simulation = 'T'
    else:
        expected_simulation = 'F'

    outfile = "{}_evalution_lazy{}_expected{}.pickle".format(algorithm_name, lazy, expected_simulation)
    return outfile

def get_outfile_name_for_P2_ISCK_pickle(algorithm_name, args):
    if args.flag_lazy:
        lazy = 'T'
    else:
        lazy = 'F'

    if args.flag_expected_simulation:
        expected_simulation = 'T'
    else:
        expected_simulation = 'F'

    outfile = "{}_{}_evalution_lazy{}_expected{}.pickle".format(algorithm_name, args.compute_pi, lazy, expected_simulation)
    return outfile

def get_outfile_name_for_P2_ISCK_overtime(algorithm_name, args):
    if args.flag_lazy:
        lazy = 'T'
    else:
        lazy = 'F'

    if args.flag_expected_simulation:
        expected_simulation = 'T'
    else:
        expected_simulation = 'F'

    outfile = "{}_{}_evalution_lazy{}_expected{}_overtime.csv".format(algorithm_name, args.compute_pi, lazy, expected_simulation)
    return outfile

def get_outfile_name_for_P3_GR_pickle(algorithm_name, args):
    if args.flag_lazy:
        lazy = 'T'
    else:
        lazy = 'F'

    if args.flag_expected_simulation:
        expected_simulation = 'T'
    else:
        expected_simulation = 'F'

    if args.flag_g_constraint:
        g_constraint = 'T'
    else:
        g_constraint = 'F'

    outfile = "{}_gconstraint{}_evalution_lazy{}_expected{}.pickle".format(algorithm_name, g_constraint, lazy, expected_simulation)
    return outfile

def get_outfile_name_for_P3_GR_penalty_arrays(algorithm_name, args):
    if args.flag_lazy:
        lazy = 'T'
    else:
        lazy = 'F'

    if args.flag_expected_simulation:
        expected_simulation = 'T'
    else:
        expected_simulation = 'F'

    if args.flag_g_constraint:
        g_constraint = 'T'
    else:
        g_constraint = 'F'

    outfile = "{}_gconstraint{}_evalution_lazy{}_expected{}_penalty_arrays.csv".format(algorithm_name, g_constraint, lazy, expected_simulation)
    return outfile

def prepare_ground_truth_table(GT_seeds_array, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval):
    S = list(GT_seeds_array.nonzero()[1])
    S_timestep = list(GT_seeds_array.nonzero()[0])
    len_P_t_over_time = [len(P_t) for P_t in list_of_sets_of_P[-n_t_for_eval:]]
    len_N_t_over_time = [len(N_t) for N_t in list_of_sets_of_N[-n_t_for_eval:]]
    df_ground_truth = pd.DataFrame(
            {"Seed_idx": [str(S)],
                "Seed_timesteps": [str(S_timestep)],
                "|P_t|last{}ts".format(n_t_for_eval): str(len_P_t_over_time),
                "|N_t|last{}ts".format(n_t_for_eval): str(len_N_t_over_time)
                })
    return df_ground_truth

def prepare_df_exp(detected_seeds_array, n_S, n_S_correct, \
                    TP, TN, FP, FN, F1, MCC, time_elapsed):
    # if detected_seeds_array == None:
        # S_detected = ['-']
        # S_timesteps = ['-']
    # else:
    S_detected = [str(list(detected_seeds_array.nonzero()[1]))]
    S_timesteps = [str(list(detected_seeds_array.nonzero()[0]))]

    df_exp = pd.DataFrame({
                "S_detected": S_detected,
                "S_timesteps": S_timesteps,
                "n_S": [n_S],
                "n_S_correct": [n_S_correct],
                "TP": [TP],
                "TN": [TN],
                "FP": [FP],
                "FN": [FN],
                "F1": [F1],
                "MCC": [MCC],
                # "Time": ["{:.3f} s".format(time_elapsed)],
                "Time(s)": [time_elapsed],
            })
    return df_exp

def concat_result_dataframes(index_list, list_of_df):
    df_result = pd.concat(list_of_df)
    df_result["Algorithm"] = index_list
    df_result.set_index("Algorithm", inplace=True)
    return df_result

def print_result_dataframes(df_ground_truth_observations, df_result):
    print("\nGround Truth observations. Seeds and number of observed outcomes at last n timesteps")
    print(df_ground_truth_observations)

    print("\nResults")
    print(df_result.iloc[:, 2:].round(2))
