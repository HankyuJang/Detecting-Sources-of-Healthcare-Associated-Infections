from eval_metrics import *
from get_seeds import *

def prepare_GT_data(args, simul_minute_level, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, GT_quality):

    k_total = np.sum(number_of_seeds_over_time)
    T = simul_minute_level.n_timesteps -1 # T is the index of the last timestep
    simul_minute_level.set_n_replicates(1)

    counter=0 # this is for setting counter for UVA_v4 graph.
    while True: # Ensure that |P| is large enough.
        # Initilaize 2-d seeds_array
        seeds_array = np.zeros((simul_minute_level.n_timesteps, simul_minute_level.number_of_nodes)).astype(bool)
        # Set seeds at multiple timesteps
        list_of_seed_idx_arrays = get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time)

        for t, seed_idx_array in enumerate(list_of_seed_idx_arrays):
            seeds_array[t, seed_idx_array] = True

        simul_minute_level.set_seeds(seeds_array)
        simul_minute_level.simulate()

        # NOTE: these arrays are 3-d. (num_rep, timestep, nodes)
        probability_array = simul_minute_level.probability_array
        infection_array = simul_minute_level.infection_array

        # Get P and N (ground truth infections and non-infections) over time
        rep_idx_to_select = 0
        list_of_sets_of_P, list_of_sets_of_N = get_P_N_over_time(infection_array, rep_idx_to_select, list_of_sets_of_V)

        obs_state_array = infection_array[rep_idx_to_select, :, :]

        I1_list_of_sets = []
        for t in range(simul_minute_level.n_timesteps):
            I1 = set(obs_state_array[t].nonzero()[0])
            I1_list_of_sets.append(I1)

        len_P_T = len(list_of_sets_of_P[T])
        len_N_T = len(list_of_sets_of_N[T])

        if args.name == "Karate_temporal":
            len_P_prime_T = len_P_T - k_total # P_primes serves as a proxy for additional infection at T
            print("Additional infection at T: {:.3f}. If the value is (-), then infection at T < size of seed set".format(len_P_prime_T / (len_P_prime_T + len_N_T)))
            # Ensure that seed set yields 5 - 10 % infections at time T, and a large number of recovery events
            if 0.05 <= len_P_prime_T / (len_P_prime_T + len_N_T) <= 0.1 : 
                print("Seeds yields additional 5 - 10% infection at T")
                print("-"*20)
                break
        elif args.name == "UIHC_HCP_patient_room_withinHCPxPx":
            # Ensure that seed set yields 2 - 10 % infections at time T, and a large number of recovery events
            print("P_T: {:.3f}, N_T: {:.3f}".format(len_P_T, len_N_T))
            if 0.02 <= len_P_T / (len_P_T + len_N_T) <= 0.1 : 
                print("Seeds yields 2 - 10% infection at T")
                print("-"*20)
                break
        elif args.name in ["G_Carilion", "G_Carilion_v3"]:
            print("P_T: {:.3f}, N_T: {:.3f}".format(len_P_T, len_N_T))
            # Ensure that seed set yields 2 - 10 % infections at time T
            if 0.02 <= len_P_T / (len_P_T + len_N_T) <= 0.1 : 
                print("Seeds yields 2 - 10% infection at T")
                print("-"*20)
                break
        elif args.name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]:
            print("P_T: {:.3f}, N_T: {:.3f}".format(len_P_T, len_N_T))
            # Ensure that seed set yields 2 - 10 % infections at time T
            if 0.02 <= len_P_T / (len_P_T + len_N_T) <= 0.1 : 
                print("Seeds yields 2 - 10% infection at T")
                print("-"*20)
                break
        if args.name=="G_UVA_v4":
            if counter < 10: # this makes it to explore 10 different seeds, before modifying the parameters.
                print(counter)
                counter += 1
                continue
            else:
                counter=0

        # Adjust parameters if infection fraction is out of range
        # Infection fraction is too large. -> Reduce the parameters. Otherwise Increase parameters
        if len_P_T / (len_P_T + len_N_T) > 0.1:
            print("-"*40)
            print("Too much infection. Dividing parameters (rho, d, q, pi) by 1.5")
            print("-"*40)
            simul_minute_level.rho /= 1.5
            simul_minute_level.d /= 1.5
            simul_minute_level.q /= 1.5
            simul_minute_level.pi /= 1.5
        elif len_P_T / (len_P_T + len_N_T) < 0.05:
            print("-"*40)
            print("Not much infection. Multiplying parameters (rho, d, q, pi) by 1.5")
            print("-"*40)
            simul_minute_level.rho *= 1.5
            simul_minute_level.d *= 1.5
            simul_minute_level.q *= 1.5
            simul_minute_level.pi *= 1.5
        print("Parameter values: rho: {}, d: {}, q: {}, pi: {}".format(simul_minute_level.rho, simul_minute_level.d, simul_minute_level.q, simul_minute_level.pi))

    print("\nGT seeds. timestep array and corresponding seed array: {}".format(seeds_array.nonzero()))
    print("P over last {} timesteps: {}".format(n_t_for_eval, list_of_sets_of_P[-n_t_for_eval:]))
    print("N over last {} timesteps: {}".format(n_t_for_eval, list_of_sets_of_N[-n_t_for_eval:]))

    return seeds_array, obs_state_array, I1_list_of_sets, list_of_sets_of_P, list_of_sets_of_N

# 1. Get the seed set at multiple timesteps (e.g., k seeds at time 0, k seeds at time 1 (on non-overlapping nodes)
# 2. Run simulation for 100 times and for each fun, compute P, N, and the scores (score is computed by itself vs all remaining)
# 3. Based on the scores and the given GT_quality, choose the outcome as the observed outcome
# def prepare_GT_data(simul, args, day0_people_idx_array, day1_people_idx_array, V_T, V_T_1, GT_quality):
def DEPRECATED_prepare_GT_data(args, simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval, GT_quality):

    k_total = np.sum(number_of_seeds_over_time)
    T = simul.n_timesteps -1 # T is the index of the last timestep

    while True: 
        # Initilaize 2-d seeds_array
        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        # Set seeds at multiple timesteps
        list_of_seed_idx_arrays = get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time)

        for t, seed_idx_array in enumerate(list_of_seed_idx_arrays):
            seeds_array[t, seed_idx_array] = True

        simul.set_seeds(seeds_array)
        simul.simulate()

        # NOTE: these arrays are 3-d. (num_rep, timestep, nodes)
        probability_array = simul.probability_array
        infection_array = simul.infection_array

        MCC_array = np.zeros((simul.n_replicates))

        # Set the rep_num simulation as the observed, and compute the scores based on the rest to this
        for rep_num in range(simul.n_replicates):
            obs_state = infection_array[rep_num, :, :]

            # Check if the observed number of infection at T is reasonable
            # if not is_num_P_T_reasonable(obs_state[T], list_of_people_idx_arrays[T], 0.05, 0.1, k_total):
                # MCC_array[rep_num] = -1
                # continue

            # Get probability arrays except that of the current replicate
            idx_except_itself = [idx for idx in range(simul.n_replicates) if idx != rep_num]
            # probability_array_except_itself = probability_array[idx_except_itself, :, :]
            infection_array_except_itself = infection_array[idx_except_itself, :, :]

            avg_tn, avg_fp, avg_fn, avg_tp, avg_f1, avg_mcc = evaluate_from_multiple_timesteps(obs_state, infection_array_except_itself, list_of_people_idx_arrays, n_t_for_eval)

            # Get P and N (ground truth infections and non-infections) over time
            # list_of_sets_of_P, list_of_sets_of_N = get_P_N_over_time(infection_array, rep_num, list_of_sets_of_V)

            # Compute P_hit and N_hit over time
            # list_of_P_hit, list_of_N_hit = get_P_hit_N_hit_over_time(list_of_sets_of_P, list_of_sets_of_N, probability_array)

            # Compute expected TP, TN, FP, FN for the last n_t_for_eval timesteps
            # TP, TN, FP, FN, F1, MCC = compute_scores_from_multiple_time_steps(list_of_sets_of_P, list_of_sets_of_N, list_of_P_hit, list_of_N_hit, n_t_for_eval)

            MCC_array[rep_num] = avg_mcc

        # If we observe all nans for MCC scores, repeat.
        if np.isnan(MCC_array).all():
            continue
        # Ensure that the MCC score is > 0.3 if not, repeat
        max_MCC = np.nanmax(MCC_array)
        median_MCC = np.nanmedian(MCC_array)
        # if max_MCC < 0.4:
        if GT_quality == "best" and max_MCC < 0.3:
            print("best MCC: {:.3f}".format(max_MCC))
            print("best seed set is not good enough.")
            continue

        if GT_quality == "bad" and median_MCC < 0.1:
            print("median MCC: {:.3f}".format(median_MCC))
            print("median seed set is not good enough.")
            continue

        if GT_quality == "bad" and median_MCC > 0.3:
            print("median MCC: {:.3f}".format(median_MCC))
            print("median seed set is too good to be used as a bad GT seedset.")
            continue

        if GT_quality == "best":
            # Selecting the simulation that is the best 
            # rep_idx_to_select = MCC_array.argmax()
            rep_idx_to_select = np.nanargmax(MCC_array)
        # NOTE: there's no meaning of choosing median quality ground set seeds. So it won't be used.
        elif GT_quality == "median":
            rep_idx_to_select = np.nanargmedian(MCC_array)
        elif GT_quality == "bad": # Additional set of experiment on the bad choice of GT quality
            # Selecting the simulation that is the best 
            # rep_idx_to_select = MCC_array.argmax()
            #NOTE: choosing min MCC yields MCC=0 for all GT and algorithms
            # rep_idx_to_select = np.nanargmedian(MCC_array) # there's no method such as nanargmedian
            rep_idx_to_select = np.nanargmin(np.abs(MCC_array-median_MCC)) # Idea from https://stackoverflow.com/a/28349688

        # Get P and N (ground truth infections and non-infections) over time
        list_of_sets_of_P, list_of_sets_of_N = get_P_N_over_time(infection_array, rep_idx_to_select, list_of_sets_of_V)

        obs_state_array = infection_array[rep_idx_to_select, :, :]

        I1_list_of_sets = []
        for t in range(simul.n_timesteps):
            I1 = set(obs_state_array[t].nonzero()[0])
            I1_list_of_sets.append(I1)

        # if len(P) >= 3+k: # Ensure that |P| is large enough. Maybe at least 3+k ?
        # NOTE: Should we keep this?
        # if len(list_of_sets_of_P[T]) >= 1 + total_number_of_seeds: # Ensure that |P| is large enough. Maybe at least one additional infection?
            # print("\nOutbreak observed. Selecting these as ground truth seeds and P, N")

        len_P_T = len(list_of_sets_of_P[T])
        len_N_T = len(list_of_sets_of_N[T])

        if args.name == "Karate_temporal":
            len_P_prime_T = len_P_T - k_total # P_primes serves as a proxy for additional infection at T
            print("Additional infection at T: {:.3f}. If the value is (-), then infection at T < size of seed set".format(len_P_prime_T / (len_P_prime_T + len_N_T)))
            # Ensure that seed set yields 5 - 10 % infections at time T, and a large number of recovery events
            if 0.05 <= len_P_prime_T / (len_P_prime_T + len_N_T) <= 0.1 : 
                print("Seeds yields additional 5 - 10% infection at T")
                print("-"*20)
                break
        elif args.name == "UIHC_HCP_patient_room_withinHCPxPx":
            # Ensure that seed set yields 5 - 10 % infections at time T, and a large number of recovery events
            print("P_T: {:.3f}, N_T: {:.3f}".format(len_P_T, len_N_T))
            if 0.05 <= len_P_T / (len_P_T + len_N_T) <= 0.1 : 
                print("Seeds yields 5 - 10% infection at T")
                print("-"*20)
                break
        elif args.name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4", "G_Carilion", "G_Carilion_v3"]:
            print("P_T: {:.3f}, N_T: {:.3f}".format(len_P_T, len_N_T))
            if args.name=="G_Carilion" and args.seeds_per_t == 1:
                if 0.02 <= len_P_T / (len_P_T + len_N_T) <= 0.1 : 
                    print("Seeds yields 2 - 10% infection at T")
                    print("-"*20)
                    break
            # Ensure that seed set yields 3 - 10 % infections at time T, and a large number of recovery events
            if 0.05 <= len_P_T / (len_P_T + len_N_T) <= 0.1 : 
                print("Seeds yields 5 - 10% infection at T")
                print("-"*20)
                break

    print("\nGT seeds. timestep array and corresponding seed array: {}".format(seeds_array.nonzero()))
    print("P over last {} timesteps: {}".format(n_t_for_eval, list_of_sets_of_P[-n_t_for_eval:]))
    print("N over last {} timesteps: {}".format(n_t_for_eval, list_of_sets_of_N[-n_t_for_eval:]))

    return seeds_array, obs_state_array, I1_list_of_sets, MCC_array, list_of_sets_of_P, list_of_sets_of_N

def prepare_GT_data_quality_x(args, simul, list_of_people_idx_arrays, list_of_sets_of_V, number_of_seeds_over_time, n_t_for_eval):

    k_total = np.sum(number_of_seeds_over_time)
    T = simul.n_timesteps -1 # T is the index of the last timestep
    simul.set_n_replicates(1)

    while True: # Ensure that |P| is large enough. TODO Maybe at least 3+k ?
        # Initilaize 2-d seeds_array
        seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
        # Set seeds at multiple timesteps
        list_of_seed_idx_arrays = get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time)

        for t, seed_idx_array in enumerate(list_of_seed_idx_arrays):
            seeds_array[t, seed_idx_array] = True

        simul.set_seeds(seeds_array)
        simul.simulate()

        # NOTE: these arrays are 3-d. (num_rep, timestep, nodes)
        probability_array = simul.probability_array
        infection_array = simul.infection_array

        # Get P and N (ground truth infections and non-infections) over time
        rep_idx_to_select = 0
        list_of_sets_of_P, list_of_sets_of_N = get_P_N_over_time(infection_array, rep_idx_to_select, list_of_sets_of_V)

        obs_state_array = infection_array[rep_idx_to_select, :, :]

        I1_list_of_sets = []
        for t in range(simul.n_timesteps):
            I1 = set(obs_state_array[t].nonzero()[0])
            I1_list_of_sets.append(I1)

        len_P_T = len(list_of_sets_of_P[T])
        len_N_T = len(list_of_sets_of_N[T])

        if args.name == "Karate_temporal":
            len_P_prime_T = len_P_T - k_total # P_primes serves as a proxy for additional infection at T
            print("Additional infection at T: {:.3f}. If the value is (-), then infection at T < size of seed set".format(len_P_prime_T / (len_P_prime_T + len_N_T)))
            # Ensure that seed set yields 5 - 10 % infections at time T, and a large number of recovery events
            if 0.05 < len_P_prime_T / (len_P_prime_T + len_N_T) < 0.1 : 
                print("Seeds yields additional 5 - 10% infection at T")
                print("-"*20)
                break
        elif args.name == "UIHC_HCP_patient_room_withinHCPxPx" and args.sampled == True:
            # Ensure that seed set yields 5 - 10 % infections at time T, and a large number of recovery events
            print("P_T: {:.3f}, N_T: {:.3f}".format(len_P_T, len_N_T))
            if 0.05 < len_P_T / (len_P_T + len_N_T) < 0.1 : 
                print("Seeds yields 5 - 10% infection at T")
                print("-"*20)
                break
        elif args.name == "UIHC_HCP_patient_room_withinHCPxPx" and args.sampled == False:
            # Ensure that seed set yields 5 - 10 % infections at time T, and a large number of recovery events
            print("P_T: {:.3f}, N_T: {:.3f}".format(len_P_T, len_N_T))
            if 0.05 < len_P_T / (len_P_T + len_N_T) < 0.1 : 
                print("Seeds yields 5 - 10% infection at T")
                print("-"*20)
                break

    print("\nGT seeds. timestep array and corresponding seed array: {}".format(seeds_array.nonzero()))
    print("P over last {} timesteps: {}".format(n_t_for_eval, list_of_sets_of_P[-n_t_for_eval:]))
    print("N over last {} timesteps: {}".format(n_t_for_eval, list_of_sets_of_N[-n_t_for_eval:]))

    return seeds_array, obs_state_array, I1_list_of_sets, list_of_sets_of_P, list_of_sets_of_N

# def is_num_P_T_reasonable(obs_state_T, people_idx_array_T, lower_bound, upper_bound, k_total):
    # V_T = set(people_idx_array_T)
    # P_T = set(obs_state_T.nonzero()[0])
    # N_T = V_T - P_T

    # len_P_T = len(P_T)
    # len_N_T = len(N_T)
    # len_P_prime_T = len_P_T - k_total

    # return lower_bound < len_P_prime_T / (len_P_prime_T + len_N_T) < upper_bound

# compute P_over_time and N_over_time
def get_P_N_over_time(infection_array, rep_num, list_of_sets_of_V):
    infection_array_2d = infection_array[rep_num, :, :]
    list_of_sets_of_P = []
    list_of_sets_of_N = []
    for t, V_t in enumerate(list_of_sets_of_V):
        P_t = set(infection_array_2d[t, :].nonzero()[0]) # This set contains any infection node (including isolates)
        # P_t = set(infection_array_2d[t, people_idx_list_t].nonzero()[0]) #NOTE: This was a bug! subselecting the indicies of an arary makes the array size smaller, and thereby leads to index mismatch!
        P_t = P_t.intersection(V_t) # Now this contains only infected nodes that are not isolates.
        N_t = V_t - P_t # By taking difference, N_t is uninfected nodes that are not isolates
        list_of_sets_of_P.append(P_t)
        list_of_sets_of_N.append(N_t)
    return list_of_sets_of_P, list_of_sets_of_N

# https://stackoverflow.com/a/64015677
def argmedian(x):
  return np.argpartition(x, len(x) // 2)[len(x) // 2]

