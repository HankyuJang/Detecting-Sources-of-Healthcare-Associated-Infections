from eval_metrics import *
from get_seeds import *
from tqdm import tqdm
import copy
from heapq import heappop, heappush, heapify

# Evaluate the solution
def evaluate_solution_seeds(simul, list_of_people_idx_arrays, GT_seeds_array, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval):

    simul.set_seeds(seeds_array) # set the seeds
    simul.simulate()
    probability_array = simul.probability_array
    infection_array = simul.infection_array

    S = set(GT_seeds_array.nonzero()[1])
    solution_S = set(seeds_array.nonzero()[1])
    solution_n_S = len(solution_S)
    solution_n_S_correct = len(set(solution_S).intersection(set(S)))

    # Computing loss
    solution_loss_1, solution_loss_total = compute_loss_per_timestep(simul, seeds_array, obs_state)
    # Compute scores
    list_of_P_hit, list_of_N_hit = get_P_hit_N_hit_over_time(list_of_sets_of_P, list_of_sets_of_N, probability_array)
    # Compute expected TP, TN, FP, FN for the last n_t_for_eval timesteps
    # TP, TN, FP, FN, F1, MCC = compute_scores_from_multiple_time_steps(list_of_sets_of_P, list_of_sets_of_N, list_of_P_hit, list_of_N_hit, n_t_for_eval)

    TN, FP, FN, TP, F1, MCC = evaluate_from_multiple_timesteps(obs_state, infection_array, list_of_people_idx_arrays, n_t_for_eval)

    return seeds_array, solution_n_S, solution_n_S_correct, solution_loss_1, solution_loss_total, \
            list_of_P_hit, list_of_N_hit, TP, TN, FP, FN, F1, MCC


# Returns True if any of the constraint is violated
def check_knapsack_constraints_violated(simul, array_of_knapsack_constraints_on_f, list_of_sets_of_N, \
                                        f_gain_to_empty_dict_over_time, X, S):
    # If any of the knapsack constraints are violated, summing up this boolean array > 0
    flag_knapsack_violated_array = np.zeros((simul.n_timesteps)).astype(bool)
    # Check if any of the knapsack constraints are violated. if so, break the while loop.
    for knapsack_t, knapsack_constraint in enumerate(array_of_knapsack_constraints_on_f):
        # NOTE: values are in floating point, so to compare if it's 0, need to compare if it's very close to 0
        if -1e-5 < knapsack_constraint < 1e-5: # do nothing if knapsack constraint is 0.
            continue
        else:
            N_t = list_of_sets_of_N[knapsack_t]
            f_t_gain_to_empty_dict = f_gain_to_empty_dict_over_time[knapsack_t]
            f_t_hat = f_hat(knapsack_t, f_t_gain_to_empty_dict, simul, X=X, N=N_t, S=S)
            knapsack_constraint_on_f_t = array_of_knapsack_constraints_on_f[knapsack_t]
            if f_t_hat > knapsack_constraint_on_f_t:
                flag_knapsack_violated_array[knapsack_t] = True
    return np.sum(flag_knapsack_violated_array) > 0

# Returns 2-d numpy boolean array, where indices of seeds are 'True'
def prep_seeds_2d_bool_array(simul, list_of_tuple_of_seeds):
    seeds_2d_bool = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    for (seed_t, seed_idx) in list_of_tuple_of_seeds:
        seeds_2d_bool[seed_t, seed_idx] = True
    return seeds_2d_bool

# Returns a list of tuple of (time, candidate seed). For all timesteps w/ GT seeds are at, find indicies of people.
def get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time):

    S_candidate_list = []

    timestep_array_with_GT_seeds = number_of_seeds_over_time.nonzero()[0]
    for t in timestep_array_with_GT_seeds:
        array_people_idx_at_t = list_of_people_idx_arrays[t]
        S_candidate_list_at_t = [(t, people_idx) for people_idx in array_people_idx_at_t]
        S_candidate_list.extend(S_candidate_list_at_t)
    return S_candidate_list

# Ground truth
def compute_GT_loss_per_timestep(simul, list_of_people_idx_arrays, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval):

    simul.set_seeds(seeds_array)
    simul.simulate()
    probability_array = simul.probability_array
    infection_array = simul.infection_array

    # Computing loss
    GT_loss_1, GT_loss_total = compute_loss_per_timestep(simul, seeds_array, obs_state)
    # Compute scores
    list_of_P_hit, list_of_N_hit = get_P_hit_N_hit_over_time(list_of_sets_of_P, list_of_sets_of_N, probability_array)
    # Compute expected TP, TN, FP, FN for the last n_t_for_eval timesteps
    # TP, TN, FP, FN, F1, MCC = compute_scores_from_multiple_time_steps(list_of_sets_of_P, list_of_sets_of_N, list_of_P_hit, list_of_N_hit, n_t_for_eval)
    TN, FP, FN, TP, F1, MCC = evaluate_from_multiple_timesteps(obs_state, infection_array, list_of_people_idx_arrays, n_t_for_eval)

    return GT_loss_1, GT_loss_total, \
            list_of_P_hit, list_of_N_hit, \
            TP, TN, FP, FN, F1, MCC

# NOTE: modify this later
def run_B_degree_report_loss_per_timestep():

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


    # compute hit ratios
    P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio = get_pos_hit_neg_hit(nodes_in_P, nodes_in_N, probability_array)

    return B_degree_S, B_degree_n_S_correct, B_degree_loss_1, B_degree_loss_total, \
             P_hit, N_hit, P_hit_frac, N_hit_frac, P_N_hit_diff, P_N_hit_ratio

# Baseline
# NOTE: Maybe repeat the process multiple times, then take average evaluation score.
def run_BR_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                                    GT_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval):

    # numpy nonzero returns indices in 2-dimensions. we take the dim1 because this represents idx of nodes
    S = set(GT_seeds_array.nonzero()[1])

    # Set seeds at multiple timesteps
    # E.g., list_of_seed_idx_arrays = [array([18]), array([7]), array([], dtype=int64), array([], dtype=int64) ...]
    list_of_seed_idx_arrays = get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time)
    print("within run_BR_report_loss_per_timestep")
    print("list_of_seed_idx_arrays: {}".format(list_of_seed_idx_arrays))

    # Initilaize 2-d seeds_array
    seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    for t, seed_idx_array in enumerate(list_of_seed_idx_arrays):
        seeds_array[t, seed_idx_array] = True

    print("\nS_random. timestep array and corresponding seed array: {}".format(seeds_array.nonzero()))

    return evaluate_solution_seeds(simul, list_of_people_idx_arrays, GT_seeds_array, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)

# Problem1: Greedy
# For each of the five dictionaries returned, key is the method used in greedy source detection algorithm
def run_greedy_source_detection_report_loss_per_timestep(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
        GT_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=False):

    if flag_lazy:
        print("\nLazy greedy source detection")
        seeds_array = lazy_greedy_source_detection(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                list_of_sets_of_P, n_t_for_eval, obs_state)
    else:
        print("\nGreedy source detection")
        seeds_array = greedy_source_detection(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                list_of_sets_of_P, n_t_for_eval, obs_state)

    print("\nG1. timestep array and corresponding seed array: {}".format(seeds_array.nonzero()))
    return evaluate_solution_seeds(simul, list_of_people_idx_arrays, GT_seeds_array, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)

# Problem1: Greedy
# For each of the five dictionaries returned, key is the method used in greedy source detection algorithm
def run_MU_source_detection_report_loss_per_timestep(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
        GT_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, flag_lazy=False):

    if flag_lazy:
        print("\nLazy MU source detection")
        seeds_array = lazy_MU_source_detection(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                list_of_sets_of_P, n_t_for_eval, obs_state)
    else:
        print("\nGreedy source detection")
        seeds_array = MU_source_detection(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
                list_of_sets_of_P, n_t_for_eval, obs_state)

    print("\nG1. timestep array and corresponding seed array: {}".format(seeds_array.nonzero()))
    return evaluate_solution_seeds(simul, list_of_people_idx_arrays, GT_seeds_array, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)


# Problem2: ISCK
# NOTE: No cardinality constraint
# Multiple knapsack constraint. One constraint per timestep.
# Takes in a list of upper bound constraints for n_t_for_eval
# Algorithm: Solve Pi (permutation) using greedy on P, such that the solution satisfy multiple knapsack constraint.
# Rest of the algorithm is same as the previous version of ISCK.
def run_ISCK_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
        GT_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
        array_of_knapsack_constraints_on_f, flag_lazy=False, flag_knapsack_in_pi=False, n_ISCK_iter=1, compute_pi="greedy"):

    print("\nISCK source detection")

    # Run ISCK and detect seeds
    return ISCK(simul, GT_seeds_array, list_of_people_idx_arrays, number_of_seeds_over_time, \
                list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, obs_state, \
                array_of_knapsack_constraints_on_f, flag_lazy, flag_knapsack_in_pi, n_ISCK_iter, compute_pi)

# Problem3: Greedy Ratio
# For each of the five dictionaries returned, key is the method used in greedy source detection algorithm
# NOTE: Do not set flag_memoize = True for greedy ratio. Current implementations led to shutting down the server
def run_greedy_ratio_report_loss_per_timestep(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
        GT_seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f, flag_lazy=False,
        flag_g_constraint=False):

    print("\nGreedy Ratio")
    if flag_lazy:
        seeds_array, intermediary_results_dict = lazy_greedy_ratio(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, obs_state, array_of_penalty_on_f, flag_g_constraint)
    else:
        seeds_array, intermediary_results_dict = greedy_ratio(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
                list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, obs_state, array_of_penalty_on_f, flag_g_constraint)

    print("\nGR. timestep array and corresponding seed array: {}".format(seeds_array.nonzero()))

    return intermediary_results_dict, evaluate_solution_seeds(simul, list_of_people_idx_arrays, GT_seeds_array, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)

# -----------------------------------------------------------------------------------------------------------
# Seed candidates: people nodes over time where ground truth seeds are at.
# Cardinality: total number of ground truth seeds (summed up over time)
# Treat candidate seed as a tuple. That is (timestep, node_index)
def greedy_source_detection(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
        list_of_sets_of_P, n_t_for_eval, obs_state):
    T = simul.n_timesteps -1 # T is the index of the last timestep

    k = cardinality_constraint

    # Prepare list of candidate seeds as tuples
    S_candidate_list = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    # Greedily add a source that yields most of the infections in I_obs
    # Compute the loss for the last n timesteps
    S = set()
    # S_candidate_list = list(simul.people_nodes.nonzero()[0])
    for i in range(k):
        max_gain = -math.inf
        best_node = None
        if len(S) == 0:
            current_footperint_in_P = 0
        else:
            current_footperint_in_P = g(simul, S, list_of_sets_of_P, n_t_for_eval)

        for v in tqdm(S_candidate_list):
            S_temp = {v}
            S_temp.update(S)

            temp_footprint_in_P = g(simul, S_temp, list_of_sets_of_P, n_t_for_eval)

            temp_gain = (temp_footprint_in_P - current_footperint_in_P) 

            if temp_gain >= max_gain:
                max_gain = temp_gain
                best_node = v

        S.add(best_node)
        S_candidate_list.remove(best_node)

    seeds_greedy_solution = prep_seeds_2d_bool_array(simul, S)

    return seeds_greedy_solution

# -----------------------------------------------------------------------------------------------------------
# Seed candidates: people nodes over time where ground truth seeds are at.
# Cardinality: total number of ground truth seeds (summed up over time)
# Treat candidate seed as a tuple. That is (timestep, node_index)
# NOTE: This implementation is Lazy greedy implementation.
def lazy_greedy_source_detection(simul, cardinality_constraint, focus_obs1, list_of_people_idx_arrays, number_of_seeds_over_time, \
        list_of_sets_of_P, n_t_for_eval, obs_state):
    T = simul.n_timesteps -1 # T is the index of the last timestep

    k = cardinality_constraint

    # Prepare list of candidate seeds as tuples
    S_candidate_list = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    print("computing g_gain to empty set")
    g_gain_to_empty_dict = precompute_g_gain_to_empty_set(simul, list_of_people_idx_arrays, number_of_seeds_over_time, list_of_sets_of_P, n_t_for_eval)
    print("Complete - computing g_gain to empty set")

    ###############################################################################
    # STEP1: Heap
    # NOTE: There's no max heap implementation.
    # When heappush, muliply value w/ -1.
    # Then, it serves as a max heap (that is min heap over negatives)
    h = []
    heapify(h)
    V = [] # V is the list of S_candidates

    print("Heap initialization...")
    for S_candidate in g_gain_to_empty_dict:
        gain = g_gain_to_empty_dict[S_candidate]
        heappush(h, (-1 * gain, S_candidate))
        V.append(S_candidate)
    print("Complete heap initialization...")
    obj_prev = 0 # gain of no seeds is 0

    pi_V = []

    ###############################################################################
    # STEP2: Lazy Greedy
    # Do this differently for the first iteration.
    # Simply, at the begging, add the node at the top of the heap to the solution.
    
    neg_obj_prev, v = heappop(h)
    obj_prev = -1 * neg_obj_prev
    pi_V.append(v)
    print("pi_V: {}".format(pi_V))
    print("obj_prev: {}".format(obj_prev))
    len_pi_V_prev = len(pi_V)

    # Starting from the second iteration, do the following
    while len(h) > 0:
        if len(pi_V) == k: # break if cardinality constraint is met
            break
        for j in tqdm(range(len(h))):
            if len(pi_V) == k: # break if cardinality constraint is met
                break

            _, v = heappop(h)
            pi_V.append(v) # NOTE: keep the same notation as the ISCK-Lazygreedy implemtation
            if len(h) == 0: # if the popped element was the last one, the heap is empty at this point
                break
            # temp footprint in P
            obj = g(simul, pi_V, list_of_sets_of_P, n_t_for_eval)
            gain = obj - obj_prev

            # NOTE: Get the upperbound gain of the maxheap
            neg_upperbound_gain, _ = h[0]

            # NOTE: we multiply the upperbound_gain by -1
            # if gain >= -1 * upperbound_gain: # keep v in the solution pi_V
            # if gain > -1 * neg_upperbound_gain: # keep v in the solution pi_V
            if gain >= -1 * neg_upperbound_gain and gain > 0: # keep v in the solution pi_V
                print("gain: {}, upperbound_gain: {}".format(gain, -1*neg_upperbound_gain))
                obj_prev = obj # update the obj_prev. This is simply g(X)
                break
            else:
                pi_V.remove(v) # remove v from the solution pi_V
                heappush(h, (-1 * gain, v))
            # There could be a case where upperbound_gain become 0. Then, do not add more nodes to the solution
            # if -1 * neg_upperbound_gain == 0:
                # break
        # If no seeds were added in the current iteration of the while loop, return current pi_V
        if len(pi_V) == len_pi_V_prev:
            print("pi_V: {}".format(pi_V))
            break

        len_pi_V_prev = len(pi_V)

    seeds_greedy_solution = prep_seeds_2d_bool_array(simul, pi_V)

    return seeds_greedy_solution

# For ISCK
def initialize_empty_lists():
    return [], [], [], [], [], \
            [], [], [], [], [], \
            [], [], []

# -----------------------------------------------------------------------------------------------------------
# NOTE: Changes from the previous version of ISCK:
# - No cardinality constraint.
# - Multiple Knapsack constraints
# - g is expected infection in P_T, P_{T-1}, ... -> these are computed on the latest n_t_for_eval timesteps
# - f is expected infection in N_T, N_{T-1}, ... -> these are computed on the latest n_t_for_eval timesteps
# - f_T is expected infection in N_T. Knapsack constraint on this
# - f_{T-1} is expected infection in N_{T-1}. Knapsack constraint on this 
# - get_pi is greedy on g, that satisfy knapsack constraints
# def ISCK(simul, V, P, N, k1, k2, how):
def ISCK(simul, GT_seeds_array, list_of_people_idx_arrays, number_of_seeds_over_time, \
        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, obs_state, \
        array_of_knapsack_constraints_on_f, flag_lazy, flag_knapsack_in_pi, n_ISCK_iter, compute_pi):

    ##############################
    # For ISCK, evalaluate the solution per iteration. Save each term in a list (L)
    L_seeds_array, L_ISCK_n_S, L_ISCK_n_S_correct, L_ISCK_loss_1, L_ISCK_loss_total,\
        L_list_of_P_hit, L_list_of_N_hit, L_TP, L_TN, L_FP, L_FN, L_F1, L_MCC = initialize_empty_lists()
    ##############################

    T = simul.n_timesteps -1 # T is the index of the last timestep

    max_iteration = n_ISCK_iter

    # precompute f(j|empty) over time.
    print("computing f_gain to empty set over time")
    f_gain_to_empty_dict_over_time = precompute_f_gain_to_empty_set_over_time(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
            list_of_sets_of_N, array_of_knapsack_constraints_on_f)
    print("Complete - computing f_gain to empty set over time")

    # precompute f(j|empty)
    print("computing g_gain to empty set")
    g_gain_to_empty_dict = precompute_g_gain_to_empty_set(simul, list_of_people_idx_arrays, number_of_seeds_over_time, list_of_sets_of_P, n_t_for_eval)
    print("Complete - computing g_gain to empty set")

    S = set()
    idx_while = 0
    while True:
        if idx_while == max_iteration:
            break

        if compute_pi=="greedy":
            W=0
            if flag_lazy:
                pi = get_pi_lazy(flag_knapsack_in_pi, array_of_knapsack_constraints_on_f, f_gain_to_empty_dict_over_time, simul, \
                        list_of_people_idx_arrays, number_of_seeds_over_time, \
                        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                        S, g_gain_to_empty_dict)
            else:
                pi = get_pi(flag_knapsack_in_pi, array_of_knapsack_constraints_on_f, f_gain_to_empty_dict_over_time, simul, \
                        list_of_people_idx_arrays, number_of_seeds_over_time, \
                        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                        S)
        elif compute_pi=="multiplicative_update":
            A = get_A(list_of_people_idx_arrays, number_of_seeds_over_time, n_t_for_eval, f_gain_to_empty_dict_over_time)

            if idx_while == 0:
                print("A: {}".format(A))

            m, n = A.shape
            b = array_of_knapsack_constraints_on_f[-m:]
            b_reshaped = np.tile(b.reshape((b.shape[0], 1)), n) # (m,) -> (m, n) by concatenating copied values over columns
            b_div_A = b_reshaped / A
            W = np.min( b_div_A[b_div_A > 0] )
            update_factor_lambda = 2*np.power(np.e, W)
            # if idx_while == 0:
                # print("A: {}".format(A))
                # print("b: {}".format(b))
                # print("A.shape: {}".format(A.shape))
                # print("b.shape: {}".format(b.shape))
                # print("W: {}".format(W))
                # print("update_factor_lambda: {}".format(update_factor_lambda))

            pi = get_pi_multiplicative_update(flag_knapsack_in_pi, array_of_knapsack_constraints_on_f, 
                    f_gain_to_empty_dict_over_time, g_gain_to_empty_dict, simul, \
                    list_of_people_idx_arrays, number_of_seeds_over_time, \
                    list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
                    S, m, n, A, b, update_factor_lambda, flag_lazy)

        S_new = set()
        i=0
        #--------------------------
        # while len(S_new) < k1 and f_hat(f_gain_to_empty_dict, simul, X=S, N=N, S=S_new) < k2:
        # while f_hat(T, f_T_gain_to_empty_dict, simul, X=S, N=N_T, S=S_new) <= array_of_knapsack_constraints_on_f[T] and \
                # f_hat(T-1, f_T_1_gain_to_empty_dict, simul, X=S, N=N_T_1, S=S_new) <= array_of_knapsack_constraints_on_f[T-1]:
        while True:
            # print("len(pi): {}".format(len(pi)))
            if not flag_knapsack_in_pi: # dont need to check if it violated knapsack constraint if pi already satisfies this 
                flag_knapsack_violated = check_knapsack_constraints_violated(simul, array_of_knapsack_constraints_on_f, list_of_sets_of_N, \
                                            f_gain_to_empty_dict_over_time, X=S, S=S_new)
                if flag_knapsack_violated: # If True, break the while loop
                    break
        #--------------------------
            if i == len(pi):
                break
            S_new.add(pi[i])
            i+=1
        #------------------------------------------------------
        # For the newly selected set of seed nodes, evaluate the seeds and save them
        seeds_array = prep_seeds_2d_bool_array(simul, S_new)
        print("\nISCK {} iteration".format(idx_while))
        # print("\nISCK {} iteration.\ntimestep array and corresponding seed array: {}".format(idx_while, seeds_array.nonzero()))
        _, ISCK_n_S, ISCK_n_S_correct, ISCK_loss_1, ISCK_loss_total, \
                list_of_P_hit, list_of_N_hit, TP, TN, FP, FN, F1, MCC = evaluate_solution_seeds(
                        simul, list_of_people_idx_arrays, GT_seeds_array, seeds_array, obs_state, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval)
        print("n_S: {}, n_S_correct: {}, TP: {:.3f}, MCC: {:.3f}".format(ISCK_n_S, ISCK_n_S_correct, TP, MCC))
        L_seeds_array.append(seeds_array)
        L_ISCK_n_S.append(ISCK_n_S)
        L_ISCK_n_S_correct.append(ISCK_n_S_correct)
        L_ISCK_loss_1.append(ISCK_loss_1)
        L_ISCK_loss_total.append(ISCK_loss_total)
        L_list_of_P_hit.append(list_of_P_hit)
        L_list_of_N_hit.append(list_of_N_hit)
        L_TP.append(TP)
        L_TN.append(TN)
        L_FP.append(FP)
        L_FN.append(FN)
        L_F1.append(F1)
        L_MCC.append(MCC)
        #------------------------------------------------------
        if S_new == S:
            break
        S = S_new
        # print("G(): ", g(simul, S, P))
        # print("f(): ", f(simul, S, N))
        idx_while += 1
    # return S
    return L_seeds_array, L_ISCK_n_S, L_ISCK_n_S_correct, L_ISCK_loss_1, L_ISCK_loss_total,\
            L_list_of_P_hit, L_list_of_N_hit, L_TP, L_TN, L_FP, L_FN, L_F1, L_MCC, W

def precompute_f_gain_to_empty_set_over_time(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
        list_of_sets_of_N, array_of_knapsack_constraints_on_f):
    # the value of this dictionary contains a dictionary of f gain to empty dict.
    # NOTE: If an entry of array_of_knapsack_constraints_on_f is 0, add an empty dict.
    f_gain_to_empty_dict_over_time = []
    for knapsack_t, knapsack_constraint in enumerate(array_of_knapsack_constraints_on_f):
        # NOTE: values are in floating point, so to compare if it's 0, need to compare if it's very close to 0
        if -1e-5 < knapsack_constraint < 1e-5: 
            f_gain_to_empty_dict_over_time.append(dict())
        else:
            f_t_gain_to_empty_dict = precompute_f_gain_to_empty_set(knapsack_t, simul, list_of_people_idx_arrays, number_of_seeds_over_time, list_of_sets_of_N)
            f_gain_to_empty_dict_over_time.append(f_t_gain_to_empty_dict)
    return f_gain_to_empty_dict_over_time

# For all the nodes j in people nodes, compute f(j|empty), save it as a dictionary
# t is the timestamp to be used to compute f_gain
def precompute_f_gain_to_empty_set(t, simul, list_of_people_idx_arrays, number_of_seeds_over_time, list_of_sets_of_N):
    N = list_of_sets_of_N[t]
    f_gain_to_empty_dict = {}

    # Prepare list of candidate seeds as tuples
    S_candidate_list = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    for j in tqdm(S_candidate_list):
        f_gain_to_empty_dict[j] = f(t, simul, [j], N)

    return f_gain_to_empty_dict

# NOTE: this function is for lazy ISCK
def precompute_g_gain_to_empty_set(simul, list_of_people_idx_arrays, number_of_seeds_over_time, list_of_sets_of_P, n_t_for_eval):
    
    g_gain_to_empty_dict = {}

    # Prepare list of candidate seeds as tuples
    # timestep_array_with_GT_seeds = number_of_seeds_over_time.nonzero()[0]
    S_candidate_list = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    for j in tqdm(S_candidate_list):
        g_gain_to_empty_dict[j] = g(simul, [j], list_of_sets_of_P, n_t_for_eval)
        # print("node: {}, g_gain: {}".format(j, g_gain_to_empty_dict[j]))

    return g_gain_to_empty_dict

def get_pi_multiplicative_update(flag_knapsack_in_pi, array_of_knapsack_constraints_on_f, 
        f_gain_to_empty_dict_over_time, g_gain_to_empty_dict, simul, \
        list_of_people_idx_arrays, number_of_seeds_over_time, \
        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
        S, m, n, A, b, update_factor_lambda, flag_lazy):

    epsilon = pow(10, -6)
    T = simul.n_timesteps -1 # T is the index of the last timestep

    # V is the set of candidate seeds
    # e.g., V = [(0, 123), (0, 128), ...]
    V = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    # add the index of the array in the tuple,
    # E.g., V_w_array_index = [((0, 123), 0), ((0, 128), 1), ...]
    V_w_array_idx = [(S_candidate, idx) for idx, S_candidate in enumerate(V)]
    V_copied = set(copy.deepcopy(V_w_array_idx))

    #########################################
    # Line1 - Initilaize S as a list (note: we need to keep track of the orders)
    S = []
    S_array_idx = []

    # Line2 - initialize w
    w = np.zeros((m))
    for i in range(m):
        w[i] = 1/b[i]

    # initialize min heap with lower bounds
    if flag_lazy:
        h = []
        heapify(h)
        for v, array_idx in V_copied:
            # term = np.sum(A[:, array_idx] * w) / g_gain_to_empty_dict[v]

            g_gain_to_empty_dict_value = g_gain_to_empty_dict[v]
            if g_gain_to_empty_dict_value < epsilon:
                term = np.sum(A[:, array_idx] * w) / epsilon
            else:
                term = np.sum(A[:, array_idx] * w) / g_gain_to_empty_dict_value

            heappush(h, (term, v, array_idx))

    # Line3-7
    while np.sum(b*w) <= update_factor_lambda and len(S) != n:
        # line 4
        # ----------------------------------------------------------------------------------
        # Original implementation
        if not flag_lazy:
            best_node = None
            best_node_array_index = None
            min_term = math.inf
            current_footperint_in_P = g(simul, S, list_of_sets_of_P, n_t_for_eval)

            for v, array_idx in V_copied:
                temp_footprint_in_P = g(simul, S + [v], list_of_sets_of_P, n_t_for_eval)
                temp_gain = (temp_footprint_in_P - current_footperint_in_P)
                if temp_gain <= 0: # NOTE: added due to stochasticity in the simulation
                    continue
                temp_term = np.sum(A[:, array_idx] * w) / temp_gain

                if temp_term < min_term:
                    best_node = v
                    best_node_array_idx = array_idx
                    min_term = temp_term
            if best_node == None:
                break
        # ----------------------------------------------------------------------------------
        # Lazy implementation
        if flag_lazy:
            best_node = None
            best_node_array_index = None
            # min_term = math.inf
            current_footperint_in_P = g(simul, S, list_of_sets_of_P, n_t_for_eval)

            for j in range(len(h)):
                _, v, array_idx = heappop(h)
                if len(h) == 0: # if the popped element was the last one, the heap is empty at this point
                    break
                temp_footprint_in_P = g(simul, S + [v], list_of_sets_of_P, n_t_for_eval)
                temp_gain = (temp_footprint_in_P - current_footperint_in_P)
                if temp_gain <= 0: # NOTE: added due to stochasticity in the simulation
                    continue
                temp_term = np.sum(A[:, array_idx] * w) / temp_gain

                # Get the lower bound term value of the minheap
                lowerbound_term, _, _ = h[0]

                if temp_term < lowerbound_term: 
                    # set v as the best node
                    best_node = v
                    best_node_array_idx = array_idx
                    break
                else:
                    heappush(h, (temp_term, v, array_idx))
                # if at the last iteration and no further nodes gives a better solution, break the for loop
                if j == len(h)-1: 
                    break
            if best_node == None:
                break
        # ----------------------------------------------------------------------------------
        # line 5
        # Inside the while loop
        S.append(best_node)
        S_array_idx.append(best_node_array_idx)
        V_copied.remove((best_node, best_node_array_idx))
        # line 6
        for i in range(m):
            w[i] = w[i] * np.power(update_factor_lambda, A[i, best_node_array_idx] / b[i])
        # print("S: {}".format(S))
        # print("w: {}".format(w))

    print("S: {}".format(S))
    print("best_node: {}".format(best_node))
    # NOTE: At this point, best_node can be None. if no node was chosen to be added.
    # NOTE: In this case, set it as the node that was added to the solution seeds at the last step
    if best_node == None:
        best_node = S[-1]

    # line 8
    # Characteristic vector of set S. NOTE: 0-1 vector of size |S|, that is True if element of the index is in S?
    x_S = np.zeros((n, 1)).astype(int)
    for array_idx in S_array_idx:
        x_S[array_idx, 0] = 1
    if np.all(np.dot(A, x_S) <= b):
        return S
    elif current_footperint_in_P >= g_gain_to_empty_dict[best_node]: # line 9
        S.remove(best_node)
        return S
    else: #line 10
        return [best_node]
    #########################################

# Helper function for multiplicate update
# NOTE: If f_t_gain_to_empty_dict[e] == 0, set the value of that idx as epsilon.
def get_A(list_of_people_idx_arrays, number_of_seeds_over_time, n_t_for_eval, f_gain_to_empty_dict_over_time):
    V = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    epsilon = pow(10, -6)

    # Compute A
    m = n_t_for_eval
    n = len(V)
    A = np.zeros((m, n))
    A_constraint_idx = 0
    for t, f_t_gain_to_empty_dict in enumerate(f_gain_to_empty_dict_over_time):
        if len(f_t_gain_to_empty_dict) == 0: # timesteps w/ no knapsack constraint have empty dict 
            continue
        for e_idx, e in enumerate(V): # Here, e is a tuple (t, candidate seed id)
            # A[A_constraint_idx, e_idx] = f_t_gain_to_empty_dict[e]
            A_value_for_the_idx = f_t_gain_to_empty_dict[e]
            if A_value_for_the_idx < epsilon:
                A[A_constraint_idx, e_idx] = epsilon
            else:
                A[A_constraint_idx, e_idx] = A_value_for_the_idx
        A_constraint_idx += 1
    # print("***********************************")
    # print("Inside get_A")
    # print("A: {}".format(A))

    return A

def get_pi(flag_knapsack_in_pi, array_of_knapsack_constraints_on_f, f_gain_to_empty_dict_over_time, simul, \
            list_of_people_idx_arrays, number_of_seeds_over_time, \
            list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
            S):
    T = simul.n_timesteps -1 # T is the index of the last timestep

    # V is the set of candidate seeds
    V = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    V_copied = set(copy.deepcopy(V))
    pi_V = []
    while len(V_copied) > 0:
        max_gain = -math.inf
        best_node = None
        # current_footperint_in_P = g(simul, pi_V, P)
        current_footperint_in_P = g(simul, pi_V, list_of_sets_of_P, n_t_for_eval)

        for v in V_copied:
            pi_V.append(v)
            temp_footprint_in_P = g(simul, pi_V, list_of_sets_of_P, n_t_for_eval)

            temp_gain = (temp_footprint_in_P - current_footperint_in_P) 
            
            # if how == "ratio":
                # temp_gain =  temp_gain / temp_f_hat
            # elif how == "diff":
                # temp_gain =  temp_gain - temp_f_hat

            if temp_gain >= max_gain:
                max_gain = temp_gain
                best_node = v
            pi_V.remove(v)

        # best_nodes are sometimes None. If this occurs, add all to pi_V then break
        #--------------------------
        if best_node == None:
            if flag_knapsack_in_pi:
                break
            else:# NOTE: in the current experiments, flag_knapsack_in_pi=True, so it never enters here.
                for v in V_copied:
                    pi_V.append(v)
                break

        #--------------------------
        if flag_knapsack_in_pi:
            flag_knapsack_violated = check_knapsack_constraints_violated(simul, array_of_knapsack_constraints_on_f, list_of_sets_of_N, \
                                        f_gain_to_empty_dict_over_time, X=S, S=set(pi_V))
            if flag_knapsack_violated: # If True, break the while loop
                break
        #--------------------------

        V_copied.remove(best_node) 
        pi_V.append(best_node)
        # print("len(pi): {}, pi_V: {}".format(len(pi_V), pi_V))
    return pi_V

def get_pi_lazy(flag_knapsack_in_pi, array_of_knapsack_constraints_on_f, f_gain_to_empty_dict_over_time, simul, \
            list_of_people_idx_arrays, number_of_seeds_over_time, \
            list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, \
            S, g_gain_to_empty_dict):

    T = simul.n_timesteps -1 # T is the index of the last timestep
    flag_knapsack_violated = False

    ###############################################################################
    # STEP1: Create heap from the precomputed g_gain_to_empty_dict
    # NOTE: There's no max heap implementation.
    # When heappush, muliply value w/ -1.
    # Then, it serves as a max heap (that is min heap over negatives)
    h = []
    heapify(h)
    V = [] # V is the list of S_candidates
    for S_candidate in g_gain_to_empty_dict:
        gain = g_gain_to_empty_dict[S_candidate]
        heappush(h, (-1 * gain, S_candidate))
        V.append(S_candidate)
    obj_prev = 0 # gain of no seeds is 0

    pi_V = []
    ###############################################################################
    # STEP2: Lazy Greedy
    # Do this differently for the first iteration.
    # Simply, at the begging, add the node at the top of the heap to the solution.
    neg_obj_prev, v = heappop(h)
    obj_prev = -1 * neg_obj_prev
    pi_V.append(v)
    print("pi_V: {}".format(pi_V))
    print("obj_prev: {}".format(obj_prev))
    len_pi_V_prev = len(pi_V)

    # V_copied = set(copy.deepcopy(V))
    # while len(V_copied) > 0:
    while len(h) > 0:
        if flag_knapsack_violated: # If True, break the while loop
            break

        for j in tqdm(range(len(h))):
            _, v = heappop(h)
            pi_V.append(v)
            if len(h) == 0: # if the popped element was the last one, the heap is empty at this point
                break
            # temp footprint in P
            obj = g(simul, pi_V, list_of_sets_of_P, n_t_for_eval)
            gain = obj - obj_prev

            # NOTE: Get the upperbound gain of the maxheap
            neg_upperbound_gain, _ = h[0]
            # NOTE: we multiply the upperbound_gain by -1
            if gain >= -1 * neg_upperbound_gain and gain > 0: # keep v in the solution pi_V
                # if flag_knapsack_in_pi is True, check if it violates any knapsack constraints. If so, break.
                if flag_knapsack_in_pi:
                    flag_knapsack_violated = check_knapsack_constraints_violated(simul, array_of_knapsack_constraints_on_f, list_of_sets_of_N, \
                                                f_gain_to_empty_dict_over_time, X=S, S=set(pi_V))
                    if flag_knapsack_violated: # If True, break the for loop
                        break

                print("gain: {}, upperbound_gain: {}".format(gain, -1*neg_upperbound_gain))
                obj_prev = obj # update the obj_prev. This is simply g(X)
                break
            else:
                pi_V.remove(v) # remove v from the solution pi_V
                heappush(h, (-1 * gain, v))

            if j == len(h)-1: # if at the last iteration and no further nodes gives a better solution,
                if flag_knapsack_in_pi:
                    h = [] # empty the heap
                    break
                else: # NOTE: in the current experiments, flag_knapsack_in_pi=True, so it never enters here.
                    # Empty the heap, add all the rest to the pi_V. Since h will be empty, it'll exit the outer while loop
                    while len(h) > 0:
                        _, v = heappop(h)
                        pi_V.append(v)
                    break

    return pi_V

# expected number of infections in P_t given X is the seed set
# Run simulations for some number of replicates, then compute avg infection of each node at the end of the timestep
# add parameters: some other parameters needed to run simulation
# NOTE: now g computes expected infection in P_T, P_{T-1}, ...
def g(simul, X, list_of_sets_of_P, n_t_for_eval):
    T = simul.n_timesteps -1 # T is the index of the last timestep

    if len(X) == 0:
        # print("No seed specified")
        return 0
    seeds_array = prep_seeds_2d_bool_array(simul, X)

    simul.set_seeds(seeds_array)
    simul.simulate()
    probability_array = simul.probability_array
    # infection_array = simul.infection_array
    
    # NOTE: these are for debugging
    # total_infection_at_T = np.sum(np.mean(probability_array[:, T, :], axis=0)) # average over replicates. Then take sum
    # print("Total infection at T: {}".format(total_infection_at_T))
    # P_T = list(list_of_sets_of_P[T])
    # print("P_T: {}".format(P_T))
    # total_infection_within_P_at_T = np.sum(np.mean(probability_array[:, T, P_T], axis=0))
    # print("Total infection wihtin P at T: {}".format(total_infection_within_P_at_T))
    # total_probability_sum = np.sum(probability_array)
    # print("Total probability sum over all replicates over all times over all nodes: {}".format(total_probability_sum))

    expected_infection_in_P = 0
    for t in range(n_t_for_eval): #t is 0, 1, ... so T-t is T, T-1, ... where T is the last timestep
        P_T_t = list(list_of_sets_of_P[T-t])
        expected_infection_in_P_at_T_t = np.sum(np.mean(probability_array[:, T-t, P_T_t], axis=0))
        expected_infection_in_P += expected_infection_in_P_at_T_t

    # if expected_infection_in_P == 0:
        # print("in this g(), expected infection in P is 0: seeds: {}".format(X))

    return expected_infection_in_P

# Compute this at timestep t
def f_hat(t, f_gain_to_empty_dict, simul, X, N, S):
    if len(S) == 0:
        return 0
    val1 = f(t, simul, X, N) - f_2(t, simul, X, N, S) 
    val1 = max(0, val1) + 0.00000001
    
    val = val1 + f_3(f_gain_to_empty_dict, simul, X, N, S)
    if val <0:
        print(" f_hat is negative")
    return val

# expected number of infections in N_t given X is the seed set
# Run simulations for some number of replicates, then compute avg infection of each node at the end of the timestep
# add parameters: some other parameters needed to run simulation
def f(t, simul, X, N):
    if X==None or len(X) == 0:
        return 0

    seeds_array = prep_seeds_2d_bool_array(simul, X)
    # seeds_array = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    # for (seed_t, seed_idx) in X: # infect seeds
        # seeds_array[seed_t, seed_idx] = True
    # seeds_array[0, list(X)] = True

    simul.set_seeds(seeds_array)
    simul.simulate()
    probability_array = simul.probability_array
    # infection_array = simul.infection_array

    expected_infection_in_N_at_t = np.sum(np.mean(probability_array[:, t, list(N)], axis=0))
    return expected_infection_in_N_at_t

# for j (node that is in X \ S), compute f(X) - f(X \ j)
# def f_gain_of_adding_j(X, S):
def f_2(t, simul, X, N, S):
    X_copied = copy.deepcopy(X)
    f_of_X = f(t, simul, X_copied, N)
    total = 0
    for j in X_copied - S:
        f_of_X_minus_j = f(t, simul, X_copied.remove(j), N)
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

# expected number of infections in P_T, P_{T-1}, ... and N_T, N_{T-1}. ...
# Run simulations for some number of replicates, then compute avg infection of each node at T, T-1, ...
# add parameters: some other parameters needed to run simulation
# NOTE: simulation object simul already has the updated seed set 
def g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f):
    T = simul.n_timesteps -1 # T is the index of the last timestep
    simul.simulate()
    probability_array = simul.probability_array

    expected_infection_in_P = 0
    expected_infection_in_N = 0

    for t in range(n_t_for_eval): #t is 0, 1, ... so T-t is T, T-1, ... where T is the last timestep
        P_T_t = list(list_of_sets_of_P[T-t])
        expected_infection_in_P_at_T_t = np.sum(np.mean(probability_array[:, T-t, P_T_t], axis=0))
        expected_infection_in_P += expected_infection_in_P_at_T_t

        penalty_T_t = array_of_penalty_on_f[T-t]
        N_T_t = list(list_of_sets_of_N[T-t])
        expected_infection_in_N_at_T_t = np.sum(np.mean(probability_array[:, T-t, N_T_t], axis=0))
        expected_infection_in_N += (penalty_T_t * expected_infection_in_N_at_T_t)

    return expected_infection_in_P, expected_infection_in_N, probability_array

# -----------------------------------------------------------------------------------------------------------
# Seed candidates: people nodes over time where ground truth seeds are at.
# NOTE: no Cardinality constraint.
# Treat candidate seed as a tuple. That is (timestep, node_index)
# NOTE: removed memoization
def greedy_ratio(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, obs_state, array_of_penalty_on_f,
        flag_g_constraint):

    epsilon = pow(10, -6)

    # Prepare list of candidate seeds as tuples
    S_candidate_list = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    # gain_ratio_best_list = []
    f_g_ratio_best_list = []
    S_list = []
    S = set()

    # NOTE: keep track of these for debugging purpose
    g_best_list = []
    f_best_list = []
    # g_gain_best_list = []
    # f_gain_best_list = []

    while len(S_candidate_list) > 0:
        S_best = None
        gain_ratio_best = math.inf
        # g_gain_best = math.inf
        # f_gain_best = math.inf

        # Compute g and f on the selected seeds till now
        seeds_2d_bool = prep_seeds_2d_bool_array(simul, S)
        simul.set_seeds(seeds_2d_bool)

        if len(S) == 0: # if no seeds, then no infection occurs.
            g_current_GR, f_current_GR = 0, 0
        else:
            g_current_GR, f_current_GR, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)
            if g_current_GR <= 0:
                g_current_GR = epsilon
            if f_current_GR <= 0:
                f_current_GR = epsilon
            # if g_current_GR <= 0 or f_current_GR <= 0:
                # break

        # Get the seed greedily that maximizes the ratio
        # Line 5
        for S_candidate in tqdm(S_candidate_list):
            S_temp = {S_candidate}
            S_temp.update(S)

            seeds_2d_bool = prep_seeds_2d_bool_array(simul, S_temp)
            simul.set_seeds(seeds_2d_bool)

            g_GR, f_GR, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

            # g_GR, f_GR, probability_array = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)
            if g_GR <= 0:
                g_GR = epsilon
            if f_GR <= 0:
                f_GR = epsilon

            # if g_GR > epsilon:
                # print("g_current_GR: {}, f_current_GR: {}".format(g_current_GR, f_current_GR))
                # print("g_GR: {}, f_GR: {}".format(g_GR, f_GR))

            # if g_GR <= 0 or f_GR <= 0:
                # continue

            # Here, g_gain can be 0 in the first iteration. If this happens, continue to the next candidate
            g_gain = g_GR - g_current_GR
            f_gain = f_GR - f_current_GR
            # These cases can occur due to the stochasticity of simulation. Avoid these
            if g_gain <= 0:
                g_gain = epsilon
            if f_gain <= 0:
                f_gain = epsilon
            # if g_gain <= 0 or f_gain <= 0:
                # continue

            f_g_gain_ratio = f_gain / g_gain

            if f_g_gain_ratio < gain_ratio_best:
                gain_ratio_best = f_g_gain_ratio
                S_best = S_candidate
                f_g_ratio_best = f_GR / g_GR

                # NOTE: keep track of these for debugging purpose
                g_best = g_GR
                f_best = f_GR
                g_gain_best = g_gain
                f_gain_best = f_gain
                # probability_array_best = probability_array
                print("Updated S_best: {}, gain_ratio_best: {}, g_best: {}".format(S_best, f_g_gain_ratio, g_best))

        # NOTE: None of the seeds may have f_gain and g_gain > 0. If this is the case, break the while loop
        if S_best == None:
            break

        # Line 6
        S.add(S_best)
        S_list.append(copy.deepcopy(S))

        print("S_list: {}".format(S_list))
        f_g_ratio_best_list.append(f_g_ratio_best)

        # NOTE: keep track of these for debugging purpose
        g_best_list.append(g_best)
        f_best_list.append(f_best)
        # g_gain_best_list.append(g_gain_best)
        # f_gain_best_list.append(f_gain_best)

        # Line 7 - TODO This leads to computing g twice. Try to incorporate this in the line 5
        S_candidate_list.remove(S_best)
        # print("Candidate list: {}".format(S_candidate_list))
        S_candidate_list = update_S_candidate_list_GR(S, g_best, S_candidate_list, simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)
        # print("Updated   list: {}".format(S_candidate_list))

    # TODO Line 10
    # gain_ratio_best_array = np.array(gain_ratio_best_list)
    print("S_list: {}".format(S_list))
    S_solution = get_solution_from_f_g_ratio_best(list_of_sets_of_P, n_t_for_eval, f_g_ratio_best_list, S_list, g_best_list, flag_g_constraint)

    # print(gain_ratio_best_array)

    print("g_best_list:      {}".format([round(val, 4) for val in g_best_list]))
    print("f_best_list:      {}".format([round(val, 4) for val in f_best_list]))
    # print("g_gain_best_list: {}".format([round(val, 4) for val in g_gain_best_list]))
    # print("f_gain_best_list: {}".format([round(val, 4) for val in f_gain_best_list]))

    seeds_greedy_solution = prep_seeds_2d_bool_array(simul, S_solution)

    intermediary_results_dict = {
            "S_list": S_list,
            "g_best_list": g_best_list,
            "f_best_list": f_best_list,
            "f_g_ratio_best_list": f_g_ratio_best_list,
            }

    return seeds_greedy_solution, intermediary_results_dict

def get_solution_from_f_g_ratio_best(list_of_sets_of_P, n_t_for_eval, f_g_ratio_best_list, S_list, g_best_list, flag_g_constraint):
    f_g_ratio_best_array = np.array(f_g_ratio_best_list)

    if flag_g_constraint:
        n_P_for_eval = sum([len(P_t) for P_t in list_of_sets_of_P[-n_t_for_eval:]])
        constraint_on_g = 0.5 * n_P_for_eval
        print("constraint_on_g: {}".format(constraint_on_g))
        min_idx, min_f_g_ratio, S_solution = 0, math.inf, None
        for idx, (f_g_ratio_at_idx_t, g_at_idx_t, S_at_idx_t) in enumerate(zip(f_g_ratio_best_list, g_best_list, S_list)):
            if f_g_ratio_at_idx_t < min_f_g_ratio and g_at_idx_t >= constraint_on_g:
                min_idx = idx
                min_f_g_ratio = f_g_ratio_at_idx_t
                S_solution = S_at_idx_t
        if S_solution == None: # If there's no seed set that satisfies the constraint on g, choose the seed at idx it maximizes g
            print("If the constraint on g cannot be met,")
            print("return the seed that yields the maximum infection in g")
            g_best_array = np.array(g_best_list)
            g_max_idx = np.argmax(g_best_array)
            S_solution = S_list[g_max_idx]
        # print("f_g_ratio_solution: {}".format(min_f_g_ratio))
    else:
        min_idx = np.argmin(f_g_ratio_best_array)
        S_solution = S_list[min_idx]
        print("f_g_ratio_solution: {}".format(np.min(f_g_ratio_best_array)))

    print("flag_g_constraint: {}".format(flag_g_constraint))
    print("f_g_ratio_best_array: {}".format(f_g_ratio_best_array))
    print("solution_idx: {}".format(min_idx))
    print("S_list: {}".format(S_list))
    print("S_solution: {}".format(S_solution))

    return S_solution

###############################################################################
# Step0: Initialize X_large - While doing so, compute over estimates on g
# Run simulations, setting each candidate seed node as source node.
# Keep track of the ratio, and its corresponding node, gain on f, and gain on g
# We'll select top k nodes w/ small ratio
# Helper function for lazy_greedy_ratio
def get_X_large(simul, S_candidate_list, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f):

    T = simul.n_timesteps -1 # T is the index of the last timestep
    len_P_T = len(list_of_sets_of_P[T])

    h_for_X_large = []
    heapify(h_for_X_large)
    g_gain_to_X_i_dict = dict() # Initially, X_i is {}.
    S_candidate_to_remove_list = []

    print("Compute X_large and upper bound on g to empty")
    for S_candidate in tqdm(S_candidate_list):
        seeds_2d_bool = prep_seeds_2d_bool_array(simul, [S_candidate])
        simul.set_seeds(seeds_2d_bool)

        g_gain_to_X_i, f_gain_to_X_i, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

        # These cases can occur due to the stochasticity of simulation. Avoid these
        if g_gain_to_X_i <= 0 or f_gain_to_X_i <= 0:
            # remove seeds from S_candidate_list that lead to these outcome 
            S_candidate_to_remove_list.append(S_candidate)
            continue

        f_g_gain_ratio = f_gain_to_X_i / g_gain_to_X_i
        heappush(h_for_X_large, (f_g_gain_ratio, S_candidate))
        g_gain_to_X_i_dict[S_candidate] = g_gain_to_X_i

    for S_candidate in S_candidate_to_remove_list:
        S_candidate_list.remove(S_candidate)

    X_large = []
    for i in range(len_P_T):
        f_g_gain_ratio, v = heappop(h_for_X_large)
        X_large.append(v)

    return X_large, g_gain_to_X_i_dict, S_candidate_list

# S holds the current solution
def update_X_large(simul, X_large, S, S_candidate_list, \
        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f):

    # Set k as 2 times the prev X_large.
    k = 2 * len(X_large)

    # Recreate this dict
    g_gain_to_X_i_dict = dict() # Initially, X_i is {}.

    # remove the seeds in the solution from the S_candidate_list
    for S_in_solution in S:
        if S_in_solution in S_candidate_list:
            S_candidate_list.remove(S_in_solution)

    h_for_X_large = []
    heapify(h_for_X_large)
    S_candidate_to_remove_list = []

    for S_candidate in tqdm(S_candidate_list):
        seeds_2d_bool = prep_seeds_2d_bool_array(simul, list(S) + [S_candidate])
        simul.set_seeds(seeds_2d_bool)

        g_gain_to_X_i, f_gain_to_X_i, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

        # These cases can occur due to the stochasticity of simulation. Avoid these
        if g_gain_to_X_i <= 0 or f_gain_to_X_i <= 0:
            # remove seeds from S_candidate_list that lead to these outcome 
            S_candidate_to_remove_list.append(S_candidate)
            continue

        f_g_gain_ratio = f_gain_to_X_i / g_gain_to_X_i
        heappush(h_for_X_large, (f_g_gain_ratio, S_candidate))
        g_gain_to_X_i_dict[S_candidate] = g_gain_to_X_i

    for S_candidate in S_candidate_to_remove_list:
        S_candidate_list.remove(S_candidate)

    X_large_new = []
    for i in range(k):
        # In case, if the heap is empty, break.
        if len(h_for_X_large) == 0:
            break
        f_g_gain_ratio, v = heappop(h_for_X_large)
        X_large_new.append(v)

    return X_large_new, g_gain_to_X_i_dict, S_candidate_list


# Step1: Compute under estimates on f
def get_min_heap_of_underestimates(simul, X_large, S_candidate_list, g_gain_to_X_i_dict,\
                                    list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f):
    seeds_2d_bool = prep_seeds_2d_bool_array(simul, X_large)
    simul.set_seeds(seeds_2d_bool)

    _, f_current_GR, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

    print("Compute min heap")
    # NOTE: Keep track of the ratio, and its corresponding node, gain on f, and gain on g
    h = []
    heapify(h)
    f_gain_to_X_large_dict = dict()
    f_to_X_large_dict = dict()

    for S_candidate in tqdm(S_candidate_list):
        seeds_2d_bool = prep_seeds_2d_bool_array(simul, X_large + [S_candidate])
        simul.set_seeds(seeds_2d_bool)

        _, f_GR, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

        g_gain = g_gain_to_X_i_dict[S_candidate]
        f_gain = f_GR - f_current_GR
        # These cases can occur due to the stochasticity of simulation. Avoid these
        if g_gain <= 0 or f_gain <= 0:
            continue

        f_gain_to_X_large_dict[S_candidate] = f_gain
        f_to_X_large_dict[S_candidate] = f_GR

        f_g_gain_ratio = f_gain / g_gain

        heappush(h, (f_g_gain_ratio, S_candidate))

    return h, f_gain_to_X_large_dict, f_to_X_large_dict


# Lazy implementation of greedy ratio
def lazy_greedy_ratio(simul, list_of_people_idx_arrays, number_of_seeds_over_time, \
        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, obs_state, array_of_penalty_on_f,
        flag_g_constraint):


    # Prepare list of candidate seeds as tuples
    S_candidate_list = get_S_candidate_list(list_of_people_idx_arrays, number_of_seeds_over_time)

    # Step0: Initialize X_large - While doing so, compute over estimates on g
    X_large, g_gain_to_X_i_dict, S_candidate_list = get_X_large(simul, S_candidate_list, \
            list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

    # Step1: Compute under estimates on f
    h, f_gain_to_X_large_dict, f_to_X_large_dict = get_min_heap_of_underestimates(simul, X_large, S_candidate_list, g_gain_to_X_i_dict, \
                                    list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)
    
    ###############################################################################
    # Step2: Lazy Greedy Ratio
    S_list = []
    # gain_ratio_best_list = []
    g_best_list = []
    f_best_list = []
    f_g_ratio_best_list = []
    S = set()

    while len(h) > 0: # repeat while heap is non-empty
        # Compute g and f on the selected seeds till now
        seeds_2d_bool = prep_seeds_2d_bool_array(simul, S)
        simul.set_seeds(seeds_2d_bool)

        if len(S) == 0: # if no seeds, then no infection occurs.
            g_current_GR = 0
        else:
            g_current_GR, _, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

        prev_len_S_list = len(S_list)
        for j in range(len(h)):
            _ , S_candidate = heappop(h)
            if len(h) == 0: # If this is the last element, break
                break
            S_temp = {S_candidate}
            S_temp.update(S)

            seeds_2d_bool = prep_seeds_2d_bool_array(simul, S_temp)
            simul.set_seeds(seeds_2d_bool)

            g_GR, _, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)
            g_gain = g_GR - g_current_GR
            if g_gain <= 0:
                continue

            f_g_gain_ratio = f_gain_to_X_large_dict[S_candidate] / g_gain

            lower_bound_gain_ratio, _ = h[0]

            if f_g_gain_ratio < lower_bound_gain_ratio:
                # Add the S_candidate to the solution
                S.add(S_candidate)
                S_list.append(copy.deepcopy(S))
                # gain_ratio_best_list.append(f_g_gain_ratio)
                f_g_ratio_best_list.append(f_to_X_large_dict[S_candidate] / g_GR)
                f_best_list.append(f_to_X_large_dict[S_candidate])
                g_best_list.append(g_GR)
                break
            else:
                heappush(h, (f_g_gain_ratio, S_candidate))

        after_len_S_list = len(S_list)
        # In case no seed node was added to the solution, break the while loop.
        if prev_len_S_list == after_len_S_list:
            break

        # After the seed is added to S, Update bounds if the seed is in X_large
        # Note: at this point, S_candidate is the seed just added to the solution
        if S_candidate not in X_large:
            print("Recompute X_large")
            X_large, g_gain_to_X_i_dict, S_candidate_list = update_X_large(simul, X_large, S, S_candidate_list, \
                        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)
            print("Recompute estimates, construct a new heap")
            h, f_gain_to_X_large_dict, f_to_X_large_dict = get_min_heap_of_underestimates(simul, X_large, S_candidate_list, g_gain_to_X_i_dict, \
                        list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

    # TODO Line 10
    S_solution = get_solution_from_f_g_ratio_best(list_of_sets_of_P, n_t_for_eval, f_g_ratio_best_list, S_list, g_best_list, flag_g_constraint)

    print("g_best_list:      {}".format([round(val, 4) for val in g_best_list]))
    print("f_best_list:      {}".format([round(val, 4) for val in f_best_list]))
    # f_g_ratio_best_array = np.array(f_g_ratio_best_list)
    # idx_solution = np.argmin(f_g_ratio_best_array)
    # S_solution = S_list[idx_solution]
    # print(gain_ratio_best_array)

    seeds_greedy_solution = prep_seeds_2d_bool_array(simul, S_solution)

    intermediary_results_dict = {
            "S_list": S_list,
            "g_best_list": g_best_list,
            "f_best_list": f_best_list,
            "f_g_ratio_best_list": f_g_ratio_best_list,
            }

    return seeds_greedy_solution, intermediary_results_dict

def update_S_candidate_list_GR(S, g_best, S_candidate_list, simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval,array_of_penalty_on_f):
    epsilon = pow(10, -6)
    # seeds_2d_bool = np.zeros((simul.n_timesteps, simul.number_of_nodes)).astype(bool)
    S_candidate_list_updated = []
    for S_candidate in S_candidate_list:
        S_temp = {S_candidate}
        S_temp.update(S)

        seeds_2d_bool = prep_seeds_2d_bool_array(simul, S_temp)
        simul.set_seeds(seeds_2d_bool)

        g_GR, f_GR, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)

        if g_GR <= 0:
            g_GR = epsilon

        # g_GR, _, _ = g_f_GR(simul, list_of_sets_of_P, list_of_sets_of_N, n_t_for_eval, array_of_penalty_on_f)
        # gain of adding a seed to the current solution
        if g_GR - g_best > 0:
            S_candidate_list_updated.append(S_candidate)
        
    return S_candidate_list_updated

