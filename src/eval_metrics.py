import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

def compute_loss_per_timestep(simul, seeds_array, obs_state):

    simul.set_seeds(seeds_array)
    simul.simulate()
    probability_array = simul.probability_array
    # infection_array = simul.infection_array

    loss_1_array = np.zeros((simul.n_timesteps)).astype(np.float32)
    loss_total_array = np.zeros((simul.n_timesteps)).astype(np.float32)
    for t in range(simul.n_timesteps):
        # I is the probability array 
        I_prob = np.mean(probability_array[:, t, :], axis=0)
        I1_inf = obs_state[t].astype(np.float32)

        loss_1 = compute_loss("loss_1", I1_inf, I_prob)
        loss_total = compute_loss("loss_total", I1_inf, I_prob)

        loss_1_array[t] = loss_1
        loss_total_array[t] = loss_total
        # print("Ground truth loss. loss_total: {:.3f}, loss_1 = {:.3f}".format(loss_total, loss_1))
    return loss_1_array, loss_total_array#, probability_array

def compute_loss(loss_type, I1_inf, I_prob):
    if loss_type == "loss_1":
        idx_True = I1_inf.nonzero()[0]
        loss = np.sum(I1_inf[idx_True].astype(float) - I_prob[idx_True])
    elif loss_type == "loss_total":
        loss = np.sum(np.abs(I1_inf - I_prob))
    return loss

# summation of the infection probabilities
# I1_inf: ground truth infection
def compute_objective_value(I1_inf, I_prob):
    idx_True = I1_inf.nonzero()[0]
    return np.sum(I_prob[idx_True])

# -----------------------------------------------------------------------------------------------------------
# y_pred is a binary vector that indicates infection state of one replicate of a simulation of people
# y_true is the binary vector that indicates observed GT infection state of people
# def evaluate_from_multiple_timesteps(list_of_people_idx_arrays, list_of_sets_of_P, list_of_sets_of_N, infection_array, n_t_for_eval):
def evaluate_from_multiple_timesteps(obs_state, infection_array, list_of_people_idx_arrays, n_t_for_eval):

    n_rep, n_timesteps, _ = infection_array.shape
    # Initialize arrays
    tn_array = np.zeros((n_rep, n_t_for_eval))
    fp_array = np.zeros((n_rep, n_t_for_eval))
    fn_array = np.zeros((n_rep, n_t_for_eval))
    tp_array = np.zeros((n_rep, n_t_for_eval))
    # f1_array = np.zeros((n_rep, n_t_for_eval))
    # mcc_array = np.zeros((n_rep, n_t_for_eval))

    t_start = n_timesteps - n_t_for_eval # idx of the starting timestep
    t_end = n_timesteps # idx of the ending timestep + 1

    

    for t in range(t_start, t_end):
        people_idx_array_t = list_of_people_idx_arrays[t]

        y_true = obs_state[t, people_idx_array_t]

        for rep_idx in range(n_rep):

            y_pred = infection_array[rep_idx, t, people_idx_array_t]

            # For each replicate get these scores
            tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[False, True]).ravel()
            # f1 = f1_score(y_true, y_pred)
            # mcc = matthews_corrcoef(y_true, y_pred)

            tn_array[rep_idx, t - t_start] = tn
            fp_array[rep_idx, t - t_start] = fp
            fn_array[rep_idx, t - t_start] = fn
            tp_array[rep_idx, t - t_start] = tp
            # f1_array[rep_idx, t - t_start] = f1
            # mcc_array[rep_idx, t - t_start] = mcc

    tn = np.sum(tn_array, axis=1)
    fp = np.sum(fp_array, axis=1)
    fn = np.sum(fn_array, axis=1)
    tp = np.sum(tp_array, axis=1)
    # f1 = tp / (tp + 0.5*(fp+fn))
    np.seterr(divide='ignore', invalid='ignore') # silence division by 0
    f1 = np.divide(tp, tp + 0.5*(fp+fn))
    mcc = get_MCC(tp, tn, fp, fn)

    # Take summation of the score over time to get one score per replicate
    # Then, take average over the scores on the replicates
    avg_tn = np.mean(tn)
    avg_fp = np.mean(fp)
    avg_fn = np.mean(fn)
    avg_tp = np.mean(tp)
    if np.isnan(f1).all():
        avg_f1 = np.nan
    else:
        avg_f1 = np.nanmean(f1) # nanmean ignores nan while taking mean
    if np.isnan(mcc).all():
        avg_mcc = np.nan
    else:
        avg_mcc = np.nanmean(mcc) # nanmean ignores nan while taking mean
    
    return avg_tn, avg_fp, avg_fn, avg_tp, avg_f1, avg_mcc

def get_P_hit_N_hit_over_time(list_of_sets_of_P, list_of_sets_of_N, probability_array):
    list_of_P_hit = []
    list_of_N_hit = []

    # NOTE: 2-d array (timestep, nodes)
    mean_probability_array = np.mean(probability_array, axis=0)

    for t, (P_t, N_t) in enumerate(zip(list_of_sets_of_P, list_of_sets_of_N)):
        P_hit, N_hit = get_P_hit_N_hit(P_t, N_t, mean_probability_array[t, :])
        list_of_P_hit.append(P_hit)
        list_of_N_hit.append(N_hit)

    return list_of_P_hit, list_of_N_hit

def get_P_hit_N_hit(P_t, N_t, mean_probability_array_at_t):
    P_hit = np.sum(mean_probability_array_at_t[list(P_t)])
    N_hit = np.sum(mean_probability_array_at_t[list(N_t)])
    return P_hit, N_hit

# -----------------------------------------------------------------------------------------------------------
def compute_scores_from_multiple_time_steps(list_of_sets_of_P, list_of_sets_of_N, list_of_P_hit, list_of_N_hit, n_t_for_eval):
    len_P, len_N, P_hit, N_hit = 0, 0, 0, 0
    for P_t, N_t, P_hit_t, N_hit_t in zip(list_of_sets_of_P[-n_t_for_eval:], list_of_sets_of_N[-n_t_for_eval:], list_of_P_hit[-n_t_for_eval:], list_of_N_hit[-n_t_for_eval:]):
        len_P += len(P_t)
        len_N += len(N_t)
        P_hit += P_hit_t
        N_hit += N_hit_t
    return compute_scores(len_P, len_N, P_hit, N_hit)

def compute_scores(len_P, len_N, P_hit, N_hit):
    TP = P_hit
    TN = len_N - N_hit
    FP = N_hit
    FN = len_P - P_hit
    F1 = TP / (TP + 0.5*(FP+FN))
    MCC = get_MCC(TP, TN, FP, FN)
    return TP, TN, FP, FN, F1, MCC

def get_MCC(TP, TN, FP, FN):
    numerator = (TP*TN) - (FP*FN)
    denominator = np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )
    np.seterr(divide='ignore', invalid='ignore') # silence division by 0
    return np.divide(numerator, denominator, dtype=float)
    # print(numerator.shape)
    # print(denominator.shape)
    # if denominator == 0:
        # MCC_score = float("nan")
    # else:
        # MCC_score = numerator / denominator
    # return MCC_score

