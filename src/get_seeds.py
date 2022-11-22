import numpy as np
# ----------------------------------------------
# SEEDS
def get_seeds_over_time(list_of_people_idx_arrays, number_of_seeds_over_time):
    set_of_seeds_so_far = set()
    list_of_seed_idx_arrays = []
    for t, k in enumerate(number_of_seeds_over_time):
        # Only add the S_temp if any of the seeds in S_temp was not chosen before. We need this while loop.
        while True:
            S_temp = np.random.choice(a=list_of_people_idx_arrays[t], size=k, replace=False)
            set_S_temp = set(S_temp)
            if len(set_S_temp.intersection(set_of_seeds_so_far)) == 0:
                list_of_seed_idx_arrays.append(S_temp)
                set_of_seeds_so_far = set_of_seeds_so_far.union(set_S_temp)
                break # exits the while loop, and proceed to the next for loop iteration
    return list_of_seed_idx_arrays
