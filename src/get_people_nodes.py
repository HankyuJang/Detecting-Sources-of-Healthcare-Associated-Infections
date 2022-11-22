import networkx as nx
import numpy as np

def get_people_idx_array_over_time(G_over_time, node_name_to_idx_mapping, people_nodes_idx):
    list_of_people_idx_arrays = []
    for G in G_over_time:
        list_of_people_idx_arrays.append(get_people_array_in_day_t(G, node_name_to_idx_mapping, people_nodes_idx))
    return list_of_people_idx_arrays 

# Get people in day t w/ at least 1 neighbor
def get_people_array_in_day_t(G, node_name_to_idx_mapping, people_nodes_idx):
    day_t_people_list = []
    for node_name, degree in G.degree:
        if degree > 0:
            node_idx = node_name_to_idx_mapping[node_name]
            if node_idx in people_nodes_idx:
                day_t_people_list.append(node_idx)
    day_t_people_array = np.array(day_t_people_list)
    return day_t_people_array 

