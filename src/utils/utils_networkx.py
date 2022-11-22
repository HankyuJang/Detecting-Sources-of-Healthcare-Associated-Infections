"""
Author: Hankyu Jang
Email: hankyu-jang@uiowa.edu
Last Modified: Feb, 2020 

Description: helper functions for networkx package
"""

import networkx as nx
import numpy as np

def get_largest_connected_component(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)
    return G_giant

# For each node on consecutive graphs, get the max degree among that node over different graphs
# Return a numpy array, each index correspond to max degree of each node
# NOTE: Assumption that all graphs have same set of nodes.
def get_max_degree_per_node_over_time(G_list):
    n_graphs = len(G_list)
    num_nodes = G_list[0].number_of_nodes()
    degree_array = np.zeros((n_graphs, num_nodes)).astype(int)

    node_name_to_idx_mapping = dict([(node_name, node_idx) for node_idx, node_name in enumerate(G_list[0].nodes())])

    for G_idx, G in enumerate(G_list):
        for node_name, degree in G.degree:
            # name to idx
            node_idx = node_name_to_idx_mapping[node_name]

            # NOTE: made modification in Feb3 for G_UVA
            if degree == 0:
                degree_array[G_idx, node_idx] = 1
            else:
                degree_array[G_idx, node_idx] = degree
    
    max_degree_array = np.max(degree_array, axis=0)

    return max_degree_array


