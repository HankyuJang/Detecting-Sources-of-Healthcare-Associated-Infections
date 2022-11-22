"""

Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Jan 2022

Generate network statistics
"""

from utils.graph_statistics import *
# from utils.networkx_operations import *
import pandas as pd
from tqdm import tqdm

def get_largest_connected_component(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)
    return G_giant

def gen_df_statistics(G):
    G_statistics = generate_graph_statistics(G)
    d = {"{}".format(name): G_statistics}
    index_list = ["n", "m", "k_mean", "k_max", "k_std", "cc", "c", "r", "n_giant", "m_giant"]
    column_list = ["{}".format(name)]
    df_statistics = pd.DataFrame(data=d, index=index_list, columns=column_list)
    return df_statistics

def gen_UVA_statistics(folder):
    day_list = [d for d in range(0, 31)]

    n_days = len(day_list)
    n_array = np.zeros((n_days)).astype(int)
    m_array = np.zeros((n_days)).astype(int)
    k_mean_array = np.zeros((n_days))
    k_max_array = np.zeros((n_days))
    k_std_array = np.zeros((n_days))
    cc_array = np.zeros((n_days))
    c_array= np.zeros((n_days))
    assortativity_array= np.zeros((n_days))
    n_giant_array = np.zeros((n_days)).astype(int)
    m_giant_array = np.zeros((n_days)).astype(int)

    for i, day in tqdm(enumerate(day_list)):
        G = nx.read_graphml("../{}/graph_day{}.graphml".format(folder, day))
        n, m, k_mean, k_max, k_std, cc, c, assortativity, n_giant, m_giant = generate_graph_statistics(G)
        n_array[i] = n
        m_array[i] = m
        k_mean_array[i] = k_mean 
        k_max_array[i] = k_max 
        k_std_array[i] = k_std 
        cc_array[i] = cc 
        c_array[i] = c
        assortativity_array[i] = assortativity
        n_giant_array[i] = n_giant 
        m_giant_array[i] = m_giant

    d_mean = [np.mean(n_array), np.mean(m_array), np.mean(k_mean_array), np.mean(k_max_array), np.mean(k_std_array), np.mean(cc_array), np.mean(c_array), np.mean(assortativity_array), np.mean(n_giant_array), np.mean(m_giant_array)]
    d_std = [np.std(n_array), np.std(m_array), np.std(k_std_array), np.std(k_max_array), np.std(k_std_array), np.std(cc_array), np.std(c_array), np.std(assortativity_array), np.std(n_giant_array), np.std(m_giant_array)]
    index_list = ["|V|", "|E|", "mean(degree)", "max(degree)", "std(degree)", "clustering coef", "num connected cpnt", "assortativity", "|V| (giant component)", "|E| (giant component)"]

    df_statistics = pd.DataFrame(
            data = {"mean": d_mean, "std": d_std},
            index = index_list
            )
    # return graph of the last day, and the statistics over 31 days.
    return G, df_statistics

def gen_edge_statistics_1day(G):
    # Number of HCP, Patient, room nodes
    HCP_nodes_bool = [G.nodes[v]["type"]=="HCP"  for v in G.nodes()]
    Patient_nodes_bool = [G.nodes[v]["type"]=="Patient"  for v in G.nodes()]
    room_nodes_bool = [G.nodes[v]["type"]=="room"  for v in G.nodes()]

    G_edges = [(v1, v2) for v1, v2 in G.edges()]
    HCP_HCP_edge_bool = np.zeros((len(G_edges))).astype(bool)
    Patient_Patient_edge_bool = np.zeros((len(G_edges))).astype(bool)
    HCP_room_edge_bool = np.zeros((len(G_edges))).astype(bool)
    Patient_room_edge_bool = np.zeros((len(G_edges))).astype(bool)
    HCP_Patient_edge_bool = np.zeros((len(G_edges))).astype(bool)
    
    for i, (v1, v2) in enumerate(G_edges):
        if (G.nodes[v1]["type"] == "HCP") and (G.nodes[v2]["type"] == "HCP"):
            HCP_HCP_edge_bool[i] = True
        elif (G.nodes[v1]["type"] == "Patient") and (G.nodes[v2]["type"] == "Patient"):
            Patient_Patient_edge_bool[i] = True
        elif (G.nodes[v1]["type"] == "HCP") and (G.nodes[v2]["type"] == "Patient"):
            HCP_Patient_edge_bool[i] = True
        elif (G.nodes[v1]["type"] == "Patient") and (G.nodes[v2]["type"] == "HCP"):
            HCP_Patient_edge_bool[i] = True
        elif (G.nodes[v1]["type"] == "HCP") and (G.nodes[v2]["type"] == "room"):
            HCP_room_edge_bool[i] = True
        elif (G.nodes[v1]["type"] == "room") and (G.nodes[v2]["type"] == "HCP"):
            HCP_room_edge_bool[i] = True
        elif (G.nodes[v1]["type"] == "Patient") and (G.nodes[v2]["type"] == "room"):
            Patient_room_edge_bool[i] = True
        elif (G.nodes[v1]["type"] == "room") and (G.nodes[v2]["type"] == "Patient"):
            Patient_room_edge_bool[i] = True

    d = [np.sum(HCP_nodes_bool), 
         np.sum(Patient_nodes_bool),
         np.sum(room_nodes_bool),
         np.sum(HCP_HCP_edge_bool),
         np.sum(Patient_Patient_edge_bool),
         np.sum(HCP_Patient_edge_bool),
         np.sum(HCP_room_edge_bool),
         np.sum(Patient_room_edge_bool)
        ]
    index_list = ["V(HCP)", "V(Patient)", "V(room)", "HCP-HCP", "Patient-Patient", "HCP-Patient", "HCP-room", "Patient-room"]

    df_nodes_edges = pd.DataFrame(
            data = d,
            columns = ["G_temporal (1day)"],
            index = index_list
            )
    return df_nodes_edges


if __name__ == "__main__":

    graph_name_list = ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]

    for graph_name in graph_name_list:
        print(graph_name)

        # The returned graph G is the graph of the last day
        G, df_statistics = gen_UVA_statistics("{}".format(graph_name))
        df_edges = gen_edge_statistics_1day(G)

        print(df_statistics)
        print(df_edges)

        # Save the graphs
        df_statistics.to_csv("../tables/G_UVA_statistics/{}_statistics.csv".format(graph_name), index=True)
        df_edges.to_csv("../tables/G_UVA_statistics/{}_edges.csv".format(graph_name), index=True)


