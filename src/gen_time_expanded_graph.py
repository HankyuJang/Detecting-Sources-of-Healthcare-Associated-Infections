"""
Author: -
Email: -
Last Modified: Jan 2022

Description: 

This script loads ground truth observations
and runs P1 - greedy

Usage

- 4 graphs for UIHC sampled

$ python gen_time_expanded_graph.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 1
$ python gen_time_expanded_graph.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response exponential -seeds_per_t 3
$ python gen_time_expanded_graph.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response linear -seeds_per_t 1
$ python gen_time_expanded_graph.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -sampled True -dose_response linear -seeds_per_t 3

- 2 graphs for UIHC whole

$ python gen_time_expanded_graph.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -dose_response exponential -seeds_per_t 1
$ python gen_time_expanded_graph.py -name UIHC_HCP_patient_room_withinHCPxPx -year 2011 -dose_response exponential -seeds_per_t 3

- 2 graphs for G_Carilion

$ python gen_time_expanded_graph.py -name G_Carilion -dose_response exponential -seeds_per_t 1
$ python gen_time_expanded_graph.py -name G_Carilion -dose_response exponential -seeds_per_t 3

"""

import argparse
from utils.load_network import *

import pickle

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
                        help= 'Quality of the ground truth simulation. best | median. Always use best')
    parser.add_argument('-seeds_per_t', '--seeds_per_t', type=int, default=1,
                        help= 'number of seeds per timestep')
    parser.add_argument('-n_t_seeds', '--n_t_seeds', type=int, default=2,
                        help= 'number of timesteps for seeds')
    parser.add_argument('-n_t_for_eval', '--n_t_for_eval', type=int, default=2,
                        help= 'number of timesteps for evaluation. If 2, evaluate on T and T-1')
    args = parser.parse_args()

    print("Load GT observations...\n")
    graph_name = get_graph_name(args)

    path = "../tables/final_exp/{}/seedspert{}_ntseeds{}_ntforeval{}/".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
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

    print("Load graphs")
    G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, _ = process_data_for_experiments(args, area_people, area_location, flag_increase_area)
    # -------------------------------------------
    # map the node id on Carilion graph from string to integer.
    # G_relabelled_over_time = []
    # if graph_name=="G_Carilion":
        # print("Relabel nodes")
        # G = G_over_time[0]
        # relabel_node_mapping = dict([(v, eval(v)) for v in G.nodes()])

        # for G in G_over_time:
            # G_relabelled = nx.relabel.relabel_nodes(G, relabel_node_mapping)
            # G_relabelled_over_time.append(G_relabelled)
        # G_over_time = G_relabelled_over_time
    # -------------------------------------------

    # -------------------------------------------
    # Step1. Add nodes
    t = 0


    G = G_over_time[t]
    if graph_name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]:
        patient_node_list = [v for v in G.nodes() if G.nodes[v]["type"]=="Patient"]
    else:
        patient_node_list = [v for v in G.nodes() if G.nodes[v]["type"]=="patient"]
    patient_node_set = set(patient_node_list)

    G_patient_node_list_with_attributes = []
    G_patient_cross_edgelist = []
    G_patient_temporal_edgelist = []

    # Add node attributes e.g., 29, 30
    timesteps_for_eval = [T-t for t in range(args.n_t_for_eval)]
    timesteps_for_seeds = [t for t in range(args.n_t_seeds)]

    print("\nAdd nodes")
    for t in range(n_timesteps):
        if t in timesteps_for_eval:
            for v in patient_node_list:
                if node_name_to_idx_mapping[v] in list_of_sets_of_P[t]:
                    print("Time: {}, Infected node: {}. idx: {}".format(t, v, node_name_to_idx_mapping[v]))
                    node_tuple = ((v, t), {"terminal": 1, "prob": 0, "S_candidate": 0})
                else:
                    node_tuple = ((v, t), {"terminal": 0, "prob": 0, "S_candidate": 0})
                G_patient_node_list_with_attributes.append(node_tuple)
        elif t in timesteps_for_seeds:
            for v in patient_node_list:
                if node_name_to_idx_mapping[v] in list_of_sets_of_V[t]:
                    print("Time: {}, S_candidate: {}. idx: {}".format(t, v, node_name_to_idx_mapping[v]))
                    node_tuple = ((v, t), {"terminal": 0, "prob": 0, "S_candidate": 1})
                else:
                    node_tuple = ((v, t), {"terminal": 0, "prob": 0, "S_candidate": 0})
                G_patient_node_list_with_attributes.append(node_tuple)
        else:
            for v in patient_node_list:
                node_tuple = ((v, t), {"terminal": 0, "prob": 0, "S_candidate": 0})
                G_patient_node_list_with_attributes.append(node_tuple)

    # Prep graph
    G_patient = nx.DiGraph()
    G_patient.add_nodes_from(G_patient_node_list_with_attributes)
    print(nx.info(G_patient))

    # -------------------------------------------
    # Step2. Add cross edges
    print("\nAdd cross edges")
    for t, G in enumerate(G_over_time[:-1]):
        for idx_v1, v1 in enumerate(patient_node_list[:-1]):
            v1_neighbors = set([v for v in G.neighbors(v1)])
            for idx_v2, v2 in enumerate(patient_node_list[idx_v1+1:]):
                # If v1 and v2 are neighboring node, add to edge list
                if v2 in v1_neighbors:
                    G_patient_cross_edgelist.append(((v1, t), (v2, t+1)))
                    G_patient_cross_edgelist.append(((v2, t), (v1, t+1)))
                    # print("case1")
                    continue

                # If v1 and v2 have common neighbors, add to edge list
                v2_neighbors = set([v for v in G.neighbors(v2)])
                if len(v1_neighbors.intersection(v2_neighbors)) > 0:
                    # print("case2")
                    G_patient_cross_edgelist.append(((v1, t), (v2, t+1)))
                    G_patient_cross_edgelist.append(((v2, t), (v1, t+1)))

    G_patient.add_edges_from(G_patient_cross_edgelist, weight=1)
    print(nx.info(G_patient))

    # -------------------------------------------
    # Step3. Add temporal edges
    print("\nAdd temporal edges")
    for t in range(n_timesteps-1):
        for v in patient_node_list:
            G_patient_temporal_edgelist.append(((v, t), (v, t+1)))

    G_patient.add_edges_from(G_patient_temporal_edgelist, weight=0)
    print(nx.info(G_patient))

    outpath = "../cult/graph/"
    outfile = "G_time_expanded_{}_seedspert{}_ntseeds{}_ntforeval{}.graphml".format(graph_name, args.seeds_per_t, args.n_t_seeds, args.n_t_for_eval)
    nx.write_graphml(G_patient, outpath+outfile, named_key_ids=True)

