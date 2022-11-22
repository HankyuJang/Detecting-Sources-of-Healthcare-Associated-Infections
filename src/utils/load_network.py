"""
Author: -
Email: -
Last Modified: July 2022

Description: Script for loading networks.

In the current version, we set HCPs as moving location nodes.
"""

from utils.utils_networkx import *

def unravel_GT_observaion_pickle(GT_output_dict):
    GT_observation_dict = GT_output_dict["GT_observation_dict"]
    # GT_evaluation_dict = GT_output_dict["GT_evaluation_dict"]

    n_timesteps = GT_observation_dict["n_timesteps"]
    n_replicates = GT_observation_dict["n_replicates"]
    area_people = GT_observation_dict["area_people"]
    area_location = GT_observation_dict["area_location"]
    T = GT_observation_dict["T"]
    # flag_increase_area = GT_observation_dict["flag_increase_area"]
    flag_increase_area = True # always set this as True for AAAI experiments
    number_of_seeds_over_time = GT_observation_dict["number_of_seeds_over_time"]
    k_total = GT_observation_dict["k_total"]
    node_name_to_idx_mapping = GT_observation_dict["node_name_to_idx_mapping"]
    node_idx_to_name_mapping = GT_observation_dict["node_idx_to_name_mapping"]
    list_of_people_idx_arrays = GT_observation_dict["list_of_people_idx_arrays"]
    list_of_sets_of_V = GT_observation_dict["list_of_sets_of_V"]
    seeds_array = GT_observation_dict["seeds_array"]
    obs_state = GT_observation_dict["obs_state"]
    I1 = GT_observation_dict["I1"]
    # MCC_array = GT_observation_dict["MCC_array"]
    MCC_array = None # We don't have this for AAAI experiments
    list_of_sets_of_P = GT_observation_dict["list_of_sets_of_P"]
    list_of_sets_of_N = GT_observation_dict["list_of_sets_of_N"]

    return n_timesteps, n_replicates, area_people, area_location, T, flag_increase_area, number_of_seeds_over_time, k_total,\
            node_name_to_idx_mapping, node_idx_to_name_mapping, list_of_people_idx_arrays, list_of_sets_of_V, seeds_array, obs_state,\
            I1, MCC_array, list_of_sets_of_P, list_of_sets_of_N

def get_graph_name(args):
    name = args.name
    if name == "UIHC_HCP_patient_room_withinHCPxPx":
        year = args.year
        sampled = args.sampled
        if sampled:
            name = "{}_{}_sampled".format(name, year)
        else:
            name = "{}_{}".format(name, year)
    return name

def process_data_for_experiments(args, area_people, area_location, flag_increase_area, flag_casestudy=False):
    name = args.name

    if name == "Karate_temporal":
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_karate_temporal_network(area_people, area_location, flag_increase_area)
    elif name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4", "G_Carilion", "G_Carilion_v3"]:
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_other_networks(name, area_people, area_location, flag_increase_area)
    elif name == "UIHC_Jan2010_patient_room_temporal":
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UIHC_Jan2010_patient_room_temporal_network(area_people, area_location, flag_increase_area)
    elif name == "UIHC_HCP_patient_room_withinHCPxPx":
        year = args.year
        sampled = args.sampled
        if sampled:
            name = "{}_{}_sampled".format(name, year)
        else:
            name = "{}_{}".format(name, year)
        # if year = 2011 # Use non-overlap data.
        # if sampled = True # Use the subgraph. Sampled based on the unit with the most number of CDI cases.
        G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array = load_UIHC_HCP_patient_room_temporal_network(year, sampled, area_people, area_location, flag_increase_area, flag_casestudy)

    return G_over_time, people_nodes, people_nodes_idx, location_nodes_idx, area_array, name

# graphs are saved in `folder` with filenames day{}.graphml for day 0, ..., 30
# nodal attribute type in patient, HCP, room
# Only patients can get infected. -> only set patients as people nodes
# patients and HCP have area of area_people.
def load_other_networks(name, area_people, area_location, flag_increase_area):

    # if name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]:
    #     folder = name
    if name == "G_UVA_v2":
        folder = "G_UVA/JANUARY_2020_Cardiology"
    elif name == "G_UVA_v4":
        folder = "G_UVA/MARCH_2011_Cardiology"
    elif name == "G_Carilion":
        # folder = "../G_Carilion/graph_v2" # one directory parent of the repo
        folder = "../G_Carilion/graph_w_edge_attributes_v2" # one directory parent of the repo
    elif name == "G_Carilion_v3":
        folder = "../G_Carilion/graph_v3" # one directory parent of the repo

    G_list = []

    day = 0

    if name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]:
        # G = nx.read_graphml("../{}/graph_day{}.graphml".format(folder, day))
        G = nx.read_graphml("../{}/day{}.graphml".format(folder, day))
    elif name == "G_Carilion" or name == "G_Carilion_v3":
        G = nx.read_graphml("../{}/day{}.graphml".format(folder, day))

    G_list.append(G)

    # NOTE: people_nodes are those that can get infected and shed pathogen. Only patients
    if name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]:
        people_nodes = np.array([G.nodes[v]["type"] == "Patient" for v in G.nodes()])
    elif name == "G_Carilion" or name == "G_Carilion_v3":
        people_nodes = np.array([G.nodes[v]["type"] == "patient" for v in G.nodes()])

    people_nodes_idx = np.nonzero(people_nodes)[0]
    location_nodes_idx = np.where(people_nodes == False)[0]

    # NOTE: HCP nodes are those that do not get infected, but with a smaller surface area. E.g., HCP
    # patients and HCPs have same area of area_people.
    if name in ["G_UVA", "G_UVA_v2", "G_UVA_v3", "G_UVA_v4"]:
        HCP_nodes = np.array([G.nodes[v]["type"] == "HCP" for v in G.nodes()])
    elif name == "G_Carilion" or name == "G_Carilion_v3":
        HCP_nodes = np.array([G.nodes[v]["type"] in ["doc", "nurse", "sitter", "transporter", "food_service", "EV", "educator", "technician", "fomite"] for v in G.nodes()])

    n_HCP = HCP_nodes.shape[0]
    if n_HCP != 0:
        HCP_nodes_idx = np.nonzero(HCP_nodes)[0]

    area_array = np.ones((len(G))) 
    area_array[location_nodes_idx] = area_location # Initially, HCPs are included in the location_nodes_idx.
    area_array[people_nodes_idx] = area_people
    if n_HCP != 0:
        area_array[HCP_nodes_idx] = area_people # Modify area of HCPs here.

    for day in range(1, 31):
        G = nx.read_graphml("../{}/day{}.graphml".format(folder, day))
        G_list.append(G)

    if flag_increase_area:
        max_degree_array = get_max_degree_per_node_over_time(G_list)
        # Increate area according to max degree per node
        area_array *= max_degree_array

    return G_list, people_nodes, people_nodes_idx, location_nodes_idx, area_array
# Modify this function to load UVA temporal network
# TODO: Currently loads Karate network.
# def load_UVA_temporal_network(area_people, area_location, flag_increase_area):
    # G_list = []

    # G = nx.karate_club_graph()
    # G_list.append(G)
    # # people node
    # people_nodes = np.ones((len(G))).astype(bool)
    # people_nodes[20:] = False
    # people_nodes_idx = np.nonzero(people_nodes)[0]
    # location_nodes_idx = np.where(people_nodes == False)[0]
    # area_array = np.ones((len(G))) 
    # area_array[people_nodes_idx] = area_people
    # area_array[location_nodes_idx] = area_location

    # for day in range(1, 31):
        # G = nx.karate_club_graph()
        # G_list.append(G)

    # if flag_increase_area:
        # max_degree_array = get_max_degree_per_node_over_time(G_list)
        # # Increate area according to max degree per node
        # area_array *= max_degree_array
    
    # return G_list, people_nodes, people_nodes_idx, location_nodes_idx, area_array

#----------------------------------------------------------
# UIHC. Nodes are HCP, patient, room.
# year = 2007: All the interactions are based on Jan 2007
# year = 2011: HCP logins data is from 2007, but all the rest of the interactions are based on Jan 2011
# NOTE
# HCP nodes acts as moving locations. They do not get infected.
# -> only patient nodes are 'people_nodes' that have a change to getting infected
def load_UIHC_HCP_patient_room_temporal_network(year, sampled, area_people, area_location, flag_increase_area, flag_casestudy):
    G_list = []

    day = 0
    if sampled and flag_casestudy: # use the graph processed previously. Case study experiments use this data (w/o edge attributes)
        G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx/UIHC_Jan{}_day{}_sampled.graphml".format(year, day))
    elif sampled:
        # G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx/UIHC_Jan{}_day{}_sampled.graphml".format(year, day))
        G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx_edge_attributes/UIHC_Jan{}_day{}_sampled.graphml".format(year, day))
    else:
        # G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx/UIHC_Jan{}_day{}.graphml".format(year, day))
        G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx_edge_attributes/UIHC_Jan{}_day{}.graphml".format(year, day))

    G_list.append(G)

    people_nodes = np.array([G.nodes[v]["type"] == "patient" for v in G.nodes()])
    people_nodes_idx = np.nonzero(people_nodes)[0]
    location_nodes_idx = np.where(people_nodes == False)[0]

    # patients and HCPs have same area of area_people.
    HCP_nodes = np.array([G.nodes[v]["type"] == "HCP" for v in G.nodes()])
    HCP_nodes_idx = np.nonzero(HCP_nodes)[0]

    area_array = np.ones((len(G))) 
    area_array[location_nodes_idx] = area_location # Initially, HCPs are included in the location_nodes_idx.
    area_array[people_nodes_idx] = area_people
    area_array[HCP_nodes_idx] = area_people # Modify area of HCPs here.

    for day in range(1, 31):
        if sampled and flag_casestudy: # use the graph processed previously. Case study experiments use this data (w/o edge attributes)
            G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx/UIHC_Jan{}_day{}_sampled.graphml".format(year, day))
        elif sampled:
            # G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx/UIHC_Jan{}_day{}_sampled.graphml".format(year, day))
            G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx_edge_attributes/UIHC_Jan{}_day{}_sampled.graphml".format(year, day))
        else:
            # G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx/UIHC_Jan{}_day{}.graphml".format(year, day))
            G = nx.read_graphml("../../G_UIHC/graph/UIHC_HCP_patient_room_withinHCPxPx_edge_attributes/UIHC_Jan{}_day{}.graphml".format(year, day))
        G_list.append(G)

    if flag_increase_area:
        max_degree_array = get_max_degree_per_node_over_time(G_list)
        # Increate area according to max degree per node
        area_array *= max_degree_array

    return G_list, people_nodes, people_nodes_idx, location_nodes_idx, area_array


#----------------------------------------------------------
# Karate
def load_karate_network(area_people, area_location, flag_increase_area):
    G = nx.karate_club_graph()
    # people node
    people_nodes = np.ones((len(G))).astype(bool)
    people_nodes[20:] = False
    people_nodes_idx = np.nonzero(people_nodes)[0]
    location_nodes_idx = np.where(people_nodes == False)[0]
    area_array = np.ones((len(G))) 
    area_array[people_nodes_idx] = area_people
    area_array[location_nodes_idx] = area_location

    if flag_increase_area:
        max_degree_array = get_max_degree_per_node_over_time(G_list)
        # Increate area according to max degree per node
        area_array *= max_degree_array

    return G, people_nodes, people_nodes_idx, location_nodes_idx, area_array

def load_karate_temporal_network(area_people, area_location, flag_increase_area):
    G_list = []

    G = nx.karate_club_graph()
    G_list.append(G)
    # people node
    people_nodes = np.ones((len(G))).astype(bool)
    people_nodes[20:] = False
    people_nodes_idx = np.nonzero(people_nodes)[0]
    location_nodes_idx = np.where(people_nodes == False)[0]
    area_array = np.ones((len(G))) 
    area_array[people_nodes_idx] = area_people
    area_array[location_nodes_idx] = area_location

    for day in range(1, 31):
        G = nx.karate_club_graph()
        G_list.append(G)

    if flag_increase_area:
        max_degree_array = get_max_degree_per_node_over_time(G_list)
        # Increate area according to max degree per node
        area_array *= max_degree_array
    
    return G_list, people_nodes, people_nodes_idx, location_nodes_idx, area_array

#----------------------------------------------------------
# UIHC. nodes are patients and rooms
def load_UIHC_Jan2010_patient_room_network(area_people, area_location, flag_increase_area):
    name = "UIHC_Jan2010_patient_room"
    G = nx.read_graphml("../../G_UIHC/graph/{}.graphml".format(name))
    G_giant = get_largest_connected_component(G)

    people_nodes = np.array([G_giant.nodes[v]["type"] == "patient" for v in G_giant.nodes()])
    people_nodes_idx = np.nonzero(people_nodes)[0]
    location_nodes_idx = np.where(people_nodes == False)[0]
    area_array = np.ones((len(G_giant))) 
    area_array[people_nodes_idx] = area_people
    area_array[location_nodes_idx] = area_location

    if flag_increase_area:
        max_degree_array = get_max_degree_per_node_over_time(G_list)
        # Increate area according to max degree per node
        area_array *= max_degree_array

    return G_giant, people_nodes, people_nodes_idx, location_nodes_idx, area_array

def load_UIHC_Jan2010_patient_room_temporal_network(area_people, area_location, flag_increase_area):
    G_list = []

    day = 0
    G = nx.read_graphml("../../G_UIHC/graph/UIHC_Jan2010_patient_room_temporal/day{}.graphml".format(day))
    G_list.append(G)

    people_nodes = np.array([G.nodes[v]["type"] == "patient" for v in G.nodes()])
    people_nodes_idx = np.nonzero(people_nodes)[0]
    location_nodes_idx = np.where(people_nodes == False)[0]
    area_array = np.ones((len(G))) 
    area_array[people_nodes_idx] = area_people
    area_array[location_nodes_idx] = area_location

    for day in range(1, 31):
        G = nx.read_graphml("../../G_UIHC/graph/UIHC_Jan2010_patient_room_temporal/day{}.graphml".format(day))
        G_list.append(G)

    if flag_increase_area:
        max_degree_array = get_max_degree_per_node_over_time(G_list)
        # Increate area according to max degree per node
        area_array *= max_degree_array

    return G_list, people_nodes, people_nodes_idx, location_nodes_idx, area_array
#----------------------------------------------------------
