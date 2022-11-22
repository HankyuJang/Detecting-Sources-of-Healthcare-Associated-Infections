"""
Simulator. Load sharing model on temporal graph. 
NOTE (July, 2022): simulates by second.
NOTE Keep track of incoming loads

Author: -
Email: -
Last Modified: July, 2022 

Description: Simulator for the load sharing model

NOTE: minor modification - do not keep track of adj matrices (these grow O(n^2))

G: graph
seeds: 2-d boolean array denoting seeds at timestep t. Dim 0: timestep, Dim 1: nodes of size(G)
people_nodes: 1-d boolean array denoting if a node is a person. True if the node is a person. (can get infected). size(G)
area_array: 1-d array. area of each node in graph. size(G)
contact_area: area of a contact
n_timesteps: number of timesteps
rho: transfer efficiency
d: decay rate
q: shedding rate. amount of pathogen an infection agent sheds on its surface

infection_threhsold: if a patient was infected for at least this threshold of time during the day, mark as infected

NOTE: constraints on the values of parameters

rho: (0, 1)
d: (0, 1)
pi (0, 1)
contact area < min(area_people, area_location)
Also, make sure that (load dieoff amount) + (load shared to others) <= 1, that is
d + rho * (contact area / min(area_people, area_location)) <= 1

"""
# Simulation class for the load sharing model
import numpy as np
import networkx as nx
from tqdm import tqdm

class Simulation:
    def __init__(self,
            G_over_time,
            seeds,
            people_nodes,
            area_array,
            contact_area,
            n_timesteps,
            rho,
            d,
            q,
            pi,
            dose_response,
            n_replicates = 10,
            infection_threhsold=0.5):
        # input parameters
        # self.G = G
        self.seeds = seeds
        self.people_nodes = people_nodes
        self.area_array = area_array
        self.contact_area = contact_area
        self.n_timesteps = n_timesteps
        self.rho = rho
        self.d = d
        self.q = q
        self.pi = pi
        self.dose_response = dose_response
        # self.adj = nx.to_numpy_array(G)# adj is 2d numpy array. adjacency.
        # self.adj_over_time = self.gen_adj_over_time(G_over_time)
        # self.number_of_nodes = len(G)
        self.number_of_nodes = len(G_over_time[0])
        # For temporal graph, compute BplusD for each timestep
        # self.BplusD_over_time = self.constructBplusD_over_time(G_over_time)
        self.n_replicates = n_replicates
        self.get_nodename_to_idx_mapping(G_over_time[0])
        self.G_over_time = G_over_time
        self.incoming_load_track_dict = self.initialize_load_track_dict(G_over_time[0])
        self.infection_threshold = infection_threhsold
    
    def initialize_load_track_dict(self, G):
        incoming_load_track_dict = dict()
        for v in G.nodes():
            incoming_load_track_dict[v] = dict()
        return incoming_load_track_dict

    def update_load_track_dict(self, src, dst, load):
        eps=pow(10, -20)
        if load < eps: # in case there's no incoming load do nothing
            return
        if src in self.incoming_load_track_dict[dst]:
            self.incoming_load_track_dict[dst][src] += load
        else:
            self.incoming_load_track_dict[dst][src] = load

    def get_nodename_to_idx_mapping(self, G):
        self.nodename_to_idx_mapping = dict([(v,i) for i,v in enumerate(G.nodes())])

    # def gen_adj_over_time(self, G_over_time):
        # adj_over_time = []
        # for G in G_over_time:
            # adj = nx.to_numpy_array(G)
            # adj_over_time.append(adj)
        # return adj_over_time

    def set_seeds(self, seeds):
        self.seeds = seeds

    def set_n_replicates(self, n_replicates):
        self.n_replicates = n_replicates

    # B(x,y) = rho * C(x,y) / Ax
    # e.g. B = rho * C(0,0)/A0 , rho * C(0,1)/A0 , rho * C(0,2)/A0,
    #          rho * C(1,0)/A1 , rho * C(1,1)/A1 , rho * C(1,2)/A1,
    #          rho * C(2,0)/A2 , rho * C(2,1)/A2 , rho * C(2,2)/A2

    # >>> area_array = np.array(["A0", "A1", "A2"])
    # >>> np.tile(area_array, (area_array.shape[0], 1)).T
    # array([['A0', 'A0', 'A0'],
           # ['A1', 'A1', 'A1'],
           # ['A2', 'A2', 'A2']], dtype='<U2')

    # D(y,y) = (1 - d - sum_x (rho * C(x,y)/Ay))
    # D[0,0] is 1 - d - (sum of column 0 below)
    # D[1,1] is 1 - d - (sum of column 1 below)
    # E.g. (C(0,0)/A0, C(0,1)/A1, C(0,2)/A2)
    #      (C(1,0)/A0, C(1,1)/A1, C(1,2)/A2)
    #      (C(2,0)/A0, C(2,1)/A1, C(2,2)/A2)
    # probability of one getting infected is based on load on itself 

    def set_probability_of_infection(self, s, probability_array, load_array):
        if self.dose_response == "exponential":
            probability_array[:,s,:] = 1 - np.exp(-1 * self.pi * load_array[:,s,:])
        elif self.dose_response == "linear":
            probability_array[:,s,:] = np.minimum(1.0, self.pi * load_array[:,s,:]) # if pi*x > 1, set the infection prob=1

        # probability_array.shape[2] == self.people_nodes.shape[0], so this automatically does the rehaping.
        probability_array[:,s,:] *= self.people_nodes

    def infect(self, s, probability_array, infection_array):
        infection_array[:,s,:] = np.random.binomial(1, probability_array[:,s,:])
        # infection_array[:,t,:] = probability_array[:,t,:] > random_array[:,t,:]
        # infection_array[:,t,:] = probability_array[:,t,:] > random_array

    # def constructBplusD_over_time(self):
        # BplusD_over_time = []
        # for adj in self.adj_over_time:
            # C = self.contact_area * adj
            # Area = np.tile(self.area_array, (self.area_array.shape[0], 1)).T
            # B = self.rho * C / Area
            # D = np.diag(1 - self.d - np.sum(self.rho * C / Area.T, axis=0))
            # BplusD_over_time.append(B+D)
        # return BplusD_over_time

    def constructBplusD_over_time(self, G_over_time):
        BplusD_over_time = []
        for G in G_over_time:
            adj = nx.to_numpy_array(G)
            C = self.contact_area * adj
            Area = np.tile(self.area_array, (self.area_array.shape[0], 1)).T
            B = self.rho * C / Area
            D = np.diag(1 - self.d - np.sum(self.rho * C / Area.T, axis=0))
            BplusD_over_time.append(B+D)
        return BplusD_over_time
    
    # NOTE: granularity in the graph is in seconds (0, 1, 2, ..., 86399) or minutes 
    # We run simulation in minutes
    # A node can have contact w/ many other nodes at the same time
    # So update the load at s w/ load at s, and at the end of the for loop, copy the load at s to s+1
    # t denotes day, s denotes second or minute
    def load_sharing(self, t, s, load_array):
        G = self.G_over_time[t]
        for e in G.edges():
            attrs = G.edges[e]
            for start in attrs:
                end = attrs[start]
                # if int(start) <= s <= int(end): # simulation granularity: 1 second
                if int(start)//60 <= s <= int(end)//60: # simulation granularity: 60 second
                    v1, v2 = e[0], e[1]
                    v1_idx, v2_idx = self.nodename_to_idx_mapping[v1], self.nodename_to_idx_mapping[v2]
                    load_from_v1_to_v2 = self.rho * load_array[:, s, v1_idx] * self.contact_area / self.area_array[v1_idx]
                    load_from_v2_to_v1 = self.rho * load_array[:, s, v2_idx] * self.contact_area / self.area_array[v2_idx]
                    # update v1's load
                    load_array[:,s,v1_idx] += load_from_v2_to_v1
                    load_array[:,s,v1_idx] -= load_from_v1_to_v2
                    # update v2's load
                    load_array[:,s,v2_idx] += load_from_v1_to_v2
                    load_array[:,s,v2_idx] -= load_from_v2_to_v1
                    # Update incoming_load_track_dict
                    self.update_load_track_dict(v1, v2, load_from_v1_to_v2)
                    self.update_load_track_dict(v2, v1, load_from_v2_to_v1)
                    
        for v in G.nodes():
            v_idx = self.nodename_to_idx_mapping[v]
            load_array[:,s+1,v_idx] = load_array[:,s,v_idx]

    def shedding(self, s, infection_array, load_array):
        load_array[:,s+1,:] += infection_array[:,s,:].astype(np.float32) * self.q

    # If any seeds are infected in day t,
    # Infect seeds at time s (throughout the day)
    # NOTE: This can be modified to run once outside the for loop
    def infect_seeds(self, t, s, probability_array, infection_array):
        seeds_at_t = self.seeds[t,:].nonzero()[0]
        probability_array[:,s, seeds_at_t] = 1
        infection_array[:,s, seeds_at_t] = True

    def simulate(self):
        # ASSERTION NOT NEEDED on the parameters
        assert 0 <= self.d <= 1, "d < 0 or d < 1. Invalid"
        assert 0 <= self.rho <= 1, "rho < 0 or rho < 1. Invalid"
        assert self.contact_area <= np.min(self.area_array), "contact area cannot be larger than the surface area. Invalid"
        assert self.pi > 0, "pi <= 0. Invalid"
        assert 0 <= self.d + self.rho * (self.contact_area / np.min(self.area_array)) <= 1, "Adjust d, rho, or areas. Outgoing load may get larger than current load. Invalid"

        n_seconds_per_day = 24*60*60
        n_minutes_per_day = 24*60

        # These arrays are updated at the end of the day
        daily_probability_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(np.float32)
        daily_infection_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(bool)
        daily_load_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(np.float32)

        # These arrays keep track of information per second
        probability_array = np.zeros((self.n_replicates, n_minutes_per_day, self.number_of_nodes)).astype(np.float32)
        infection_array = np.zeros((self.n_replicates, n_minutes_per_day, self.number_of_nodes)).astype(bool)
        load_array = np.zeros((self.n_replicates, n_minutes_per_day, self.number_of_nodes)).astype(np.float32)

        # When t = 0, infect sources. Load sharing starts at t=0. But shedding starts at the end of time 0.
        # probability_array[0, self.seeds] = 1
        # infection_array[0, self.seeds] = True
        # load sharing
        # self.load_sharing(0, load_array)
        # Shedding
        # self.shedding(0, infection_array, load_array)
        # for t in range(0, self.n_timesteps-1):
        for t in range(0, self.n_timesteps):
            if t > 0: # Initialize the second=0 w/ last days info
                probability_array[:, 0, :] = daily_probability_array[:, t-1, :]
                load_array[:, 0, :] = daily_load_array[:, t-1, :]
                infection_array[:, 0, :] = daily_infection_array[:, t-1, :]
            for s in tqdm(range(0, n_minutes_per_day-1)):
                # Probabilty of infection (probability of infection is based on the load at the start)
                self.set_probability_of_infection(s, probability_array, load_array)

                self.infect(s, probability_array, infection_array)

                # Set seeds as infected (if there are any seeds at time t)
                self.infect_seeds(t, s, probability_array, infection_array)

                self.load_sharing(t, s, load_array) # from s to s+1
                self.shedding(s, infection_array, load_array) # from s to s+1

            # Set a node to be infected if at any time of the day the node was infected
            # daily_infection_array[:, t, :] = infection_array.sum(axis=1)

            daily_infection_array[:, t, :] = (infection_array.astype(float).mean(axis=1) >= self.infection_threshold )

            # Set the load array at the latest second
            daily_load_array[:, t, :] = load_array[:, s-1, :]
            daily_probability_array[:, t, :] = probability_array[:, s-1, :]

            # print daily results
            print("day{}, daily_infection_array[:,t,:].sum() : {}".format(t, daily_infection_array[:,t,:].sum()))
            print("daily_load_array[:,t,:].sum() : {}".format(daily_load_array[:,t,:].sum()))

        # Last timestep
        # self.set_probability_of_infection(self.n_timesteps-1, probability_array, load_array)
        # self.infect(self.n_timesteps-1, probability_array, infection_array)

        self.probability_array = daily_probability_array
        self.infection_array = daily_infection_array
        self.load_array = daily_load_array