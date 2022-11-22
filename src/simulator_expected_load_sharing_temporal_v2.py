"""
Simulator. Expected Load sharing model on temporal graph.

Author: -
Email: -
Last Modified: Jan, 2022

v2. Do 1 expected load sharing till T - n_t_for_original_simulation. Then resume 100 replicates from there.
NOTE: Call n_replicates same as original simulation.

Description: Simulator for the load sharing model in expectation.

NOTE: only infect nodes at their final timestep

G: graph
seeds: 2-d boolean array denoting seeds at timestep t. Dim 0: timestep, Dim 1: nodes of size(G)
people_nodes: 1-d boolean array denoting if a node is a person. True if the node is a person. (can get infected). size(G)
area_array: 1-d array. area of each node in graph. size(G)
contact_area: area of a contact
n_timesteps: number of timesteps
rho: transfer efficiency
d: decay rate
q: shedding rate. amount of pathogen an infection agent sheds on its surface

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
            n_t_for_original_simulation = 2
            ):
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
        self.BplusD_over_time = self.constructBplusD_over_time(G_over_time)
        self.n_replicates = n_replicates
        self.n_t_for_original_simulation = n_t_for_original_simulation

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

    def set_n_t_for_original_simulation(self, n_t_for_original_simulation):
        self.n_t_for_original_simulation = n_t_for_original_simulation

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

    def set_probability_of_infection(self, t, probability_array, load_array):
        if self.dose_response == "exponential":
            probability_array[:,t,:] = 1 - np.exp(-1 * self.pi * load_array[:,t,:])
        elif self.dose_response == "linear":
            probability_array[:,t,:] = np.minimum(1.0, self.pi * load_array[:,t,:]) # if pi*x > 1, set the infection prob=1

        probability_array[:,t,:] *= self.people_nodes

    # ONLY infect nodes at the final timestep
    def infect(self, t, probability_array, infection_array):
        infection_array[:,t,:] = np.random.binomial(1, probability_array[:,t,:])

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
    
    def load_sharing(self, t, load_array):
        L = load_array[:,t,:]
        L_next = np.dot(L, self.BplusD_over_time[t]) # no need to reshape here.
        load_array[:,t+1,:] = L_next

    def shedding(self, t, probability_array, load_array):
        load_array[:,t+1,:] += probability_array[:,t,:] * self.q

    # This is original shedding of those from infected nodes. NOTE: NOT EXPECTED SHEDDING
    def shedding_original(self, t, infection_array, load_array):
        load_array[:,t+1,:] += infection_array[:,t,:].astype(np.float32) * self.q

    def infect_seeds(self, t, probability_array):#, infection_array):
        seeds_at_t = self.seeds[t,:].nonzero()[0]
        probability_array[:,t, seeds_at_t] = 1
        # infection_array[:,t, seeds_at_t] = True

    def simulate(self):
        # ASSERTION on the parameters
        assert 0 <= self.d <= 1, "d < 0 or d < 1. Invalid"
        assert 0 <= self.rho <= 1, "rho < 0 or rho < 1. Invalid"
        assert self.contact_area <= np.min(self.area_array), "contact area cannot be larger than the surface area. Invalid"
        assert self.pi > 0, "pi <= 0. Invalid"
        assert 0 <= self.d + self.rho * (self.contact_area / np.min(self.area_array)) <= 1, "Adjust d, rho, or areas. Outgoing load may get larger than current load. Invalid"

        probability_array_1rep = np.zeros((1, self.n_timesteps, self.number_of_nodes)).astype(np.float32)
        infection_array_1rep = np.zeros((1, self.n_timesteps, self.number_of_nodes)).astype(bool)
        load_array_1rep = np.zeros((1, self.n_timesteps, self.number_of_nodes)).astype(np.float32)
        ##############################################
        # Do expected load sharing for time 0, 1, ... , T - n_t_for_original_simulation
        # Set T = self.n_timesteps-1
        # If n_t_for_original_simulation == 1, then expected load sharing for 0, 1, ... , T-1  , then infect nodes for T
        # If n_t_for_original_simulation == 2, then expected load sharing for 0, 1, ... , T-2  , Do original load sharing for T-1, then infect nodes for T
        T = self.n_timesteps-1

        for t in range(0, T - self.n_t_for_original_simulation + 1):
            # print("t: {} - expected simulation".format(t))
            # Probabilty of infection (probability of infection is based on the load at the start)
            self.set_probability_of_infection(t, probability_array_1rep, load_array_1rep)

            # Set seeds as infected (if there are any seeds at time t)
            self.infect_seeds(t, probability_array_1rep)
            self.load_sharing(t, load_array_1rep)
            self.shedding(t, probability_array_1rep, load_array_1rep)

        probability_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(np.float32)
        infection_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(bool)
        load_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(np.float32)

        t_copy = T - self.n_t_for_original_simulation + 1
        # print("t: {} - time when copying arrays".format(t_copy))
        # Copy the values in *1rep arrays to the arrays for running original simulation
        # NOTE: Only need to copy the load array at t.
        for idx_replicate in range(self.n_replicates):
            # probability_array[idx_replicate, t, :] = probability_array_1rep[0, t, :]
            # infection_array[idx_replicate, t, :] = infection_array_1rep[0, t, :]
            load_array[idx_replicate, t_copy, :] = load_array_1rep[0, t_copy, :]

        ##############################################
        # Do original load sharing for time T - n_t_for_original_simulation + 1 , ... , T - 1
        for t in range(t_copy, T):
            # print("t: {} - original simulation".format(t))
            self.set_probability_of_infection(t, probability_array, load_array)
            self.infect(t, probability_array, infection_array)
            self.load_sharing(t, load_array)
            self.shedding_original(t, infection_array, load_array)

        ##############################################
        # Infect those at T. Last timestep
        self.set_probability_of_infection(T, probability_array, load_array)
        self.infect(T, probability_array, infection_array)

        self.probability_array = probability_array
        self.infection_array = infection_array
        self.load_array = load_array
