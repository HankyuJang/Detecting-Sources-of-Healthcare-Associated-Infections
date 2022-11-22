"""
Simulator. Load sharing model on temporal graph.

Author: -
Email: -
Last Modified: Jan, 2022

Description: Simulator for the load sharing model

NOTE: use scipy.sparse for large graphs

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
from scipy import sparse
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
            n_replicates = 10):
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
        # self.number_of_nodes = len(G)
        self.number_of_nodes = len(G_over_time[0])
        # For temporal graph, compute BplusD for each timestep
        self.BplusD_over_time = self.constructBplusD_over_time(G_over_time)
        self.n_replicates = n_replicates

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

    def set_probability_of_infection(self, t, probability_array, load_array):
        if self.dose_response == "exponential":
            probability_array[:,t,:] = 1 - np.exp(-1 * self.pi * load_array[:,t,:])
        elif self.dose_response == "linear":
            probability_array[:,t,:] = np.minimum(1.0, self.pi * load_array[:,t,:]) # if pi*x > 1, set the infection prob=1

        # probability_array.shape[2] == self.people_nodes.shape[0], so this automatically does the rehaping.
        probability_array[:,t,:] *= self.people_nodes

    def infect(self, t, probability_array, infection_array):
        infection_array[:,t,:] = np.random.binomial(1, probability_array[:,t,:])
        # infection_array[:,t,:] = probability_array[:,t,:] > random_array[:,t,:]
        # infection_array[:,t,:] = probability_array[:,t,:] > random_array

    def constructBplusD_over_time(self, G_over_time):
        BplusD_over_time = []
        print("Preparing list of B+D sparse matrices")
        for G in tqdm(G_over_time):
            adj = nx.to_numpy_array(G)
            # adj = nx.to_scipy_sparse_matrix(G) #NOTE: Don't use this. behavior when np.diag in the later line was different.
            C = self.contact_area * adj
            Area = np.tile(self.area_array, (self.area_array.shape[0], 1)).T
            B = self.rho * C / Area
            D = np.diag(1 - self.d - np.sum(self.rho * C / Area.T, axis=0))
            BplusD_over_time.append(B+D)
            # sparse.csr_matrix(B+D)
            # BplusD_over_time.append(sparse.csr_matrix(B+D))
        return BplusD_over_time
    
    def load_sharing(self, t, load_array):
        L = load_array[:,t,:]
        L_next = np.dot(L, self.BplusD_over_time[t]) # no need to reshape here.
        load_array[:,t+1,:] = L_next

    def shedding(self, t, infection_array, load_array):
        load_array[:,t+1,:] += infection_array[:,t,:].astype(np.float32) * self.q

    def infect_seeds(self, t, probability_array, infection_array):
        seeds_at_t = self.seeds[t,:].nonzero()[0]
        probability_array[:,t, seeds_at_t] = 1
        infection_array[:,t, seeds_at_t] = True

    def simulate(self):
        # ASSERTION on the parameters
        assert 0 <= self.d <= 1, "d < 0 or d < 1. Invalid"
        assert 0 <= self.rho <= 1, "rho < 0 or rho < 1. Invalid"
        assert self.contact_area <= np.min(self.area_array), "contact area cannot be larger than the surface area. Invalid"
        assert self.pi > 0, "pi <= 0. Invalid"
        assert 0 <= self.d + self.rho * (self.contact_area / np.min(self.area_array)) <= 1, "Adjust d, rho, or areas. Outgoing load may get larger than current load. Invalid"

        probability_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(np.float32)
        infection_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(bool)
        load_array = np.zeros((self.n_replicates, self.n_timesteps, self.number_of_nodes)).astype(np.float32)
        # random_array = np.random.random((self.n_replicates, self.n_timesteps, self.number_of_nodes))
        # random_array = np.random.random((self.number_of_nodes))

        # When t = 0, infect sources. Load sharing starts at t=0. But shedding starts at the end of time 0.
        # probability_array[0, self.seeds] = 1
        # infection_array[0, self.seeds] = True
        # load sharing
        # self.load_sharing(0, load_array)
        # Shedding
        # self.shedding(0, infection_array, load_array)
        # for t in range(1, self.n_timesteps-1):
        for t in range(0, self.n_timesteps-1):
            # Probabilty of infection (probability of infection is based on the load at the start)
            self.set_probability_of_infection(t, probability_array, load_array)
            # self.infect(t, probability_array, infection_array, random_array)
            self.infect(t, probability_array, infection_array)

            # Set seeds as infected (if there are any seeds at time t)
            self.infect_seeds(t, probability_array, infection_array)

            self.load_sharing(t, load_array)
            self.shedding(t, infection_array, load_array)

        # Last timestep
        self.set_probability_of_infection(self.n_timesteps-1, probability_array, load_array)
        # self.infect(self.n_timesteps-1, probability_array, infection_array, random_array)
        self.infect(self.n_timesteps-1, probability_array, infection_array)

        self.probability_array = probability_array
        self.infection_array = infection_array
        self.load_array = load_array
