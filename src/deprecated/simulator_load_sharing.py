"""
Simulator. Load sharing model.

Author: Bijaya Adhikari, Hankyu Jang
Email: bijaya-adhikari@uiowa.edu, hankyu-jang@uiowa.edu, 
Last Modified: Feb, 2020 

Description: Simulator for the load sharing model

G: graph
seeds: 2-d boolean array denoting seeds at timestep t. Dim 0: timestep, Dim 1: nodes of size(G)
people_nodes: 1-d boolean array denoting if a node is a person. True if the node is a person. (can get infected). size(G)
area_array: 1-d array. area of each node in graph. size(G)
contact_area: area of a contact
n_timesteps: number of timesteps
rho: transfer efficiency
d: decay rate
q: shedding rate. amount of pathogen an infection agent sheds on its surface
"""
# Simulation class for the load sharing model
import numpy as np
import networkx as nx

class Simulation:
    def __init__(self,
            G,
            seeds,
            people_nodes,
            area_array,
            contact_area,
            n_timesteps,
            rho,
            d,
            q):
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
        self.adj = nx.to_numpy_array(G)# adj is 2d numpy array. adjacency.
        self.number_of_nodes = len(G)
        # NOTE: For temporal graph, compute BplusD for each timestep
        self.BplusD = self.constructBplusD()

    def set_seeds(self, seeds):
        self.seeds = seeds

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
        probability_array[t,:] = 1 - np.exp(-1 * load_array[t,:])
        probability_array[t,:] *= self.people_nodes

    def infect(self, t, probability_array, infection_array):
        infection_array[t,:] = np.random.binomial(1, probability_array[t,:])

    def constructBplusD(self):
        # adj = nx.to_numpy_array(self.G) 
        C = self.contact_area * self.adj
        Area = np.tile(self.area_array, (self.area_array.shape[0], 1)).T
        B = self.rho * C / Area
        D = np.diag(1 - self.d - np.sum(self.rho * C / Area.T, axis=0))
        return B + D
    
    def load_sharing(self, t, load_array):
        L = load_array[t,:]
        L_next = np.dot(L, self.BplusD)
        load_array[t+1,:] = L_next

    def shedding(self, t, infection_array, load_array):
        load_array[t+1,:] += infection_array[t,:].astype(np.float32) * self.q

    def infect_seeds(self, t, probability_array, infection_array):
        seeds_at_t = self.seeds[t,:].nonzero()[0]
        probability_array[t, seeds_at_t] = 1
        infection_array[t, seeds_at_t] = True

    def simulate(self):
        probability_array = np.zeros((self.n_timesteps, self.number_of_nodes)).astype(np.float32)
        infection_array = np.zeros((self.n_timesteps, self.number_of_nodes)).astype(bool)
        load_array = np.zeros((self.n_timesteps, self.number_of_nodes)).astype(np.float32)

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
            self.infect(t, probability_array, infection_array)

            # Set seeds as infected (if there are any seeds at time t)
            self.infect_seeds(t, probability_array, infection_array)

            self.load_sharing(t, load_array)
            self.shedding(t, infection_array, load_array)

        # Last timestep
        self.set_probability_of_infection(self.n_timesteps-1, probability_array, load_array)
        self.infect(self.n_timesteps-1, probability_array, infection_array)

        self.probability_array = probability_array
        self.infection_array = infection_array
        self.load_array = load_array
