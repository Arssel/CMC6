import pickle
import numpy as np

import gym
from gym import spaces


from src.utils import path_distance, graph_generation, pickup_and_delivery_generation,\
    p_and_d_features, demand_generation, tw_generation
from sklearn.metrics import pairwise_distances

class LogEnv(gym.Env):
    
    def __init__(self, n = 20, batch_size = 16, opts=None):
        super().__init__()
        self._n = n
        self._bsz = batch_size
        self._opts = opts

    def reset(self, data={}):
        if self._opts is None:
            self._opts = {}
        if 'graph_type' not in self._opts:
            self._opts['graph_type'] = {}
        if 'pickup_and_delivery' not in self._opts:
            self._opts['pickup_and_delivery'] = {}
        if 'demand_type' not in self._opts:
            self._opts['demand_type'] = {}
        if 'tw_type' not in self._opts:
            self._opts['tw_type'] = {}
        if 'supply' not in self._opts['demand_type']:
            self._opts['demand_type']['supply'] = {}
          
        flags = {}
        if self._opts['pickup_and_delivery'] == True:
            flags['pd'] = True
        else:
            flags['pd'] = False
        if len(self._opts['demand_type']) == 1:
            flags['demand'] = False
        else:
            flags['demand'] = True
        if len(self._opts['tw_type']) == 0:
            flags['tw'] = False
        else:
            flags['tw'] = True
        if len(self._opts['demand_type']['supply']) == 0:
            flags['supply'] = False
        else:
            flags['supply'] = True
        self._flags = flags    
        
        if 'dots' in data:
            self._dots = data['dots']
        else:
            self._dots, _ = graph_generation(self._n, self._bsz, self._opts['graph_type']) 
        if 'pairs' in data:
            self._pairs = data['pairs']
        else:
            self._pairs = pickup_and_delivery_generation(self._n, self._bsz, self._opts) 
        if self._pairs is not None:
            self._pd = p_and_d_features(self._pairs, self._dots)
        else:
            self._pd = None
        if 'demand' in data:
            self._demand = data['demand']
        else:
            self._demand = demand_generation(self._n, self._bsz, self._opts['demand_type'], self._pairs)
        if 'tw' in data:
            self._tw = data['tw']
        else:
            self._tw = tw_generation(self._n, self._bsz, self._opts['tw_type'], self._pairs)
        if self._pd is None:
            self._pd = np.zeros((self._bsz, self._n, 2))
        if flags['demand'] and not flags['pd']:
            if 'depot' not in self._opts:
                self._opts['depot'] = None
            self._depot = self.generate_depot(self._flags)
            self._flags['depot'] = True
        else:
            self._flags['depot'] = False
            
        self._features = np.concatenate((self._dots, self._demand, self._tw, self._pd), axis=2)
        self._distances = np.array([pairwise_distances(self._dots[i, :, :]) for i in range(self._bsz)])
        self._time_d = self._distances/10
        self.set_parameters()
        self.init_mask()
        return self._features, self._distances, self._mask_visited*self._mask_pd*self._mask_demand*self._mask_tw
        
    def set_parameters(self):
        if self._flags['demand']:
            demand_type = self._opts['demand_type']
            if 'capacity' not in self._opts['demand_type']:
                demand_type['capacity'] = None
            if demand_type['capacity'] is None or demand_type['capacity'] == 'default':
                if self._n == 20:
                    self._capacity = 20
                else:
                    self._capacity = 30
            else:
                self._capacity = demand_type['capacity']
        if self._flags['tw']:
            tw_type = self._opts['tw_type']
            if 'service_time' not in tw_type:
                tw_type['service_time'] = None
            if tw_type['service_time'] is None or tw_type['service_time'] == 'default':
                self._service_time = 0.005
            else:
                self._service_time = tw_type['service_time']
        self._cur_cap = np.zeros((self._bsz))
        self._cur_time = np.zeros((self._bsz))
        
        if self._flags['demand'] and not self._flags['pd']:
            self._cur_route = np.ones((self._bsz, 1))*(self._n + 1)
        else:
            self._cur_route = []
                
    def generate_depot(self, flags):
        total_demand = np.zeros((self._bsz, 1, 1))
        if 'supply' in self._opts['demand_type']:
            if self._opts['demand_type']['supply'] is not None:
                total_demand = self._demand.sum(axis=1).reshape(self._bsz, -1, 1)
        if 'depot' not in self._opts['demand_type']:
            self._opts['demand_type']['depot'] = None
        depot = self._opts['demand_type']['depot']
        if depot is None:
            coord = np.repeat(np.array([[0.5, 0.5]]), self._bsz, axis=0)
        else:
            if depot['distribution'] == 'random':
                coord = np.repeat(np.random.rand(1, 2), self._bsz, axis=0)
            elif depot['distribution'] == 'deafult':
                coord = np.repeat(np.array([[0.5, 0.5]]), self._bsz, axis=0)
            else:
                coord = np.repeat(np.array([depot['distribution']]), self._bsz, axis=0)
        if flags['tw']:
            tw = np.repeat(np.array([[0, 1]]), self._bsz, axis=0)
        else:
            tw = np.repeat(np.array([[0, 0]]), self._bsz, axis=0)
        if flags['pd']:
            pd = np.repeat(np.array([[0.5, 0.5]]), self._bsz, axis=0)
        else:
            pd = np.repeat(np.array([[0, 0]]), self._bsz, axis=0)
        coord = coord.reshape(self._bsz, -1, 2)
        tw = tw.reshape(self._bsz, -1, 2)
        pd = pd.reshape(self._bsz, -1, 2)
        self._dots = np.concatenate((self._dots, coord), axis=1)
        self._demand = np.concatenate((self._demand, total_demand), axis=1)    
        self._pd = np.concatenate((self._pd, pd), axis=1)    
        self._tw = np.concatenate((self._tw, tw), axis=1)

    def init_mask(self):
        r = np.arange(self._bsz)
        self._mask_visited = np.ones((self._bsz, self._n), dtype=int)
        self._mask_pd = np.ones((self._bsz, self._n), dtype=int)
        self._mask_demand = np.ones((self._bsz, self._n), dtype=int)
        self._mask_tw = np.ones((self._bsz, self._n), dtype=int)
        if self._flags['demand'] and not self._flags['pd']:
            self._mask_visited = np.append(self._mask_visited, np.ones((self._bsz, 1), dtype=int), axis=1)
            self._mask_demand = np.append(self._mask_demand, np.ones((self._bsz, 1), dtype=int), axis=1)
            self._mask_tw = np.append(self._mask_tw, np.ones((self._bsz, 1), dtype=int), axis=1)
            self._mask_pd = np.append(self._mask_pd, np.ones((self._bsz, 1), dtype=int), axis=1)
            if self._flags['supply']:
                self._mask_demand[self._demand > 0] = 0
        elif self._flags['pd']:
            for i in range(self._bsz):
                self._mask_pd[i, self._pairs[i,:,1]] = 0
        return self._mask_visited*self._mask_pd*self._mask_demand*self._mask_tw
            
    def step(self, actions):
        if len(self._cur_route) == 0:
            self._cur_route = actions.reshape(-1,1)
        else:
            self._cur_route = np.append(self._cur_route, actions, axis=1)
        r = np.arange(self._bsz)
        self._mask_visited[r, actions.squeeze()] = 0
        if self._flags['pd']:
            for i in range(len(actions)):
                if actions[i] in self._pairs[i, :, 0]:
                    source = self._pairs[i, :, 0] == actions[i]
                    self._mask_pd[i, self._pairs[i, source, 0]] = 0
                    self._mask_pd[i, self._pairs[i, source, 1]] = 1
        if self._flags['demand'] and not self._flags['pd']:
            if not self._flags['supply']:
                self._mask_visited[(actions != self._n).squeeze(1), -1] = 1
                self._mask_visited[self._mask_visited.sum(axis=1) == 0, -1] = 1
                self._cur_cap += self._demand[r, actions.squeeze(1),:].squeeze()
                self._cur_cap[(actions == self._n).squeeze(1)] = 0
                self._mask_demand = (self._cur_cap + self._demand.squeeze(2)) <= self._capacity
                self._mask_demand[:, -1] = 1
            else:
                self._cur_cap -= self.demand[r, actions.squeeze()]
                self._mask_demand = np.logical_and(0 <= (self._cur_cap.reshape(-1, 1) - self._demand.squeeze()), (self._cur_cap.reshape(-1, 1) - self._demand.squeeze()) <= self._capacity)
        elif self._flags['demand'] and self._flags['pd']:
            self._cur_cap -= self._demand[r, actions.squeeze()].squeeze()
            self._mask_demand = np.logical_and((0 <= (self._cur_cap.reshape(-1, 1) - self._demand.squeeze())), ((self._cur_cap.reshape(-1, 1) - self._demand.squeeze()) <= self._capacity))
        if self._flags['tw']:
            if self._cur_route.shape[1] == 1:
                self._cur_time += self._service_time + (self._tw[r, actions.squeeze(), 0] - self._cur_time)
            else:
                self._cur_time += self._service_time + \
                    (self._tw[r, actions.squeeze(), 0] >= self._cur_time + self._time_d[r, self._cur_route[:, -2].squeeze(), actions.squeeze()])*(self._tw[r, actions.squeeze(), 0] - self._cur_time) +\
                    (self._tw[r, actions.squeeze(), 0] < self._cur_time + self._time_d[r, self._cur_route[:, -2].squeeze(), actions.squeeze()])*(self._time_d[r, self._cur_route[:, -2].squeeze(), actions.squeeze()])
            self._mask_tw = self._tw[:,:,1] > self._cur_time.reshape(-1, 1) + self._time_d[r, actions.squeeze()]
            if (self._mask_visited*self._mask_tw)[:, :self._n].sum() > 0 and ((self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0).any():
                self._mask_tw[r[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0], self._cur_route[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0, 0]] = 1
                self._mask_visited[r[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0], self._cur_route[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0, 0]] = 1
                self._mask_pd[r[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0], self._cur_route[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0, 0]] = 1
                self._mask_demand[r[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0], self._cur_route[(self._mask_visited*self._mask_tw)[:, :self._n].sum(axis=1) == 0, 0]] = 1
        full_mask = self._mask_visited*self._mask_pd*self._mask_demand*self._mask_tw
        return full_mask, (self._mask_visited*self._mask_tw)[:, :self._n].sum() == 0
 
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            
    def save_graph(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self._dots, f)

    @staticmethod
    def load(filename) -> "TSPEnv":
        with open(filename, "rb") as f:
            return pickle.load(f)