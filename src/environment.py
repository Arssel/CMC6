import pickle
import numpy as np

import gym
from gym import spaces


from src.utils import path_distance
from sklearn.metrics import pairwise_distances

class LogEnv(gym.Env):
    
    def __init__(self, n = 20, batch_size = 16, opts=None):
        super().__init__()
        self._n = n
        self._bsz = batch_size
        self._opts = opts

    def reset(self):
        if 'graph_type' not in self._opts:
            self._opts['graph_type'] = None
        if 'pickup_and_delivery' not in self._opts:
            self._opts['pickup_and_delivery'] = None
        if 'demand_type' not in self._opts:
            self._opts['demand_type'] = None
        if 'tw_type' not in self._opts:
            self._opts['tw_type'] = None
        if 'supply' not in self._opts['demand_type']:
            self._opts['demand_type']['supply'] = None
          
        flags = {}
        if self._opts['pickup_and_delivery'] is None:
            flags['pd'] = False
        else:
            flags['pd'] = True
        if self._opts['demand_type'] is None:
            flags['demand'] = False
        else:
            flags['demand'] = True
        if self._opts['tw_type'] is None:
            flags['tw'] = False
        else:
            flags['tw'] = True
        if self._opts['demand_type']['supply'] is None:
            flags['supply']
        self._flags = flags    
        
        self._dots, _ = graph_generation(self._n, self._bsz, self._opts['graph_type']) 
        self._pairs = pickup_and_delivery_generation(self._n, self._bsz, self._opts['pickup_and_delivery']) 
        if self._pairs is not None:
            self._pd = p_and_d_features(self._pairs, self._dots)
        else:
            self._pd = np.zeros((self._bsz, self._n, 2))
        self._demand = demand_generation(self._n, self._bsz, self._opts['demand_type'], pd)
        self._tw = tw_generation(self._n, self._bsz, self._opts['tw_type'], pd)
        
        if flags['demand'] and not flags['pd']:
            if 'depot' not in self._opts:
                self._opts['depot'] = None
            self._depot = self.__generate_depot()
            self._flags['depot'] = True
        else:
            self._flags['depot'] = False
            
        self._features = np.concatenate((self._dots, self._demand, self._tw, self._pd), axis=2)
        self._distances = np.array([pairwise_distances(self._dots[i, :, :]) for i in range(self._bsz)])
        self.__set_parameters()
        self.__init_mask()
        return self._features, self._distances
        
    def __set_parameters(self):
        if self._flag['demand']:
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
        if self._flag['tw']:
            tw_type = self._opts['tw_type']
            if 'service_time' not in tw_type:
                tw_type['service_time'] = None
            if tw_type['service_time'] is None or tw_type['service_time'] == 'default':
                self._service_time = 0.1
            else:
                self._service_time = tw_type['service_time']
        self._cur_cap = np.zeros((self._bsz))
        self._cur_time = np.zeros((self._bsz))
        
        if self._flags['demand'] and not self._flags['pd']:
            self._cur_route = np.ones((self._bsz))*(self._n + 1)
        else:
            self._cur_route = []
                
    def __generate_depot(self, flags):
        total_demand = np.zeros((self._bsz, 1, 1))
        if 'supply' in self._opts['demand_type']:
            if self._opts['demand_type']['supply'] is not None:
                total_demand = self._demand.sum(axis=1).reshape(self._bsz, -1, 1)
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
        if flags['p_and_d']:
            pd = np.repeat(np.array([[0.5, 0.5]]), self._bsz, axis=0)
        else:
            pd = np.repeat(np.array([[0, 0]]), self._bsz, axis=0)
        coord = coord.reshape(self._bsz, -1, 2)
        tw = tw.reshape(self._bsz, -1, 2)
        pd = pd.reshape(self._bsz, -1, 2)
        self._dots = np.concat((self._dots, coord), axis=1)
        self._demand = np.concat((self._demand, total_demand), axis=1)    
        self._pd = np.concat((self._pd, pd), axis=1)    
        self._tw = np.concat((self._tw, tw), axis=1)

    def __init_mask(self):
        self._mask_visited = np.ones((self._bsz, self._n))
        self._mask_pd = np.ones((self._bsz, self._n))
        self._mask_demand = np.ones((self._bsz, self._n))
        self._mask_tw = np.ones((self._bsz, self._n))
        if self._flags['demand'] and not self._flags['pd']:
            self._mask_visited = np.append((self._mask_visited, np.ones(self._bsz, 1)), axis=1)
            self._mask_demand = np.append((self._mask_demand, np.ones(self._bsz, 1)), axis=1)
            self._mask_tw = np.append((self._mask_tw, np.ones(self._bsz, 1)), axis=1)
            if self._flags['supply']:
                self._mask_demand[self._demand > 0] = 0
        elif flags['pd']:
            self._mask_pd[self._pairs[:,:,1]] = 0
            
    def step(self, actions):
        if self._cur_route == []:
            self._cur_route = actions
        else:
            self._cur_route = np.append(self._cur_route, actions, axis=1)
        self._mask_visited[actions] = 0
        r = np.arange(self._bsz)
        if self._flag['pd']:
            for i in range(len(actions)):
                if actions[i] in self._pairs[i, :, 0]:
                    self._mask_pd[i, self._pairs[i, actions[i], 0]] = 0
                    self._mask_pd[i, self._pairs[i, actions[i], 1]] = 1
        if self._flags['demand'] and not self._flags['pd']:
            if not flags['supply']:
                self._mask_visited[actions != self._n + 1, -1] = 1
                self._mask_visited[self._mask_visited.sum(axis=1) == 0, -1] = 1
                self._cur_cap[actions == self._n + 1] = 0
                self._cur_cap += self._demand[r, actions]
                self._mask_demand = self._cur_cap + self._demand <= self._capacity
            else:
                self._cur_cap -= self.demand[r, actions]
                self._mask_demand = 0 <= (self._cur_cap - self._demand) & (self._cur_cap - self._demand) <= self._capacity
        elif self._flags['demand'] and self._flags['pd']:
            self._cur_cap -= self._demand[r, actions]
            self._mask_demand = 0 <= (self._cur_cap - self._demand) & (self._cur_cap - self._demand) <= self._capacity
        if self._flags['tw']:
            tw_type = self._opts['tw_type']
            self._cur_time += self._service_time + (np.tw[r, actions, 0] >= self._cur_time)*(np.tw[r, actions, 0] - self._cur_time)
            self._mask_tw = self._tw[:,:,1] > self._cur_time
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