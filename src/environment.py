import io
import pickle
from random import shuffle
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
from PIL import Image

from src.utils import path_distance

class TSPEnv(gym.Env):
    
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
            
        self._dots, self._distances = graph_generation(self._n, self._bsz, self._opts['graph_type']) 
        pd = pickup_and_delivery_generation(self._n, self._bsz, self._opts['pickup_and_delivery']) 
        if pd is not None:
            self._pd = p_and_d_features(pd, self._dots)
        else:
            self._pd = np.zeros((self._bsz, self._n, 2))
        self._demand = demand_generation(self._n, self._bsz, self._opts['demand_type'], pd)
        self._tw = tw_generation(self._n, self._bsz, self._opts['tw_type'], pd)
        self._features = np.concatenate((self._dots, self._demand, self._tw, self._pd), axis=2)
        return self._features
        
    def step(self, actions):
        return self._observations, reward, self._cur_step >= self._t

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