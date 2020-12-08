import numpy as np
import torch

from src.environment import LogEnv
from src.utils import path_distance_new

def compute_mean_metric(model, device="cuda", n=20, batch_size=250, T=40, opts={}, sample=False):
    env = LogEnv(n=n, batch_size=batch_size, opts=opts)
    metric = np.zeros((T,))
    for i in range(T):
        
        features, distances, mask, context = env.reset()
        features = torch.Tensor(features).to(device)
        context = torch.Tensor(context).to(device)
        flag_done = False
        t = 0
        
        while not flag_done:
            v, _ = model(features, mask, t, context, flags=env._flags, sample=sample)
            v = v.to('cpu')
            with torch.no_grad():
                mask, flag_done, context = env.step(v)
                context = torch.Tensor(context).to(device)
            t += 1
            
        route = torch.tensor(env._cur_route, dtype=int)
        metric[i] = path_distance_new(distances, route).mean()
    return metric.mean()

def compute_data_metric(model, data, dist, device="cuda", n=20, opts={}, sample=False):
    T = len(data)
    env = LogEnv(n=n, batch_size=1, opts=opts)
    metric = np.zeros((T,))
    for i in range(T):
        features, distances, mask, context = env.reset(data[i])
        features = torch.Tensor(features).to(device)
        context = torch.Tensor(context).to(device)
        flag_done = False
        t = 0
        
        while not flag_done:
            v, _ = model(features, mask, t, context, flags=env._flags, sample=sample)
            v = v.to('cpu')
            with torch.no_grad():
                mask, flag_done, context = env.step(v)
                context = torch.Tensor(context).to(device)
            t += 1
            
        route = torch.tensor(env._cur_route, dtype=int)
        metric[i] = path_distance_new(dist[i], route)
    return metric