import numpy as np
import torch

from src.environment_am import LogEnv
from sklearn.metrics import pairwise_distances
from src.utils import path_distance_new, check_missing_vertexes
from src.or_functions_am import compute_distance

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
        if env._flags['tw']:
            metric[i] = path_distance_new(env._time_d, route).mean()
            metric[i] += (check_missing_vertexes(route, env._flags, opts, n)*4).mean()
        else:
            metric[i] = path_distance_new(distances, route).mean()
    return metric.mean()

def compute_mean_metric_with_or(model, device="cuda", n=20, batch_size=250, T=40, options={}, sample=False, time_limit=0.5):
    opts = options
    env = LogEnv(n=n, batch_size=batch_size, opts=options)
    metric_model = np.zeros((T,))
    metric_or = np.zeros((T, batch_size))
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
        metric_model[i] = path_distance_new(distances, route).mean()
        dists = env._distances
        demand = env._demand
        
        for j in range(batch_size):
            data = {}
            dots = env._dots[j].squeeze()
            data['dist'] = dists[j].squeeze()
            if 'demand_type' in opts:
                data['demand'] = demand[j].squeeze()
                data['num_vehicles'] = 1
            if 'pickup_and_delivery' in opts:
                data['pickup_and_delivery'] = env._pairs[j].squeeze()
                data['depot'] = np.random.choice(np.arange(env._pairs[j].squeeze().shape[0]))
                n_x = data['pickup_and_delivery'][data['depot'], 0]
                tmp = dots[n_x].copy()
                dots[n_x] = dots[0]
                dots[0] = tmp
                print(data['pickup_and_delivery'])
                print(n_x)
                data['pickup_and_delivery'][data['pickup_and_delivery'] == n_x] = 30000
                data['pickup_and_delivery'][data['pickup_and_delivery'] == 0] = data['depot']
                data['pickup_and_delivery'][data['pickup_and_delivery'] == 30000] = 0 
                print(data['pickup_and_delivery'])
                data['dist'] = pairwise_distances(dots)
                data['depot'] = 0
            #print(data)
            #print(opts)
            metric_or[i, j] = compute_distance(data, eps=1e-5, time_limit=time_limit)
            
    return metric_model.mean(), metric_or.mean()

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