import numpy as np
import torch
from tqdm import tqdm
import time

from src.environment_jampr import LogEnv
from sklearn.metrics import pairwise_distances
from src.utils import path_distance_jampr, check_missing_vertexes_jampr
from src.or_functions_jampr import compute_distance

def compute_mean_metric(model, device="cuda", n=20, batch_size=250, T=40, sample=False):
    env = LogEnv(n=n, batch_size=batch_size)
    metric = np.zeros((T,))
    for i in tqdm(range(T)):
        
            features, distances, mask = env.reset()
            features = list(map(lambda x: None if x is None else x.to(device), features))
            flag_done = False
            t = 0
            while not flag_done:
                v, p = model(features, mask, t, sample)
                v = v.to('cpu')
                with torch.no_grad():
                    features, mask, flag_done = env.step(v)
                    features = list(map(lambda x: None if x is None else x.to(device), features))
                t += 1
            routes_length = features[2][:, :, 4].sum(dim=1).to('cpu')
            routes_length += check_missing_vertexes_jampr(env.tour_plan, n) * 1000
            print(check_missing_vertexes_jampr(env.tour_plan, n) * 1000)
            metric[i] = routes_length.mean()
            
    return metric.mean()

def compute_mean_metric_with_or(model, device="cuda", n=20, batch_size=250, T=40, time_limit=0.5, sample=False, eps=1e-5):
    env = LogEnv(n=n, batch_size=batch_size)
    metric_model = np.zeros((T,))
    metric_or = np.zeros((T, batch_size))
    for i in range(T):
        
        features, distances, mask = env.reset()
        features = list(map(lambda x: None if x is None else x.to(device), features))
        flag_done = False
        t = 0
        while not flag_done:
            v, _ = model(features, mask, t, sample)
            v = v.to('cpu')
            with torch.no_grad():
                features, mask, flag_done = env.step(v)
                features = list(map(lambda x: None if x is None else x.to(device), features))
            t += 1
        routes_length = features[2][:, :, 4].sum(dim=1).to('cpu')
        routes_length += check_missing_vertexes_jampr(env.tour_plan, n) * 1000
        print(check_missing_vertexes_jampr(env.tour_plan, n) * 1000)
        metric_model[i] = routes_length.mean()
        print(env.tour_plan)
        for j in range(batch_size):
            data = {}
            data['time_matrix'] = env.distance.numpy()[j].squeeze()
            data['num_vehicles'] = 10
            data['time_windows'] = env.tw.numpy()[j].squeeze()
            data['demands'] = env.demand.numpy()[j].squeeze()*env.capacity
            data['vehicle_capacities'] = [env.capacity]*data['num_vehicles']
            data['pickups_deliveries'] = env.pairs
            print(data['time_windows'])
            print(env.pairs)
            print(env.tour_plan[j])
            metric_or[i, j] = compute_distance(data, eps=eps, time_limit=time_limit)
            print(metric_or[i, j])
            
    return metric_model.mean(), metric_or.mean()