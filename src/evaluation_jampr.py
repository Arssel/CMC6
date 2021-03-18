import numpy as np
import torch
from tqdm import tqdm
import time

import GPUtil

from src.environment_jampr_real import LogEnv
from sklearn.metrics import pairwise_distances
from src.utils import path_distance_jampr, check_missing_vertexes_jampr
from src.or_functions_jampr import compute_distance

def compute_mean_metric(model, device="cuda", n=20, batch_size=250, T=40, sample=False):
    env = LogEnv(n=n, batch_size=batch_size, active_num=1, K=12)
    metric = np.zeros((T,))
    for i in tqdm(range(T)):

            features, distances, mask = env.reset()
            features[0] = features[0].to(device)
            features[1] = features[1].to(device)
            flag_done = False
            t = 0
            precomputed = None
            while not flag_done:
                v, _, precomputed = model(features, mask, t, precomputed, sample)
                v = v.to('cpu')
                with torch.no_grad():
                    features, mask, flag_done = env.step(v)
                    features[1] = features[1].to(device)
                t += 1
            routes_length = env.vehicle[:, :, -1].sum(dim=1).to('cpu')
            routes_length += check_missing_vertexes_jampr(env.tour_plan, n)
            print(check_missing_vertexes_jampr(env.tour_plan, n))
            metric[i] = routes_length.mean()
            
    return metric.mean()

def compute_mean_metric_with_or(model, device="cuda", n=20, batch_size=250, T=40, time_limit=0.5, sample=False, eps=1e-5,
                                K=12):
    env = LogEnv(n=n, batch_size=batch_size, active_num=1, K=K)
    metric_model = np.zeros((T,))
    metric_or = np.zeros((T, batch_size))
    for i in range(T):

        features, distances, mask = env.reset()
        features[0] = features[0].to(device)
        features[1] = features[1].to(device)
        flag_done = False
        t = 0
        precomputed = None
        while not flag_done:
            v, _, precomputed = model(features, mask, t, precomputed, sample)
            v = v.to('cpu')
            with torch.no_grad():
                features, mask, flag_done = env.step(v)
                features[1] = features[1].to(device)
            t += 1
        routes_length = env.vehicle[:, :, -1].sum(dim=1).to('cpu')
        routes_length += check_missing_vertexes_jampr(env.tour_plan, n)
        print(check_missing_vertexes_jampr(env.tour_plan, n))
        metric_model[i] = routes_length.mean()
        #print(env.tour_plan)
        for j in range(batch_size):
            data = {}
            data['time_matrix'] = env.distance.numpy()[j].squeeze()
            data['num_vehicles'] = K
            data['time_windows'] = env.tw.numpy()[j].squeeze()
            data['demands'] = env.demand.numpy()[j].squeeze() * env.capacity
            data['vehicle_capacities'] = [env.capacity] * data['num_vehicles']
            data['starts'] = [0] * data['num_vehicles']
            data['ends'] = [n+1] * data['num_vehicles']
            #data['pickups_deliveries'] = env.pairs
            #print(data['time_windows'])
            #print(env.pairs)
            #print(env.tour_plan[j])
            metric_or[i, j] = compute_distance(data, eps=eps, time_limit=time_limit)
            #print(metric_or[i, j])
            
    return metric_model.mean(), metric_or.mean()

def compute_metric_on_data(model, data, device="cuda", n=100, time_limit=0.5, sample=False, eps=1e-5, K=12):
    env = LogEnv(n=100, batch_size=1, active_num=1, K=K)
    features, distances, mask = env.reset(data=data)
    features[0] = features[0].to(device)
    features[1] = features[1].to(device)
    flag_done = False
    t = 0
    precomputed = None
    while not flag_done:
        v, _, precomputed = model(features, mask, t, precomputed, sample)
        v = v.to('cpu')
        with torch.no_grad():
            features, mask, flag_done = env.step(v)
            features[1] = features[1].to(device)
        t += 1
    routes_length = env.vehicle[:, :, -1].sum(dim=1).to('cpu')
    routes_length += check_missing_vertexes_jampr(env.tour_plan, n)
    print(check_missing_vertexes_jampr(env.tour_plan, n))
    metric_model = routes_length.detach().item()
    model_route = env.tour_plan
    metric_or, or_route = compute_distance(data, eps=eps, time_limit=time_limit)
            # print(metric_or[i, j])

    return metric_model, metric_or, (model_route, or_route)