from src.architecture import AttentionModel
from src.environment import LogEnv
from src.utils import path_distance_new, check_repeating_vertexes, check_missing_vertexes

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import time

from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter

def adjust_learning_rate(optimizer, epoch, lr, decay):
    lr_new = lr * (decay ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

def parse_flags(flags):
    problem = ''
    if flags['pd']:
        problem += 'PD'
    if flags['demand']:
        problem += 'D'
    if flags['tw']:
        problem += 'TW'
    problem += 'TSP'
    return problem
    
        
def train(model, opts, device="cuda", batch_size=256, epochs=100, T=40, lr=1e-4, problem_size=20, decay=0.8):
    ''' 
    actor - Actor class for action choice
    critic - Critic class for state evaluation
    device - name of device to train on
    batch_size - batch size parameter
    gamma - discounting factor for reward
    lr - learning rate
    epochs - number of epochs to train
    T - number of improvement steps
    TSP_size - the number of points in TSP problem
    n - the number of steps to evaluate the Bellman function
    '''
   
    
    env = LogEnv(n=problem_size, batch_size=batch_size, opts=opts)
    if problem_size == 'random':
        n_range = np.arange(20, 80)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #writer = SummaryWriter()
    
    with torch.no_grad():
        greedy_model = AttentionModel().to(device)
    
    global_iteration = 0
    for e in tqdm(range(epochs), leave=False):
        
        with torch.no_grad():
            greedy_model.load_state_dict(model.state_dict(), strict=False)
        
        for step in range(T):
            if problem_size == 'random':
                n = np.random.choice(n_range)
                env = LogEnv(n=n, batch_size=batch_size, opts=opts)
            global_iteration += 1
            features, distances, mask, context = env.reset()
            features = torch.Tensor(features).to(device)
            context = torch.Tensor(context).to(device)
            flag_done = False
            t = 0
            log_prob_seq = []
            while not flag_done:
                v, p = model(features, mask, t, context, flags=env._flags)
                v = v.to('cpu')
                log_prob_seq.append(torch.log(p))
                with torch.no_grad():
                    mask, flag_done, context = env.step(v)
                    context = torch.Tensor(context).to(device)
                t += 1
            log_prob_tensor = torch.stack(log_prob_seq, 1).squeeze()
            route = torch.tensor(env._cur_route, dtype=int)
            mask = check_repeating_vertexes(route)
            if env._flags['demand'] and not env._flags['pd']:
                log_prob_tensor[mask[:, 1:] == 0] = 0
            else:
                log_prob_tensor[mask == 0] = 0
            with torch.no_grad():
                routes_length = path_distance_new(distances, route)
                if env._flags['tw']:
                    routes_length += check_missing_vertexes(route, env._flags, opts)*np.sqrt(2)
                
            
            
            if e > 0:
                env.set_parameters()
                mask = env.init_mask()
                context = env.set_context()
                context = torch.Tensor(context).to(device)
                t = 0
                flag_done = False
                while not flag_done:
                    v, _ = greedy_model(features, mask, t, context, sample=False, flags=env._flags)
                    v = v.to('cpu')
                    with torch.no_grad():
                        mask, flag_done, context = env.step(v)
                        context = torch.Tensor(context).to(device)
                    t += 1
                gr_route = torch.tensor(env._cur_route, dtype=int)
                greedy_routes_length = path_distance_new(distances, gr_route)
                if env._flags['tw']:
                    greedy_routes_length += check_missing_vertexes(gr_route, env._flags, opts)*2*np.sqrt(2)
                with torch.no_grad():
                    delta = torch.Tensor(routes_length - greedy_routes_length).to(device)
                    
            else:
                with torch.no_grad():
                    if global_iteration == 1:
                        M = routes_length
                    else:
                        M = decay*M + (1 - decay)*routes_length
                    delta = torch.Tensor(routes_length - M).to(device)
            
            loss = (delta.squeeze()*log_prob_tensor.sum(dim=1).squeeze()).mean()
            optimizer.zero_grad()
            loss.backward()
            #writer.add_scalar('Loss', loss, global_iteration)
            #writer.add_scalar('Sample Reward', -routes_length.mean(), global_iteration)
            #writer.add_scalar('Greedy Reward', -greedy_routes_length.mean(), global_iteration)
            optimizer.step()
    problem_name = parse_flags(env._flags)
    time_point = time.asctime()[4:].replace(':', '_').replace(' ', '_')
    file_name = problem_name + '_' + str(problem_size) + '_' + time_point
    return model.state_dict(), file_name