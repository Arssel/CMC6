from src.architecture import Encoder, Decoder
from src.environment import LogEnv
from src.utils import path_distance, check_repeating_vertexes

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def adjust_learning_rate(optimizer, epoch, lr, decay):
    lr_new = lr * (decay ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new

def train(encoder, decoder, opts, device="cpu", batch_size=256, epochs=200, T=40, lr=1e-4, problem_size=20):
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
    criteria = torch.nn.MSELoss() 
    
    env = LogEnv(n=problem_size, batch_size=batch_size, opts=opts)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params)

    writer = SummaryWriter()
    
    with torch.no_grad():
        greedy_encoder = Encoder().to(device)
        greedy_decoder = Decoder().to(device)
    
    global_iteration = 0
    for e in tqdm(range(epochs), leave=False):
        
        greedy_encoder.load_state_dict(encoder.state_dict(), strict=False)
        greedy_decoder.load_state_dict(decoder.state_dict(), strict=False)
        
        for step in range(T):
            global_iteration += 1
            t = 0
            features, distances, mask = env.reset()
            features = torch.Tensor(features).to(device)
            flag_done = False
            
            h, h_g  = encoder(features)
            t = 0
            while not flag_done:
                v, p = decoder(h, h_g, mask, t)
                v = v.to('cpu')
                if t == 0:
                    vertexes = v
                    probs = p
                else:
                    vertexes = torch.cat((vertexes, v), dim=1)
                    probs = torch.cat((probs, p), dim=1)
                mask, flag_done = env.step(v)
                t += 1
            routes_length = path_distance(distances, vertexes)
            repeat_mask = check_repeating_vertexes(vertexes)
            probs[repeat_mask == 0] = 1
            log_probs = torch.log(probs)
            
            mask = env.init_mask()
            env.set_parameters()
            
            gr_h, gr_h_g = greedy_encoder(features)
            t = 0
            flag_done = False
            while not flag_done:
                v, _ = greedy_decoder(gr_h, gr_h_g, mask, t, sample=False)
                v = v.to('cpu')
                if t == 0:
                    gr_vertexes = v
                else:
                    gr_vertexes = torch.cat((gr_vertexes, v), dim=1)
                mask, flag_done = env.step(v)
                t += 1
            greedy_routes_length = path_distance(distances, gr_vertexes)
            with torch.no_grad():
                delta = torch.Tensor(routes_length - greedy_routes_length).to(device)
            loss = (delta.squeeze()*log_probs.sum(axis=1).squeeze()).mean()
            loss.backward()
            writer.add_scalar('Loss', loss, global_iteration)
            writer.add_scalar('Sample Reward', -routes_length.mean(), global_iteration)
            writer.add_scalar('Greedy Reward', -greedy_routes_length.mean(), global_iteration)
            optimizer.step()
            optimizer.zero_grad()
    return env