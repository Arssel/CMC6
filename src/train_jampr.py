from src.architecture_jampr_2 import AttentionModel
from src.environment_jampr import LogEnv
from src.utils import path_distance_jampr, check_missing_vertexes_jampr

import numpy as np
import torch
import torch.optim as optim
import time

from tqdm import tqdm


#from torch.utils.tensorboard import SummaryWriter

def adjust_learning_rate(optimizer, epoch, lr, decay):
    lr_new = 1/(1+epoch*decay)*lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

def train(model, device="cuda", batch_size=256, epochs=100, T=40, lr=1e-4, problem_size=20, decay=0.001):
    env = LogEnv(n=problem_size, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #writer = SummaryWriter()

    with torch.no_grad():
        greedy_model = AttentionModel().to(device)

    global_iteration = 0
    loss_list = []
    reward_list = []
    for e in tqdm(range(epochs), leave=False):

        lr = adjust_learning_rate(optimizer, e, lr, decay)

        with torch.no_grad():
            greedy_model.load_state_dict(model.state_dict(), strict=False)

        for step in range(T):
            global_iteration += 1
            features, distances, mask = env.reset()
            features = list(map(lambda x: None if x is None else x.to(device), features))
            flag_done = False
            t = 0
            log_prob_seq = []
            while not flag_done:
                v, p = model(features, mask, t,)
                v = v.to('cpu')
                log_prob_seq.append(torch.log(p))
                with torch.no_grad():
                    features, mask, flag_done = env.step(v)
                    features = list(map(lambda x: None if x is None else x.to(device), features))
                t += 1
            log_prob_tensor = torch.stack(log_prob_seq, 1).squeeze(1)
            routes_length = path_distance_jampr(distances, env.tour_plan)
            routes_length += check_missing_vertexes_jampr(env.tour_plan, problem_size) * 100

            if e > 0:
                features, distances, mask = env.reset(full_reset=False)
                features = list(map(lambda x: None if x is None else x.to(device), features))
                flag_done = False
                t = 0
                while not flag_done:
                    v, _ = greedy_model(features, mask, t, sample=False)
                    v = v.to('cpu')
                    with torch.no_grad():
                        features, mask, flag_done = env.step(v)
                        features = list(map(lambda x: None if x is None else x.to(device), features))
                    t += 1
                greedy_routes_length = path_distance_jampr(distances, env.tour_plan)
                greedy_routes_length += check_missing_vertexes_jampr(env.tour_plan, problem_size) * 100
                with torch.no_grad():
                    delta = torch.Tensor(routes_length - greedy_routes_length).to(device)
            else:
                with torch.no_grad():
                    if global_iteration == 1:
                        M = routes_length
                    else:
                        M = decay * M + (1 - decay) * routes_length
                    delta = routes_length - M
                    delta = delta.to(device)

            loss = (delta.squeeze() * log_prob_tensor.sum(dim=1).squeeze()).mean()
            loss_list.append(loss.detach().to('cpu'))
            reward_list.append(routes_length.mean())
            optimizer.zero_grad()
            # print(env._cur_route[0])
            loss.backward()
            #writer.add_scalar('Loss', loss, global_iteration)
            #writer.add_scalar('Sample Reward', -routes_length.mean(), global_iteration)
            #writer.add_scalar('Greedy Reward', -greedy_routes_length.mean(), global_iteration)
            optimizer.step()
    print(reward_list)
    time_point = time.asctime()[4:].replace(':', '_').replace(' ', '_')
    file_name = 'JAMPR_TW1_' + str(problem_size) + '_' + time_point
    return model.state_dict(), file_name, loss_list, reward_list