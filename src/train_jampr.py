from src.environment_jampr_real import LogEnv
from src.utils import path_distance_jampr, check_missing_vertexes_jampr

import torch
import torch.optim as optim
import time
import copy
import numpy as np

from tqdm import tqdm
import pickle
#from torch.utils.tensorboard import SummaryWriter

def adjust_learning_rate(optimizer, epoch, lr, decay):
    lr_new = 1/(1+epoch*decay)*lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

def train(model, device="cuda", batch_size=256, epochs=100, T=40, lr=1e-4, problem_size=20, decay=0.001,
          penalty_num_vertexes=1, penalty_num_vehicles=0, num_vehicles=10, save_inbetween=False, output=None, r=None):
    if not type(problem_size) is str:
        env = LogEnv(n=problem_size, batch_size=batch_size, active_num=1, K=num_vehicles)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #writer = SummaryWriter()
    time_point = time.asctime()[4:].replace(':', '_').replace(' ', '_')
    file_name = 'jampr_mod_real_' + str(problem_size) + '_' + time_point
    global_iteration = 0
    loss_list = []
    reward_list = []
    best_reward = 1e10
    best_weights = None
    for e in tqdm(range(epochs), leave=False):
        lr = adjust_learning_rate(optimizer, e, lr, decay)

        for step in range(T):
            global_iteration += 1
            if type(problem_size) is str:
                rand_pr_size = np.random.choice(np.arange(1, 151))
                env = LogEnv(n=rand_pr_size, batch_size=batch_size, active_num=1, K=num_vehicles)
            features, distances, mask = env.reset(r=r)
            features[0] = features[0].to(device)
            features[1] = features[1].to(device)
            flag_done = False
            t = 0
            log_prob_seq = []
            precomputed = None
            while not flag_done:
                v, p, precomputed = model(features, mask, t, precomputed, True)
                v = v.to('cpu')
                log_prob_seq.append(torch.log(p))
                features, mask, flag_done = env.step(v)
                features[1] = features[1].to(device)
                t += 1
            log_prob_tensor = torch.stack(log_prob_seq, 1).squeeze(1)
            routes_length = env.vehicle[:, :, -1].sum(dim=1).to('cpu')
            routes_length += check_missing_vertexes_jampr(env.tour_plan, problem_size) * penalty_num_vertexes
            routes_length += (env.tour_plan.sum(dim=2) > 0).type(dtype=torch.float).sum(dim=1)*float(penalty_num_vehicles)
            mf = (env.vehicle[:, :, -1].sum(dim=1).to('cpu') +
                   check_missing_vertexes_jampr(env.tour_plan, problem_size) * penalty_num_vertexes).mean()


            with torch.no_grad():
                if global_iteration == 1:
                    M = routes_length
                else:
                    M = 0.9 * M + (1 - 0.9) * routes_length
                delta = routes_length - M
                delta = delta.to(device)

            loss = (delta.squeeze() * log_prob_tensor.sum(dim=1).squeeze()).mean()
            loss_list.append(loss.detach().to('cpu').item())
            reward_list.append(routes_length.mean().detach().item())

            if mf < best_reward:
                best_weights = copy.deepcopy(model.state_dict())
                best_reward = mf
                if save_inbetween:
                    f = open(output + file_name + '_best_on_iteration.pkl', 'wb')
                    pickle.dump(best_weights, f)
                    f.close()
                    f = open(output + file_name + '_loss_on_iteration.pkl', 'wb')
                    pickle.dump(loss_list, f)
                    f.close()
                    f = open(output + file_name + '_reward_on_iteration.pkl', 'wb')
                    pickle.dump(reward_list, f)
                    f.close()
                #print(best_reward)
            optimizer.zero_grad()
            # print(env._cur_route[0])
            loss.backward()
            #writer.add_scalar('Loss', loss, global_iteration)
            #writer.add_scalar('Sample Reward', -routes_length.mean(), global_iteration)
            #writer.add_scalar('Greedy Reward', -greedy_routes_length.mean(), global_iteration)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            del loss, log_prob_tensor
        #print(env.tour_plan[0])
    #print(reward_list)
    return best_weights, model.state_dict(), file_name, loss_list, reward_list
