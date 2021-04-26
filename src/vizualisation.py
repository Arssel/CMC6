import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
import torch

from IPython.display import clear_output

def env_plot(env):
    features = env.features
    pairs = env.pairs
    nodes = features[0][:, :2].squeeze()
    tw = features[0][:, 3:5].squeeze()
    demand = features[0][:, 2].squeeze()
    n = env.n
    fig = plt.figure(figsize=(14, 8))
    plt.xlim = (0, 1)
    plt.ylim = (0, 1)
    plt.grid()
    plt.plot()
    plt.title('demonstration')
    plt.scatter(nodes[1:, 0], nodes[1:, 1], c='blue', marker='o', s=200)
    plt.scatter(nodes[:1, 0], nodes[:1, 1], c='red', marker='o', s=200)
    for i in range(1, n + 1):
        plt.annotate('[' + "{:0.2f}".format(tw[i][0]) + ',' + "{:0.2f}".format(tw[i][1]) + ']',
                     (nodes[i, 0] + 0.025, nodes[i, 1]))
        plt.annotate(demand[i].numpy(), (nodes[i, 0] - 0.025, nodes[i, 1] + 0.03))
    for i in range(n // 2):
        l_1 = (nodes[pairs[i, 0], 0], nodes[pairs[i, 0], 1])
        l_2 = (nodes[pairs[i, 1], 0], nodes[pairs[i, 1], 1])
        plt.arrow(l_1[0], l_1[1], l_2[0] - l_1[0], l_2[1] - l_1[1], head_width=0.02, length_includes_head=True)

def demonstration(env, model, device):
    features, distances, mask = env.reset(full_reset=False)
    features = list(map(lambda x: None if x is None else x.to(device), features))
    flag_done = False
    t = 0
    log_prob_seq = []
    while not flag_done:
        v, _ = model(features, mask, t, sample=False)
        v = v.to('cpu')
        with torch.no_grad():
            features, mask, flag_done = env.step(v)
            features = list(map(lambda x: None if x is None else x.to(device), features))
        t += 1

    features = list(map(lambda x: None if x is None else x.to('cpu'), features))    
        
    tour_plan = env.tour_plan.squeeze()   
    route_plan = env.tour_plan.squeeze().view(-1)

    nodes = features[0][:, :, :2].squeeze()
    tw = features[0][:, :, 3:5].squeeze()
    demand = features[0][:, :, 2].squeeze()
    n = env.n
    fig = plt.figure(figsize=(14, 8))
    plt.xlim=(0, 1)
    plt.ylim=(0, 1)
    plt.grid()
    plt.plot()
    plt.title('demonstration')
    plt.scatter(nodes[1:, 0], nodes[1:, 1], c='blue', marker='o', s=200)
    plt.scatter(nodes[:1, 0], nodes[:1, 1], c='red', marker='o', s=200)
    for i in range(1, n+1):
        plt.annotate('['+ "{:0.2f}".format(tw[i][0]) +','+ "{:0.2f}".format(tw[i][1]) + ']' , (nodes[i,0]+0.025, nodes[i,1]))
        plt.annotate(demand[i].numpy(), (nodes[i,0]-0.025, nodes[i,1]+0.03))
    for k in range(10):
        t = tour_plan[k][tour_plan[k]!=0]
        plt.plot(nodes[t, 0], nodes[t, 1], c=np.random.rand(3))
    n_set = set(list(np.arange(n+1)))
    act_set = set(list(route_plan.squeeze().view(-1).numpy()))
    difference = list(map(lambda x: str(x), list(n_set - act_set)))
    if len(difference) == 0:
        print('not attended vertexes: None')
    else:
        print('not attended vertexes: ' + ','.join(difference))