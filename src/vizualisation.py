import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera
import torch

from IPython.display import clear_output

def demonstration(env, opts, actions_ext=None):
    if opts is None:
        opts = {}
    if 'graph_type' not in opts:
        opts['graph_type'] = {}
    if 'pickup_and_delivery' not in opts:
        opts['pickup_and_delivery'] = {}
    if 'demand_type' not in opts:
        opts['demand_type'] = {}
    if 'tw_type' not in opts:
        opts['tw_type'] = {}
    if 'supply' not in opts['demand_type']:
        opts['demand_type']['supply'] = {}

    flags = {}
    if opts['pickup_and_delivery'] == True:
        flags['pd'] = True
    else:
        flags['pd'] = False
    if len(opts['demand_type']) == 1:
        flags['demand'] = False
    else:
        flags['demand'] = True
    if len(opts['tw_type']) == 0:
        flags['tw'] = False
    else:
        flags['tw'] = True
    if len(opts['demand_type']['supply']) == 0:
        flags['supply'] = False
    else:
        flags['supply'] = True

    plt.ion()
    if flags['demand'] and not flags['pd']:
        n = env.n + 1
    else:
        n = env.n
    p = env._dots.squeeze()
    if flags['pd']:
        pairs = env._pairs.squeeze()
    if flags['tw']:
        tw = env._tw.squeeze()
    if flags['demand']:
        demand = env._demand.squeeze()
    mask = env.init_mask()
    mask = mask.squeeze()
    env.set_parameters()

    done = False

    actions=np.array(0)
    t = 0
    while not done:
        clear_output()
        fig = plt.figure(figsize=(14, 8))
        plt.xlim=(0, 1)
        plt.ylim=(0, 1)
        if flags['tw'] and flags['demand']:
            print('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()) \
                  + ', current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))
        elif flags['tw']:
            print('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()))
        elif flags['demand']:
            print('current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))

        for i in range(n):
            if flags['tw']:
                plt.annotate('['+ "{:0.2f}".format(tw[i][0]) +','+ "{:0.2f}".format(tw[i][1]) + ']' , (p[i,0]+0.025, p[i,1]))
            if flags['demand']:
                plt.annotate(demand[i], (p[i,0]-0.025, p[i,1]+0.03))
            plt.annotate(i, (p[i,0]-0.005, p[i,1]-0.0005), fontsize=14)
        if flags['pd']:
            for i in range(n//2):
                l_1 = (p[pairs[i,0], 0], p[pairs[i,0], 1])
                l_2 = (p[pairs[i,1], 0], p[pairs[i,1], 1])
                plt.arrow(l_1[0], l_1[1], l_2[0]-l_1[0], l_2[1]-l_1[1], head_width=0.02, length_includes_head=True)
        plt.grid()
        plt.title('Data demonstration')
        plt.scatter(p[(np.arange(n)+1)*mask - 1 >= 0, 0], p[(np.arange(n)+1)*mask - 1 >= 0, 1], c='C2', marker='o', s=200)
        plt.scatter(p[(np.arange(n)+1)*(1-mask) - 1 >= 0, 0], p[(np.arange(n)+1)*(1-mask) - 1 >= 0, 1], c='r', marker='o', s=200)
        if t > 1:
            plt.plot(p[actions, 0], p[actions, 1], c='b')
        plt.draw()
        plt.pause(0.05)
        fig.clf()
        if actions_ext is None:
            action = torch.tensor(int(input()), dtype=int).reshape(-1,1)
        else:
            action = actions_ext[t]
        if t == 0:
            actions = action
        else:
            actions = np.append(actions, action)
        mask, done, _ = env.step(action)
        mask = mask.squeeze()
        t +=1

    actions = env._cur_route.squeeze()
    clear_output()
    fig = plt.figure(figsize=(14, 8))
    plt.xlim=(0, 1)
    plt.ylim=(0, 1)
    if flags['tw'] and flags['demand']:
        print('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()) \
              + ', current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))
    elif flags['tw']:
        print('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()))
    elif flags['demand']:
        print('current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))
    for i in range(n):
        plt.annotate(i, (p[i,0]-0.005, p[i,1]-0.0005), fontsize=14)
    plt.grid()
    plt.title('Data and masking demonstration')
    plt.scatter(p[:, 0], p[:, 1], c='pink', marker='o', s=200)
    plt.plot(p[actions, 0], p[actions, 1], c='black')
    plt.plot((p[actions[-1], 0], p[actions[0], 0]), (p[actions[-1], 1], p[actions[0], 1]), c='black')
    n_set = set(list(np.arange(n)))
    act_set = set(list(actions))
    difference = list(map(lambda x: str(x), list(n_set - act_set)))
    if len(difference) == 0:
        print('not attended vertexes: None')
    else:
        print('not attended vertexes: ' + ','.join(difference))
       
def action_animation(env, opts, actions_ext, file_name, intervals=1):
    if opts is None:
        opts = {}
    if 'graph_type' not in opts:
        opts['graph_type'] = {}
    if 'pickup_and_delivery' not in opts:
        opts['pickup_and_delivery'] = {}
    if 'demand_type' not in opts:
        opts['demand_type'] = {}
    if 'tw_type' not in opts:
        opts['tw_type'] = {}
    if 'supply' not in opts['demand_type']:
        opts['demand_type']['supply'] = {}

    flags = {}
    if opts['pickup_and_delivery'] == True:
        flags['pd'] = True
    else:
        flags['pd'] = False
    if len(opts['demand_type']) == 1:
        flags['demand'] = False
    else:
        flags['demand'] = True
    if len(opts['tw_type']) == 0:
        flags['tw'] = False
    else:
        flags['tw'] = True
    if len(opts['demand_type']['supply']) == 0:
        flags['supply'] = False
    else:
        flags['supply'] = True

    if flags['demand'] and not flags['pd']:
        n = env.n + 1
    else:
        n = env.n
    p = env._dots.squeeze()
    if flags['pd']:
        pairs = env._pairs.squeeze()
    if flags['tw']:
        tw = env._tw.squeeze()
    if flags['demand']:
        demand = env._demand.squeeze()
    mask = env.init_mask()
    mask = mask.squeeze()
    env.set_parameters()        
        
        
    fig = plt.figure(figsize=(14,8))
    ax = plt.axes(xlim=(0, 1), ylim=(0, 1))

    camera = Camera(fig)

    for i in range(len(actions_ext)):
        if flags['tw'] and flags['demand']:
            plt.title('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()) \
                  + ', current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))
        elif flags['tw']:
            plt.title('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()))
        elif flags['demand']:
            plt.title('current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))

        for j in range(n):
            if flags['tw']:
                plt.annotate('['+ "{:0.2f}".format(tw[j][0]) +','+ "{:0.2f}".format(tw[j][1]) + ']' , (p[j,0]+0.025, p[j,1]))
            if flags['demand']:
                plt.annotate(demand[j], (p[j,0]-0.025, p[j,1]+0.03))
            plt.annotate(j, (p[j,0]-0.005, p[j,1]-0.001), fontsize=14)
        if flags['pd']:
            for j in range(n//2):
                l_1 = (p[pairs[j,0], 0], p[pairs[j,0], 1])
                l_2 = (p[pairs[j,1], 0], p[pairs[j,1], 1])
                plt.arrow(l_1[0], l_1[1], l_2[0]-l_1[0], l_2[1]-l_1[1], head_width=0.02, length_includes_head=True)
        plt.grid()
        plt.scatter(p[(np.arange(n)+1)*mask - 1 >= 0, 0], p[(np.arange(n)+1)*mask - 1 >= 0, 1], c='C2', marker='o', s=200)
        plt.scatter(p[(np.arange(n)+1)*(1-mask) - 1 >= 0, 0], p[(np.arange(n)+1)*(1-mask) - 1 >= 0, 1], c='r', marker='o', s=200)
        if i > 0:
            plt.plot(p[actions_ext[:i], 0], p[actions_ext[:i], 1], c='b')
        action = np.array([actions_ext[i]]).reshape(-1,1)
        mask, done, _ = env.step(action)
        mask = mask.squeeze()
        camera.snap()
    
    
    if flags['tw'] and flags['demand']:
        print('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()) \
              + ', current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))
    elif flags['tw']:
        print('current time: ' + '{:0.2f}'.format(env._cur_time.squeeze()))
    elif flags['demand']:
        print('current capacity: ' + '{:0.2f}'.format(env._cur_cap.squeeze()))
    for j in range(n):
        plt.annotate(j, (p[j,0]-0.005, p[j,1]-0.0005), fontsize=14)
    plt.grid()
    plt.title('Data and masking demonstration')
    plt.scatter(p[:, 0], p[:, 1], c='pink', marker='o', s=200)
    plt.plot(p[actions_ext, 0], p[actions_ext, 1], c='black')
    plt.plot((p[actions_ext[-1], 0], p[actions_ext[0], 0]), (p[actions_ext[-1], 1], p[actions_ext[0], 1]), c='black')
    
    camera.snap()
    ani = camera.animate(intervals)
    ani.save(file_name, writer='imagemagick')        
        