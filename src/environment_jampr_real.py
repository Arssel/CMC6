import numpy as np
import torch
import gym

def pairwise_distance(X, p):
    if p == 'max':
        dist = (torch.abs(X[:,:,:,None] - X.permute(0,2,1)[:, None, :, :])).max(dim=2)
    else:
        dist = (torch.abs(X[:,:,:,None] - X.permute(0,2,1)[:, None, :, :])**p).sum(dim=2)**(1/p)
    return dist


class LogEnv(gym.Env):
    def __init__(self, n=20, batch_size=16, K=10, active_num=3, max_route_len=80, max_window=720):
        super().__init__()
        self.n = n
        self.bsz = batch_size
        self.k = K
        self.act_num = active_num
        self.max_route_len = max_route_len
        self.max_window = max_window

    def reset(self, full_reset=True):
        if full_reset:
            location = torch.rand((self.bsz, self.n + 2, 2))
            self.location = location
            distance = pairwise_distance(location, 1)/(0.4)
            distance[distance > 0] += 15
            distance[:, :, -1] = 0
            distance[:, -1, 0] = 0
            self.distance = distance
            demand = torch.as_tensor(np.random.choice(np.arange(1, 101), replace=True, size=(self.bsz, self.n + 2, 1)))
            demand[:, 0, :] = 0
            demand[:, -1, :] = 0
            self.capacity = 700
            demand = demand/self.capacity
            self.demand = demand
            h = (torch.ceil(distance[:, 1:, 0])+1)
            a = torch.zeros((self.bsz, self.n + 2, 1))
            b = torch.zeros((self.bsz, self.n + 2, 1))
            for i in range(self.bsz):
                for j in range(1, self.n + 1):
                    a[i, j] = torch.rand((1,1))*(self.max_window - h[i,j-1]) + h[i,j-1]
                    b[i, j] = a[i, j] + 120
                    if b[i, j] > self.max_window:
                        b[i,j] = self.max_window
            a[:, 0] = 0
            b[:, 0] = 150
            b[:, -1] = self.max_window
            tw = torch.cat((a,b), dim=2)
            self.tw = tw
            features = torch.cat((location, demand, tw/self.max_window), dim=2)
            self.features = features
        self.time_step = 0
        self.tour_plan = torch.zeros((self.bsz, self.k, self.max_route_len+1), dtype=torch.int64)
        self.act_vind = torch.zeros((self.bsz, self.act_num, self.n + 2), dtype=torch.int64)
        self.acted_v = torch.zeros((self.bsz, self.k), dtype=torch.int64)
        for i in range(self.act_num):
            self.act_vind[:, i, :] = i
            self.acted_v[:, i] = 1
        mask = self.init_mask()
        vehicle = self.init_vehicle()
        self.fillness = torch.zeros((self.bsz, self.k))
        self.vehicle = vehicle
        self.last_location = torch.zeros((self.bsz), dtype=torch.int64)
        act = self.act_vind[:, :, 0]
        last = torch.zeros((self.bsz), dtype=torch.int64)
        features_return = [self.features,
                           self.vehicle.gather(1, act[:, :, None].expand(act.shape[0], act.shape[1], 5)), last]
        return features_return, self.distance.clone(), mask.reshape(-1, self.act_num*(self.n + 2))

    def init_vehicle(self):
        vehicle = torch.zeros((self.bsz, self.k, 5))
        vehicle[:, :, 0] = torch.arange(0, self.k).view(1, -1) / self.k
        vehicle[:, :, 1] = 0
        vehicle[:, :, 2:4] = self.location[:, 0, :2].unsqueeze(1)
        vehicle[:, :, 4] = 0
        return vehicle

    def init_mask(self):
        self.mask_visited = torch.ones((self.bsz, self.k, self.n + 2), dtype=torch.bool)
        self.mask_visited[:, :, 0] = 0
        self.mask_tw = torch.ones((self.bsz, self.k, self.n + 2), dtype=torch.bool)
        self.mask_demand = torch.ones((self.bsz, self.k, self.n + 2), dtype=torch.bool)
        return (self.mask_demand*self.mask_tw*self.mask_visited).gather(index=self.act_vind, dim=1)
            
    def step(self, actions):
        r = torch.arange(0, self.bsz, dtype=torch.int64)
        k = self.act_vind[r, actions[:, 0], 0]
        i = actions[:, 1]
        if (i == (self.n + 1)).any():
            change_mask = i != self.n + 1
            print(change_mask)
            print(r[change_mask])
            print(k[change_mask])
            if change_mask.sum() > 0:
                self.tour_plan[r[change_mask], k[change_mask],
                           self.tour_plan[r[change_mask], k[change_mask], :].argmin(dim=1)] = i[change_mask]
        else:
            self.tour_plan[r, k, self.tour_plan[r, k, :].argmin(dim=1)] = i
        self.vehicle[r, k, 2:4] = self.location[r, i, :]
        self.vehicle[r, k, 1] = self.distance[r, 0, i]/self.max_window
        d_t = self.vehicle[r, k, -1] + self.distance[r, self.last_location, i]/self.max_window
        self.last_location = i
        d_f = (d_t <= self.tw[r, i, 0]/self.max_window)
        self.vehicle[r, k, -1] = d_f.type(dtype=torch.float)*self.tw[r, i, 0]/self.max_window + (~d_f).type(dtype=torch.float)*d_t
        self.fillness[r, k] += self.demand[r, i, 0]
        print(i)
        print(r)
        print(i == self.n + 1)
        r_ind = r[i == self.n + 1]
        r_ind_other = r[i != self.n + 1]
        if len(r_ind) != 0:
            self.mask_visited[r_ind, k[r_ind], i[r_ind]] = 0
        if len(r_ind_other) != 0:
            self.mask_visited[r_ind_other, :, i[r_ind_other]] = 0
        if len(r_ind) != 0:
            self.mask_visited[r_ind, k[r_ind], :-1] = 0

        self.mask_tw[r, k, :] = (self.vehicle[r, k, -1].reshape(-1, 1) + self.distance[r, i, :]/self.max_window) \
                                    <= self.tw[:, :, 1]/self.max_window
        future_weight = self.fillness[r, k].reshape(-1, 1) + self.demand[r, :, 0]
        self.mask_demand[r, k, :] = future_weight <= 1
        mask = (self.mask_demand * self.mask_tw * self.mask_visited).gather(index=self.act_vind, dim=1)
        if (mask.sum(dim=2).sum(dim=1).squeeze() == 0).any():
            r_ind = r[mask.sum(dim=2).sum(dim=1).squeeze() == 0]
            self.mask_visited[r_ind, :, -1] = 1
        if ((self.tour_plan[r, k, :] == 0).sum(axis=1) == 0).any():
            r_ind = r[(self.tour_plan[r, k, :] == 0).sum(axis=1) == 0]
            self.mask_visited[r_ind, :, -1] = 1
        total_mask = (self.mask_demand*self.mask_tw*self.mask_visited)
        if (total_mask[:, :, :-1].sum(dim=2) == 0).all():
            flag_done = True
        else:
            flag_done = False
        if (total_mask[r, k, :-1].sum(dim=1) == 0).any() and not flag_done:
            batch_to_change_mask = total_mask[r, k, :-1].sum(dim=1) == 0
            batch_to_change = r[batch_to_change_mask]
            last_vehicle = self.acted_v.argmin(dim=1)
            last_not_null = last_vehicle != 0
            self.acted_v[batch_to_change, last_vehicle[batch_to_change_mask]] = 1
            total_batch_mask = batch_to_change_mask*last_not_null
            batch_total = r[total_batch_mask]
            self.act_vind[batch_total, actions[:, 0][total_batch_mask], :] = last_vehicle[total_batch_mask].view(-1, 1)
        self.time_step += 1
        total_mask = (self.mask_demand * self.mask_tw * self.mask_visited).gather(index=self.act_vind, dim=1)
        act = self.act_vind[:, :, 0]
        last = i
        features = [torch.zeros((1)),
                    self.vehicle.gather(1, act[:, :, None].expand(act.shape[0], act.shape[1], 5)), last]
        return features, total_mask.reshape(-1, self.act_num*(self.n + 2)), flag_done