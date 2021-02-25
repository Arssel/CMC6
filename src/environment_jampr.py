import numpy as np
import torch


def pairwise_distance(X, p):
    if p == 'max':
        dist = (torch.abs(X[:,:,:,None] - X.permute(0,2,1)[:, None, :, :])).max(dim=2)
    else:
        dist = (torch.abs(X[:,:,:,None] - X.permute(0,2,1)[:, None, :, :])**p).sum(dim=2)**(1/p)
    return dist


class LogEnv:
    def __init__(self, n=20, batch_size=16, K=10, active_num=3):
        super().__init__()
        self.n = n
        self.bsz = batch_size
        self.k = K
        self.act_num = active_num

    def reset(self, full_reset=True):
        if full_reset:
            location = torch.rand((self.bsz, self.n + 1, 2))
            self.location = location
            distance = pairwise_distance(location, 2)*100
            distance[distance > 0] += 10
            self.distance = distance
            demand = torch.floor(torch.abs(torch.normal(mean=15, std=10, size=(self.bsz, self.n + 1, 1))))
            demand[demand > 42] = 42
            demand[demand < 1] = 1
            demand[:, 0, :] = 0
            if self.n == 20:
                self.capacity = 500
            elif self.n == 50:
                self.capacity = 750
            else:
                self.capacity = 300
            demand = demand/self.capacity
            self.demand = demand
            h = (torch.ceil(distance[:, :, 0])+1)
            a = torch.zeros((self.bsz, self.n + 1, 1))
            b = torch.zeros((self.bsz, self.n + 1, 1))
            permutation = torch.randperm(self.n) + 1
            pairs = permutation.reshape(-1, 2)
            for i in range(self.bsz):
                for j in range(0, int((self.n + 1)/2)):
                    a[i, pairs[j, 0]] = torch.rand((1,1))*(1000 - 6 - distance[i, pairs[j, 0], pairs[j, 1]]
                        - h[i,pairs[j, 1]] - h[i,pairs[j, 0]]) + h[i,pairs[j, 0]]
                    eps_1 = torch.abs(torch.normal(mean=torch.Tensor([0]), std=torch.Tensor([1])))
                    if eps_1 > 1/100:
                        eps_1 = 1/100
                    b[i, pairs[j, 0]] = a[i, pairs[j, 0]] + eps_1*300
                    if b[i, pairs[j, 0]] > 1000 - h[i, pairs[j, 0]]:
                        b[i, pairs[j, 0]] = 1000 - h[i, pairs[j, 0]]
                    a[i, pairs[j, 1]] = torch.rand((1, 1)) * (1000 - 3 - h[i, pairs[j, 1]] - b[i, pairs[j, 0]]
                                                              - distance[i, pairs[j, 0], pairs[j, 1]]) \
                                        + b[i, pairs[j, 0]] + distance[i, pairs[j, 0], pairs[j, 1]]
                    eps_2 = torch.abs(torch.normal(mean=torch.Tensor([0]), std=torch.Tensor([1])))
                    if eps_2 > 1/100:
                        eps_2 = 1/100
                    b[i, pairs[j, 1]] = a[i, pairs[j, 1]] + eps_2*300
                    if b[i, pairs[j, 1]] > 1000 - h[i, pairs[j, 1]]:
                        b[i, pairs[j, 1]] = 1000 - h[i, pairs[j, 1]]
            a[:, 0] = 0
            b[:, 0] = 1000
            tw = torch.cat((a,b), dim=2)
            self.tw = tw
            pd_locations = torch.zeros(size=(self.bsz, self.n+1, 2))
            pd_locations[:, pairs[:, 1]] = location[:, pairs[:, 0]]
            pd_locations[:, pairs[:, 0]] = location[:, pairs[:, 1]]
            pd_tw = torch.zeros(size=(self.bsz, self.n+1, 2))
            pd_tw[:, pairs[:, 1]] = tw[:, pairs[:, 0]]
            pd_tw[:, pairs[:, 0]] = tw[:, pairs[:, 1]]
            self.pd_tw = pd_tw
            self.pd_locations = pd_locations
            self.demand[:, pairs[:, 0]] = self.demand[:, pairs[:, 1]]*(-1)
            features = torch.cat((location, demand, tw, pd_locations), dim=2)
            self.pairs = pairs
            self.features = features
        self.time_step = 0
        self.tour_plan = torch.zeros((self.bsz, self.k, 20), dtype=torch.int64)
        self.act_vind = torch.zeros((self.bsz, self.act_num, self.n + 1), dtype=torch.int64)
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
        features_return = [self.features, self.tour_plan, self.vehicle, act, torch.zeros((1)), last]
        return features_return, self.distance.clone(), mask.reshape(-1, self.act_num*(self.n +1))

    def init_vehicle(self):
        vehicle = torch.zeros((self.bsz, self.k, 5))
        vehicle[:, :, 0] = torch.arange(0, self.k).view(1, -1) / self.k
        vehicle[:, :, 1] = 0
        vehicle[:, :, 2:4] = self.location[:, 0, :2].unsqueeze(1)
        vehicle[:, :, 4] = 0
        return vehicle

    def init_mask(self):
        self.mask_visited = torch.ones((self.bsz, self.k, self.n + 1), dtype=torch.bool)
        self.mask_tw = torch.ones((self.bsz, self.k, self.n + 1), dtype=torch.bool)
        self.mask_demand = torch.ones((self.bsz, self.k, self.n + 1), dtype=torch.bool)
        self.mask_pd = torch.ones((self.bsz, self.k, self.n + 1), dtype=torch.bool)
        self.mask_pd[:, :, self.pairs[:, 1]] = 0
        return (self.mask_pd*self.mask_demand*self.mask_tw*self.mask_visited).gather(index=self.act_vind, dim=1)
            
    def step(self, actions):
        r = torch.arange(0, self.bsz, dtype=torch.int64)
        k = self.act_vind[r, actions[:, 0], 0]
        i = actions[:, 1]
        self.tour_plan[r, k, self.tour_plan[r, k, :].argmin(dim=1)] = i
        self.vehicle[r, k, 2:4] = self.location[r, i, :]
        self.vehicle[r, k, 1] = self.distance[r, 0, i]
        d_t = self.vehicle[r, k, -1] + self.distance[r, self.last_location, i]
        d_f = (d_t <= self.tw[r, i, 0])
        self.last_location = i
        self.vehicle[r, k, -1] = d_f.type(dtype=torch.float)*self.tw[r, i, 0] + (~d_f).type(dtype=torch.float)*d_t
        self.fillness[r, k] -= self.demand[r, i, 0]
        if self.bsz > 1:
            pd_start_mask = i.reshape(-1, 1) == self.pairs[:, 0].reshape(1, -1)
            pd_i = pd_start_mask.type(dtype=torch.float).argmax(axis=1)
            pd_r = pd_start_mask.sum(axis=1).type(dtype=torch.bool)
            if pd_r.sum() > 0:
                self.mask_pd[pd_r, k[pd_r], self.pairs[pd_i[pd_r], 1]] = 1
        else:
            if i in self.pairs[:, 0]:
                pd_i = (self.pairs[:, 0] == i).type(dtype=torch.float).argmax()
                self.mask_pd[:, k, self.pairs[pd_i, 1]] = 1
        r_ind = r[i == 0]
        r_ind_other = r[i != 0]
        if len(r_ind) != 0:
            self.mask_visited[r_ind, k[r_ind], i[r_ind]] = 0
        if len(r_ind_other) != 0:
            self.mask_visited[r_ind_other, :, i[r_ind_other]] = 0
        r_ind = r[i == 0]
        if len(r_ind) != 0:
            self.mask_visited[r_ind, k[r_ind], 1:] = 0
        self.mask_tw[r, k, :] = (self.vehicle[r, k, -1].reshape(-1, 1) + self.distance[r, i, :]) <= self.tw[:, :, 1]
        future_weight = self.fillness[r, k].reshape(-1, 1) - self.demand[r, :, 0]
        self.mask_demand[r, k, :] = (-1e-5 <= future_weight.type(dtype=torch.float)) & (future_weight.type(dtype=torch.float) <= 1 + 1e+5)
        mask = (self.mask_pd*self.mask_demand * self.mask_tw * self.mask_visited).gather(index=self.act_vind, dim=1)
        if (mask.sum(dim=2).sum(dim=1).squeeze() == 0).any():
            r_ind = r[mask.sum(dim=2).sum(dim=1).squeeze() == 0]
            self.mask_visited[r_ind, :, 0] = 1
        total_mask = (self.mask_pd*self.mask_demand*self.mask_tw*self.mask_visited)
        if (total_mask[r, k, 1:].sum(dim=1) == 0).any():
            batch_to_change_mask = total_mask[r, k, 1:].sum(dim=1) == 0
            self.vehicle[batch_to_change_mask, k[batch_to_change_mask], -1] \
                += self.distance[batch_to_change_mask, i[batch_to_change_mask], 0]
            batch_to_change = r[batch_to_change_mask]
            last_vehicle = self.acted_v.argmin(dim=1)
            last_not_null = last_vehicle != 0
            self.acted_v[batch_to_change, last_vehicle[batch_to_change_mask]] = 1
            total_batch_mask = batch_to_change_mask*last_not_null
            batch_total = r[total_batch_mask]
            self.act_vind[batch_total, actions[:, 0][total_batch_mask], :] = last_vehicle[total_batch_mask].view(-1, 1)
        if (total_mask[:, :, 1:].sum(dim=2) == 0).all():
            flag_done = True
        else:
            flag_done = False
        self.time_step += 1
        total_mask = (self.mask_pd*self.mask_demand * self.mask_tw * self.mask_visited).gather(index=self.act_vind, dim=1)
        act = self.act_vind[:, :, 0]
        last = i
        features = [self.features, self.tour_plan, self.vehicle, act, k, last]
        return features, total_mask.reshape(-1, self.act_num*(self.n +1)), flag_done
