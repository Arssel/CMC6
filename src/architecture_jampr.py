import torch
import torch.nn as nn

import numpy as np


class EncoderLayer(nn.Module):
    def __init__(self, dm=128, num_heads=8, ff_dim=512):
        super().__init__()
        self.hid_dim = dm
        self.num_head = num_heads
        self.head_dim = dm // num_heads
        self.lin_q = nn.Linear(dm, dm)
        self.lin_k = nn.Linear(dm, dm)
        self.lin_v = nn.Linear(dm, dm)
        self.lin_o = nn.Linear(dm, dm)
        self.batch_norm_1 = nn.BatchNorm1d(dm)
        self.batch_norm_2 = nn.BatchNorm1d(dm)
        self.ff = nn.Sequential(nn.Linear(dm, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dm))

    def forward(self, x):
        bsz, gr_size, _ = x.shape
        V = self.lin_v(x)  # V.shape == (batch_size, n, dv)
        K = self.lin_k(x)  # K.shape == (batch_size, n, dk)
        Q = self.lin_q(x)  # Q.shape == (batch_size, n, dq)

        K = K.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        Q = Q.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=float)).to(x.device)
        QK_norm = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        QK_sm = torch.softmax(QK_norm, dim=-1)

        QKV = torch.matmul(QK_sm, V)
        QKV = QKV.permute(0, 2, 1, 3).contiguous()
        QKV = QKV.view(bsz, -1, self.hid_dim)
        QKV_o = self.lin_o(QKV)

        x = QKV_o + x
        x = self.batch_norm_1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.ff(x) + x
        x = self.batch_norm_2(x.transpose(-1, -2)).transpose(-1, -2)
        return x


class Encoder(nn.Module):
    def __init__(self, dm=128, num_heads=8, ff_dim=512, N=3):
        super().__init__()
        self.projection = nn.Linear(5, dm)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dm=dm, num_heads=num_heads, ff_dim=ff_dim) for _ in range(N)])

    def forward(self, node):
        node = self.projection(node)
        for module in self.encoder_layers:
            node = module(node)
        return node, node.mean(dim=1).squeeze(1)


class Decoder(nn.Module):
    def __init__(self, en_dm=128, dec_dm=256, num_heads=8):
        super().__init__()
        self.hid_dim = dec_dm
        self.num_head = num_heads
        self.head_dim = int(dec_dm // num_heads)
        self.lin_q = nn.Linear(5 * en_dm, dec_dm)
        self.lin_o = nn.Linear(dec_dm, dec_dm)
        self.infty = torch.tensor(-1e15)
        self.lin_veh = nn.Linear(en_dm, dec_dm)
        self.lin_nodes_veh = nn.Linear(en_dm + 1, dec_dm)
        self.lin_k = nn.Linear(dec_dm, dec_dm)
        self.lin_v = nn.Linear(dec_dm, dec_dm)
        self.lin_lg = nn.Linear(dec_dm, dec_dm)

    def forward(self, tup, mask, t, sample=True, precomputed=None):
        h, h_g, vehicle, fleet, act, last, act_num = tup
        bsz = h.shape[0]
        graph_size = h.shape[1]
        k = vehicle.shape[1]
        h_proj = precomputed
        h_c = torch.cat((h_g, fleet, act, h[:, 0, :].squeeze(1), last), dim=1)
        h_c = h_c.unsqueeze(1)
        Q = self.lin_q(h_c)

        veh_proj = self.lin_veh(vehicle)

        mm = torch.matmul(vehicle, h.permute(0, 2, 1)).reshape(bsz, graph_size * k, -1)
        pwm = (vehicle[:, :, :, None] * (h.permute(0, 2, 1).unsqueeze(1))).permute(0, 1, 3, 2).reshape(bsz,
                                                                                                         graph_size * k,
                                                                                                         -1)
        g_a = self.lin_nodes_veh(torch.cat((mm, pwm), dim=2)).reshape(bsz, k, graph_size, -1)
        g_a = g_a + h_proj[:, None, :, :] + veh_proj[:, :, None, :]
        g_a = g_a.reshape(bsz, graph_size * k, -1)
        K = self.lin_k(g_a)
        V = self.lin_v(g_a)
        K_lg = self.lin_lg(g_a)

        mask_for_transform = mask[:, None, None, :].expand(bsz, self.num_head, 1, act_num*graph_size)
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float)).to(h.device)
        Q = Q.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        QK_norm = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        QK_norm[mask_for_transform == 0] = self.infty.to(h.device)
        QK_sm = torch.softmax(QK_norm, dim=-1)
        u = torch.matmul(QK_sm, V)
        u = u.permute(0, 2, 1, 3)
        u = u.view(bsz, 1, self.hid_dim)
        u = self.lin_o(u)
        u = torch.matmul(u, K_lg.permute(0, 2, 1)). \
                view(bsz, k * graph_size) / torch.sqrt(torch.tensor(self.hid_dim, dtype=torch.float)).to(h.device)
        u = torch.tensor(10).to(h.device) * torch.tanh(u)
        w_t_m = (mask == 0)
        u[w_t_m] = self.infty.to(h.device)
        u = torch.softmax(u, dim=1)

        if sample:
            vertexes = u.multinomial(1)
        else:
            vertexes = u.argmax(dim=1).unsqueeze(1)
        probs = u.gather(1, vertexes).view(-1, 1)

        k = vertexes // graph_size
        i = vertexes % graph_size

        return torch.cat((k, i), dim=1), probs


class AttentionModel(nn.Module):
    def __init__(self, en_dm=128, dec_dm=256, veh_dim=64, num_heads=8, ff_dim=512, N=3, active_num=3):
        super().__init__()
        self.act_num = active_num
        self.en_dm = en_dm
        self.dec_dim = dec_dm
        self.head_dim = dec_dm // num_heads
        self.num_head = num_heads
        self.encoder = Encoder(dm=en_dm, num_heads=num_heads, ff_dim=ff_dim, N=N)
        self.decoder = Decoder(en_dm=en_dm, dec_dm=dec_dm, num_heads=num_heads)
        self.lin_k = nn.Linear(en_dm, dec_dm)
        self.lin_k_lg = nn.Linear(en_dm, dec_dm)
        self.lin_v = nn.Linear(en_dm, dec_dm)
        self.veh_emb = nn.Linear(5, en_dm)
        self.veh_1 = nn.Linear(en_dm, veh_dim)
        self.veh_2 = nn.Linear(veh_dim, veh_dim)
        self.veh_3 = nn.Linear(veh_dim, veh_dim)
        self.tour_1 = nn.Linear(en_dm, veh_dim)
        self.tour_2 = nn.Linear(veh_dim, veh_dim)

    def precompute(self, h):
        bsz = h.shape[0]
        K = self.lin_k(h)
        return (K)

    def forward(self, features, mask, t, sample=True):
        with torch.autograd.set_detect_anomaly(True):
            tour_plan = features[1]
            L = tour_plan.shape[2]
            K = tour_plan.shape[1]
            bsz = tour_plan.shape[0]
            vehicle_features = features[2]
            act = features[3]
            k = features[4]
            last = features[5]
            r = torch.arange(bsz, dtype=torch.int64)
            n = features[0].shape[1]
            if t == 0:
                node = features[0]
                h, h_g = self.encoder(node)
                self.precomputed = self.precompute(h)
                self.h, self.h_g = h, h_g
                vehicle_emb = self.veh_emb(vehicle_features)
                self.vehicle = self.veh_3(torch.sigmoid(self.veh_2(torch.sigmoid(self.veh_1(vehicle_emb)))))
                node_emb = self.tour_2(torch.sigmoid(self.tour_1(self.h[:, 0, :])))
                node_emb = node_emb[:, None, :]
                node_emb = torch.repeat_interleave(node_emb, K, dim=1)/L
                self.vehicle = torch.cat((self.vehicle, node_emb), dim=2)
            else:
                vehicle_emb = vehicle_features[r, k, :].clone()
                vehicle_emb = self.veh_emb(vehicle_emb)
                vehicle_k = self.veh_3(torch.sigmoid(self.veh_2(torch.sigmoid(self.veh_1(vehicle_emb)))))
                plan_ind = tour_plan[r, k, :]
                plan_ind_mask = (plan_ind != 0)
                plan_ind_mask = plan_ind_mask[:, :, None]
                nodes_ind = torch.repeat_interleave(plan_ind[:, :, None], self.en_dm, dim=2)
                nodes = self.h.gather(index=nodes_ind, dim=1)
                nodes = self.tour_2(torch.sigmoid(self.tour_1(nodes)))
                vehicle_k = torch.cat((vehicle_k, (nodes * plan_ind_mask.type(dtype=torch.float)).sum(dim=1) / L), dim=1)
                v = self.vehicle.clone()
                v[np.arange(bsz), k, :] = vehicle_k
                self.vehicle = v
            fleet = self.vehicle.mean(dim=1)
            act = torch.repeat_interleave(act[:, :, None], self.en_dm, dim=2)
            vehicle_act = self.vehicle.gather(index=act, dim=1)
            act = vehicle_act.mean(dim=1)
            last = self.h[r, last, :].view(bsz, -1)
            tup = (self.h, self.h_g, vehicle_act, fleet, act, last, self.act_num)
            return self.decoder(tup, mask, t, sample, self.precomputed)
