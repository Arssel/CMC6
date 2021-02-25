import torch
import torch.nn as nn
import time
import numpy as np

from pytorch_memlab import profile, profile_every

from src.attention_modules import StandardAttention, NystromAttention, PerformerAttention, LinformerAttention


class EncoderLayer(nn.Module):
    def __init__(self, dm=128, num_heads=8, ff_dim=512, attention_type='standard', attention_parameters=None):
        super().__init__()
        self.hid_dim = dm
        self.num_head = num_heads
        self.head_dim = dm // num_heads
        self.lin_q = nn.Linear(dm, dm)
        self.lin_k = nn.Linear(dm, dm)
        self.lin_v = nn.Linear(dm, dm)
        if attention_type == 'standard':
            self.attention = StandardAttention()
        elif attention_type == 'nystrom':
            if attention_parameters is None:
                self.attention = NystromAttention(10, self.head_dim, self.num_head)
            else:
                self.attention = NystromAttention(attention_parameters['num_landmarks'], self.head_dim, self.num_head)
        elif attention_type == 'linformer':
            if attention_parameters is None:
                self.attention = LinformerAttention(seq_len=50)
            else:
                self.attention = LinformerAttention(attention_parameters['seq_len'], dim=dm, heads=num_heads)
        elif attention_type == 'performer':
            self.attention = PerformerAttention(self.head_dim)
        else:
            assert 'error: no such attention'
        self.batch_norm_1 = nn.BatchNorm1d(dm)
        self.batch_norm_2 = nn.BatchNorm1d(dm)
        self.ff = nn.Sequential(nn.Linear(dm, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dm))
        self.lin_o = nn.Linear(dm, dm)

    def forward(self, x):
        bsz, gr_size, _ = x.shape
        V = self.lin_v(x)  # V.shape == (batch_size, n, dv)
        K = self.lin_k(x)  # K.shape == (batch_size, n, dk)
        Q = self.lin_q(x)  # Q.shape == (batch_size, n, dq)

        K = K.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        Q = Q.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)

        QKV = self.attention(Q, K, V)
        QKV_o = self.lin_o(QKV)

        x = QKV_o + x
        x = self.batch_norm_1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.ff(x) + x
        x = self.batch_norm_2(x.transpose(-1, -2)).transpose(-1, -2)
        return x



class Encoder(nn.Module):
    def __init__(self, dm=128, num_heads=8, ff_dim=512, N=3, attention_type='standard', attention_parameters=None):
        super().__init__()
        self.projection = nn.Linear(7, dm)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dm=dm, num_heads=num_heads, ff_dim=ff_dim,
                    attention_type=attention_type, attention_parameters=attention_parameters) for _ in range(N)])

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

    def forward(self, tup, prec, mask, t, sample=True, precomputed=None):
        h, h_g, vehicle, fleet, act, last, act_num = tup
        bsz = h.shape[0]
        graph_size = h.shape[1]
        k = vehicle.shape[1]
        h_c = torch.cat((h_g, fleet, act, h[:, 0, :].squeeze(1), last), dim=1)
        h_c = h_c.unsqueeze(1)
        Q = self.lin_q(h_c)

        K, V, K_lg = prec
        mask_for_transform = mask[:, None, None, :].expand(bsz, self.num_head, 1, act_num*graph_size)
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float)).to(h.device)
        Q = Q.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        K_lg = K_lg.view(bsz, -1, self.hid_dim)

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
        #print(u)
        if sample:
            vertexes = u.multinomial(1)
        else:
            vertexes = u.argmax(dim=1).unsqueeze(1)
        probs = u.gather(1, vertexes).view(-1, 1)

        k = vertexes // graph_size
        i = vertexes % graph_size

        return torch.cat((k, i), dim=1), probs


class AttentionModel(nn.Module):
    def __init__(self, en_dm=128, dec_dm=256, veh_dim=64, num_heads=8, ff_dim=512, N=3, active_num=3,
                 attention_type='standard', attention_parameters=None):
        super().__init__()
        self.act_num = active_num
        self.en_dm = en_dm
        self.dec_dim = dec_dm
        self.head_dim = dec_dm // num_heads
        self.num_head = num_heads
        self.encoder = Encoder(dm=en_dm, num_heads=num_heads, ff_dim=ff_dim, N=N, attention_type=attention_type,
                               attention_parameters=attention_parameters)
        self.decoder = Decoder(en_dm=en_dm, dec_dm=dec_dm, num_heads=num_heads)
        self.veh_emb = nn.Linear(5, veh_dim)
        self.veh_dim = veh_dim
        self.veh_1 = nn.Linear(veh_dim, veh_dim)
        self.veh_2 = nn.Linear(veh_dim, veh_dim)
        self.veh_3 = nn.Linear(veh_dim, veh_dim)
        self.tour_1 = nn.Linear(en_dm, veh_dim)
        self.tour_2 = nn.Linear(veh_dim, veh_dim)
        self.lin_veh = nn.Linear(en_dm, en_dm)
        self.lin_nodes_veh = nn.Linear(en_dm + 1, en_dm)
        self.lin_prec = nn.Linear(en_dm, en_dm)
        self.lin_k = nn.Linear(en_dm, dec_dm)
        self.lin_v = nn.Linear(en_dm, dec_dm)
        self.lin_lg = nn.Linear(en_dm, dec_dm)
        self.relu = nn.ReLU()

    def precompute(self, h):
        bsz = h.shape[0]
        K = self.lin_prec(h)
        return K

    @profile
    def forward(self, features, mask, t, sample=True):
        tour_plan = features[1]
        L = tour_plan.shape[2]
        K_shape = tour_plan.shape[1]
        bsz = tour_plan.shape[0]
        vehicle_features = features[2]
        act = features[3]
        k = features[4]
        last = features[5]
        r = torch.arange(bsz, dtype=torch.int64)
        n = features[0].shape[1]
        if t == 0:
            node = features[0]
            self.h, self.h_g = self.encoder(node)
            self.precomputed = self.precompute(self.h)
            vehicle_emb = self.veh_emb(vehicle_features)
            del vehicle_features
            self.vehicle = self.veh_3(self.relu(self.veh_2(self.relu(self.veh_1(vehicle_emb)))))
            del vehicle_emb
            node_emb = self.tour_2(self.relu(self.tour_1(self.h[:, 0, :])))
            node_emb = node_emb[:, None, :].expand(bsz, K_shape, self.veh_dim)
            self.vehicle = torch.cat((self.vehicle, node_emb/L), dim=2)
            del node_emb
            self.vehicle_proj = self.lin_veh(self.vehicle)
            mm = torch.matmul(self.vehicle, self.h.permute(0, 2, 1)).unsqueeze(-1)
            pwm = (self.vehicle[:, :, :, None] * (self.h.permute(0, 2, 1).unsqueeze(1))).permute(0, 1, 3, 2)
            g_a = self.lin_nodes_veh(torch.cat((mm, pwm), dim=3))
            g_a = g_a + self.precomputed[:, None, :, :] + self.vehicle_proj[:, :, None, :]
            self.g_a = g_a
            K = self.lin_k(g_a)
            V = self.lin_v(g_a)
            K_lg = self.lin_lg(g_a)
            self.K, self.V, self.K_lg = K, V, K_lg
            self.mm, self.pwm = mm, pwm
            del g_a, mm, pwm
        else:
            vehicle_emb = vehicle_features[r, k, :].clone()
            del vehicle_features
            vehicle_emb = self.veh_emb(vehicle_emb)
            vehicle_k = self.veh_3(self.relu(self.veh_2(self.relu(self.veh_1(vehicle_emb)))))
            del vehicle_emb
            plan_ind = tour_plan[r, k, :]
            plan_ind_mask = (plan_ind != 0)
            plan_ind_mask = plan_ind_mask[:, :, None]
            nodes_ind = plan_ind[:, :, None].expand(bsz, 20, self.en_dm)
            nodes = self.h.gather(index=nodes_ind, dim=1)
            nodes = self.tour_2(self.relu(self.tour_1(nodes)))
            vehicle_k = torch.cat((vehicle_k, (nodes * plan_ind_mask.type(dtype=torch.float)).sum(dim=1) /
                                    L), dim=1)
            del plan_ind_mask, plan_ind
            v = self.vehicle.clone()
            del self.vehicle
            v[r, k, :] = vehicle_k
            self.vehicle = v
            vehicle_proj_k = self.lin_veh(vehicle_k)
            vehicle_k = vehicle_k.reshape(bsz, 1, -1)
            mm_k = torch.matmul(vehicle_k, self.h.permute(0, 2, 1)).unsqueeze(-1)
            pwm_k = (vehicle_k[:, :, :, None] * (self.h.permute(0, 2, 1).unsqueeze(1))).permute(0, 1, 3, 2)
            g_a_k = self.lin_nodes_veh(torch.cat((mm_k, pwm_k), dim=3))
            g_a_k = g_a_k + self.precomputed[:, None, :, :] + vehicle_proj_k[:, None, None, :]
            g_a_k = g_a_k.reshape(bsz, n, -1)
            g_a = self.g_a.clone()
            del self.g_a
            g_a[r, k, :, :] = g_a_k
            self.g_a = g_a
            K_k = self.lin_k(g_a_k)
            V_k = self.lin_v(g_a_k)
            K_lg_k = self.lin_lg(g_a_k)
            K = self.K.clone()
            del self.K
            V = self.V.clone()
            del self.V
            K_lg = self.K_lg.clone()
            del self.K_lg
            K[r, k, :, :], V[r, k, :, :], K_lg[r, k, :, :] = K_k, V_k, K_lg_k
            self.K, self.V, self.K_lg = K, V, K_lg
        fleet = self.vehicle.mean(dim=1)
        act_v = act[:, :, None].expand(bsz, self.act_num, self.en_dm)
        vehicle_act = self.vehicle.gather(index=act_v, dim=1)
        act = act[:, :, None, None].expand(bsz, self.act_num, n, self.dec_dim)
        K = self.K.gather(index=act, dim=1)
        V = self.V.gather(index=act, dim=1)
        K_lg = self.K_lg.gather(index=act, dim=1)
        act = vehicle_act.mean(dim=1)
        last = self.h[r, last, :].view(bsz, -1)
        tup = (self.h, self.h_g, vehicle_act, fleet, act, last, self.act_num)
        prec = (K, V, K_lg)
        v, p = self.decoder(tup, prec, mask, t, sample, self.precomputed)
        return v, p
