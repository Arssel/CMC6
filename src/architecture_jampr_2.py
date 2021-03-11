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
        self.lin_q = nn.Linear(4 * en_dm, dec_dm)
        self.lin_o = nn.Linear(dec_dm, dec_dm)
        self.infty = torch.tensor(-1e15)

    def forward(self, tup, prec, mask, t, sample=True):
        h, h_g, vehicle, act, last, act_num = tup
        bsz = h.shape[0]
        graph_size = h.shape[1]
        k = vehicle.shape[1]
        h_c = torch.cat((h_g, act, h[:, 0, :].squeeze(1), last), dim=1)
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
        self.veh_3 = nn.Linear(veh_dim, en_dm)
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

    def forward(self, features, mask, t, precomputed=None, sample=True):
        vehicle_features = features[1]
        bsz = vehicle_features.shape[0]
        last = features[2]
        r = torch.arange(bsz, dtype=torch.int64)
        if precomputed is None:
            node = features[0]
            self.n = node.shape[1]
            h, h_g = self.encoder(node)
            h_pr = self.precompute(h)
            precomputed = (h, h_g, h_pr)
        else:
            h, h_g, h_pr = precomputed
        vehicle_emb = self.veh_emb(vehicle_features)
        vehicle = self.veh_3(self.relu(self.veh_2(self.relu(self.veh_1(vehicle_emb)))))
        mm = torch.matmul(vehicle, h.permute(0, 2, 1)).unsqueeze(-1)
        pwm = (vehicle[:, :, :, None] * (h.permute(0, 2, 1).unsqueeze(1))).permute(0, 1, 3, 2)
        g_a = self.lin_nodes_veh(torch.cat((mm, pwm), dim=3))
        g_a = g_a + h_pr[:, None, :, :] + vehicle[:, :, None, :]
        K = self.lin_k(g_a)
        V = self.lin_v(g_a)
        K_lg = self.lin_lg(g_a)

        act = vehicle.mean(dim=1)
        last = h[r, last.squeeze(), :].view(bsz, -1)
        tup = (h, h_g, vehicle, act, last, self.act_num)
        prec = (K, V, K_lg)
        v, p = self.decoder(tup, prec, mask, t, sample)
        return v, p, precomputed