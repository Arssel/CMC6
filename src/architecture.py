import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        V = self.lin_v(x) # V.shape == (batch_size, n, dv)
        K = self.lin_k(x) # K.shape == (batch_size, n, dk)
        Q = self.lin_q(x) # Q.shape == (batch_size, n, dq)
        
        K = K.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        Q = Q.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=float)).to(x.device)
        QK_norm = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        QK_sm = torch.softmax(QK_norm, dim = -1)
        
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
        self.projection = nn.Linear(7, dm)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dm=dm, num_heads=num_heads, ff_dim=ff_dim) for _ in range(N)])
        
    def forward(self, x):
        x = self.projection(x)
        for module in self.encoder_layers:
            x = module(x)
        return x, x.mean(dim=1).squeeze(1)
    
class Decoder(nn.Module):
    def __init__(self, dm=128, num_heads=8):
        super().__init__()
        self.hid_dim = dm
        self.num_head = num_heads
        self.head_dim = int(dm//num_heads)
        #self.v = nn.Parameter(2*torch.rand(1, 2*dm)-1)
        self.lin_q = nn.Linear(3*dm + 2, dm)
        self.lin_o = nn.Linear(dm,dm)
        self.infty = torch.tensor(-1e15)
        
        
    def forward(self, x, h_g, mask, t, context, sample=True, precomputed=None, flags={}):
        bsz = x.shape[0]
        graph_size = x.shape[1]
        K, V, K_lg = precomputed
        if t == 0:
            if flags['demand'] and not flags['pd']:
                self.first = x[:, -1, :].squeeze(1)
                self.last = x[:, -1, :].squeeze(1)
                h_c = torch.cat((h_g, self.first, self.last, context),dim=1)
            else:
                zeros = torch.zeros((bsz, 2*self.hid_dim)).to(x.device)
                #h_c = torch.cat((h_g, self.v.expand((bsz, 2*self.hid_dim)), context), dim=1)
                h_c = torch.cat((h_g, zeros, context), dim=1)
        else:
            h_c = torch.cat((h_g, self.first, self.last, context),dim=1)
        h_c = h_c.unsqueeze(1)
        Q = self.lin_q(h_c)


        mask_for_transform = torch.Tensor(mask[:, None, None, :]).expand(bsz, self.num_head, 1, graph_size)
        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=float)).to(x.device)
        Q = Q.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)


        QK_norm = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        QK_norm[mask_for_transform == 0] = self.infty.to(x.device)
        QK_sm = torch.softmax(QK_norm, dim = -1)
        u = torch.matmul(QK_sm, V)
        u = u.permute(0, 2, 1, 3)
        u = u.view(bsz, 1, self.hid_dim)
        u = self.lin_o(u)
        u = torch.matmul(u, K_lg.permute(0, 2, 1)).view(bsz, graph_size)/torch.sqrt(torch.tensor(self.hid_dim, dtype=float)).to(x.device)
        u = torch.tensor(10).to(x.device)*torch.tanh(u)
        mask = torch.tensor(mask, dtype=int)
        w_t_m = (mask == 0)
        u[w_t_m] = self.infty.to(x.device)
        u = torch.softmax(u, axis=1)
        
        if sample:
            vertexes = u.multinomial(1)
        else:
            vertexes = u.argmax(dim=1).unsqueeze(1)
        probs = u.gather(1, vertexes).view(-1,1)
        
        v_ind = vertexes.view(-1, 1, 1)
        v_ind = v_ind.expand(bsz, 1, self.hid_dim)
        
        if t == 0 and not flags['demand']:
            self.first = x.gather(1, v_ind).squeeze(1)
        self.last = x.gather(1, v_ind).squeeze(1)
        
        return vertexes, probs
    
class AttentionModel(nn.Module):
    def __init__(self, dm=128, num_heads=8, ff_dim=512, N=3):
        super().__init__()
        self.hid_dim = dm
        self.head_dim = dm//num_heads
        self.num_head = num_heads
        self.encoder = Encoder(dm=dm, num_heads=num_heads, ff_dim=ff_dim, N=N)
        self.decoder = Decoder(dm=dm, num_heads=num_heads)
        self.lin_k = nn.Linear(dm, dm)
        self.lin_k_lg = nn.Linear(dm, dm)
        self.lin_v = nn.Linear(dm, dm)
        
    def embedding(self, x):
        h, h_g = self.encoder(x)
        return self.h, self.h_g
        
    def precompute(self, h):
        bsz = h.shape[0]
        graph_size = h.shape[1]
        K = self.lin_k(h)
        V = self.lin_v(h)
        K_lg = self.lin_k_lg(h)
        K = K.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        return (K, V, K_lg)
        
    def forward(self, x, mask, t, context, sample=True, flags={}):
        if t == 0:
            h, h_g = self.encoder(x)
            self.precomputed = self.precompute(h)
            self.h, self.h_g = h, h_g
        return self.decoder(self.h, self.h_g, mask, t, context, sample, self.precomputed, flags)