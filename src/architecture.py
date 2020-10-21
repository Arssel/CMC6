import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class LinearEmbedding(nn.Module):
    def __init__(self, dm=128):
        super().__init__()
        self.dm  = dm
        self.projection = nn.Linear(7, dm)
        
        
    def forward(self, x):
        return self.projection(x)

    
class EncoderLayer(nn.Module):
    def __init__(self, dm=128, num_heads=8, ff_dim=512):
        super().__init__()
        self.s_att = nn.MultiheadAttention(dm=dm, num_heads=num_heads)
        self.batch_norm_1 = nn.BatchNorm1d(dm)
        self.batch_norm_2 = nn.BatchNorm1d(dm)
        self.ff = nn.Sequential(nn.Linear(dm, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dm))
        
    def forward(self, x):
        x = self.s_att(x) + x
        x = self.batch_norm_1(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.ff(x) + x
        x = self.batch_norm_2(x.transpose(-1, -2)).transpose(-1, -2)
        return x

    
class Encoder(nn.Module):
    def __init__(self, dm=128, num_heads=8, ff_dim=512, N=3):
        super().__init__()
        self.lin_proj() = LinearEmbedding(dm=dm)
        self.encoder_layers = nn.ModuleList([EncoderLayer(dm=dm, num_heads=num_heads, ff_dim=ff_dim) for _ in range(N)])
        
    def forward(self, x):
        x = self.lin_proj(x)
        for module in self.encoder_layers:
            x = module(x)
        return x, x.mean(dim=1).squeeze()
    
class Decoder(nn.Module):
    def __init__(self, dm=128, num_heads=8):
        super().__init__()
        self.hid_dim = dm
        self.num_head = num_heads
        self.head_dim = dm/num_heads
        self.v1 = nn.parameter.Parameter(torch.rand(1, dm))
        self.v2 = nn.parameter.Parameter(torch.rand(1, dm))
        self.lin_k = nn.Linear(dm, dm)
        self.lin_v = nn.Linear(dm, dm)
        self.lin_q = nn.Linear(3*dm, dm)
        self.lin_h = nn.Linear(dm,dm)
        self.infty = torch.tensor(-1e10)
        
        
    def forward(self, x, h_g, mask, t, sample=True):
        bsz = x.shape[0]
        if t == 0:
            v1_bsz = torch.zeros((bsz, self.hid_dim))
            v2_bsz = torch.zeros((bsz, self.hid_dim))
            v1_bsz[:,] = self.v1
            v2_bsz[:,] = self.v2
            h_c = torch.cat((h_g, v1_bsz, v2_bsz), dim=1)
        else:
            h_c = torch.cat((h_g, self.first, self.last),dim=1)
        h_c = h_c.unsqueeze(1)
        K = self.lin_k(x)
        Q = self.lin_q(h_c)
        V = self.lin_v(x)
        
        mask_for_transform = mask.unsqueeze(1).unsqueeze(1)
        scale = torch.sqrt(torch.tensor(self.head_dim)).to(x.device)
        
        K = K.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        Q = Q.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_head, self.head_dim).permute(0, 2, 1, 3)
        
        QK_norm = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        QK_norm = QK_norm.masked_fill(mask == 0, self.infty.to(x.device))
        QK_sm = torch.softmax(QK_norm, dim = -1)
        u = torch.matmul(QK_sm, V)
        
        u = u.permute(0, 2, 1, 3)
        u = u.view(batch_size, self.hid_dim)
        u = self.lin_h(u)
        
        if sample:
            vertexes = u.multinomial(1)
        else:
            vertexes = u.argmax(dim=1)
        probs = u[np.arange(bsz), vertexes]
        
        if t == 1:
            self.first = x[np.arange(bsz), vertexes]
        self.last = x[np.arange(bsz), vertexes]
        
        return vertexes, probs