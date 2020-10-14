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
        
        # x.size == (batch_size, n, 2)
        # s.size == (batch_size, n)
        # self.projection(x).size == (batch_size, n, dm)
        # positional_encoding(s, self.dm, self.n) == (batch_size, n, dm)
        batch_size, n, _ = x.shape
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
        self.encoder_layers = nn.ModuleList([EncoderLayer(dm=dm, num_heads=num_heads, ff_dim=ff_dim) for _ in range(N)])
        
    def forward(self, x):
        for module in self.encoder_layers:
            x = module(x)
        return x


