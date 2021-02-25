import torch
import math
from torch import nn
from functools import partial


def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class StandardAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        scale = torch.sqrt(torch.tensor(Q.shape[-1], dtype=float)).to(Q.device)
        QK_norm = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        QK_sm = torch.softmax(QK_norm, dim=-1)

        QKV = torch.matmul(QK_sm, V)
        QKV = QKV.permute(0, 2, 1, 3).contiguous()
        QKV = QKV.view(QKV.shape[0], -1, QKV.shape[-1]*QKV.shape[-2])

        return QKV

class NystromAttention(nn.Module):
    def __init__(self, num_landmarks=10, head_dim=16, num_head=8):
        super().__init__()

        self.head_dim = head_dim
        self.num_head = num_head
        self.num_landmarks = num_landmarks

    def forward(self, Q, K, V):

        seq_len = Q.shape[[-2]]
        Q = Q / math.sqrt(math.sqrt(self.head_dim))
        K = K / math.sqrt(math.sqrt(self.head_dim))

        Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, seq_len // self.num_landmarks,
                                    self.head_dim).mean(dim=-2)
        K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, seq_len // self.num_landmarks,
                                    self.head_dim).mean(dim=-2)

        kernel_1 = torch.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel_2 = torch.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel_3 = torch.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)), dim=-1)
        X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        return X

    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat
        V = 1 / (torch.max(torch.sum(torch.abs(K), dim=-2)) * torch.max(torch.sum(torch.abs(K), dim=-1))) * K.transpose(
            -1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V


class PerformerAttention(nn.Module):
    def __init__(self, dim_heads):
        super().__init__()
        nb_features = int(dim_heads * math.log(dim_heads))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        projection_matrix = self.gaussian_orthogonal_random_matrix(nb_features, dim_heads)
        self.register_buffer('projection_matrix', projection_matrix)

    def linear_attention(self, q, k, v):
        k_cumsum = k.sum(dim=-2)
        D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
        context = k.permute(0, 2, 1) @ v
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out

    def softmax_kernel(self, data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4):
        b, h, *_ = data.shape
        data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
        ratio = (projection_matrix.shape[0] ** -0.5)

        projection = projection_matrix.expand(b, h, projection_matrix.shape[0], projection_matrix.shape[1])
        data_dash = (data_normalizer * data) @ projection.permute(0, 1, 3, 2)
        data_norm = ((data ** 2).sum(dim=-1)/ 2.0) * (data_normalizer ** 2)

        if is_query:
            data_dash = ratio * (
                    torch.exp(data_dash - data_norm -
                              torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
        else:
            data_dash = ratio * (
                    torch.exp(data_dash - data_norm - torch.max(data_dash)) + eps)

        return data_dash

    def orthogonal_matrix_chunk(self, cols, device='cuda'):
        unstructured_block = torch.randn((cols, cols), device=device)
        q, r = torch.qr(unstructured_block.cpu(), some=True)
        q, r = map(lambda t: t.to(device), (q, r))

        return q.t()

    def gaussian_orthogonal_random_matrix(self, nb_rows, nb_columns, device='cuda'):
        nb_full_blocks = int(nb_rows / nb_columns)

        block_list = []
        for _ in range(nb_full_blocks):
            q = self.orthogonal_matrix_chunk(nb_columns, device=device)
            block_list.append(q)

        remaining_rows = nb_rows - nb_full_blocks * nb_columns
        if remaining_rows > 0:
            q = self.orthogonal_matrix_chunk(nb_columns, device=device)
            block_list.append(q[:remaining_rows])

        final_matrix = torch.cat(block_list)
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)

        return torch.diag(multiplier) @ final_matrix

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        create_kernel = partial(self.softmax_kernel, projection_matrix=self.projection_matrix)
        q = create_kernel(q, is_query=True)
        k = create_kernel(k, is_query=False)
        out = self.linear_attention(q, k, v)
        return out


class LinformerAttention(nn.Module):
    def __init__(self, seq_len, k=20, dim=16, heads=8, dim_head=None):
        super().__init__()

        self.seq_len = seq_len
        self.k = k

        self.heads = heads
        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))
        self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

    def forward(self, queries, keys, values):
        k = self.k
        b, h, n, d_h = queries.shape

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)
        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        out = out.transpose(1, 2).reshape(b, n, -1)
        return out
