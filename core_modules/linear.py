import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            if bias
            else None
        )
        self._init_weight()

    def _init_weight(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = x @ self.W.T
        # out = torch.matmul(x, self.W.T)
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.W = nn.Parameter(
            _weight
            if _weight is not None
            else torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            requires_grad=not _freeze,
        )
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        nn.init.uniform_(self.W)
        if padding_idx is not None:
            self.W[padding_idx] = 0

    def forward(self, x):
        if self.scale_grad_by_freq:
            freq = torch.zeros(self.num_embeddings)
            for val in x.flatten():
                freq[val] += 1
            self.W.register_hook(lambda grad: (1 / freq) @ grad)
        return self.W[x]
