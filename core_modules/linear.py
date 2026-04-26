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
