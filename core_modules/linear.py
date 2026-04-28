import math
import torch
import torch.nn as nn


# Lessons:
# 1. torch.matmul's broadcast rule requires row vector multiplication for batched input vectors.
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


# Lessons:
# 1. in-place op on leaf variable that requires grad (Parameter) cause runtime error generally for two reasons:
#   1) doing in-place op confuse torch computation graph as such node is not a leaf if assigned but parameter has
#   to be a leaf so optimizer knows what to update; 2)leaf variables are meant to be updated by optimizer after
#   .backward(). torch optimizer use no_grad for such update exactly.
# 2. requires_grad_(False) and fill_(0) are only allowed under torch.no_grad() as above, but their behavior differ
#   requires_grad_ only modify the slice view, the fill_ modifies the underlying data, so
#   self.W[padding_idx].requires_grad_(False) doesn't work, it is a no op silently, which can be test using assert
#   self.W[padding_idx].requires_grad is False, which will fail, because requires_grad is parameter level data, it
#   only gets update if we call self.W.requires_grad_(False)
# 3. use register_buffer whenever you have a tensor that is not a parameter (doesn't need gradients) but is still
#   a part of the model's state that needs to stay on the same device as your weights. Common examples include
#   attention masks in Transformers or the running mean and variance in BatchNorm layers.
# 4. scale_grad_by_freq in PyTorch's context they use the last batch's stat instead of running stat.
class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(W, x):
        return W[x]

    @staticmethod
    def setup_context(ctx, inputs, output):
        W, x = inputs
        ctx.save_for_backward(W.shape[0], x, output)

    @staticmethod
    def backward(ctx, grad_output):
        num_embeddngs, x, output = ctx.saved_tensors
        return torch.sparse_coo_tensor(
            indices=x * num_embeddngs,
            values=grad_output,
            size=(num_embeddngs, output.shape[-1]),
        )


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
        if _weight is not None:
            self.W = nn.Parameter(_weight, requires_grad=not _freeze)
        else:
            self.W = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                requires_grad=not _freeze,
            )
            nn.init.normal_(self.W)

        if padding_idx is not None:
            with torch.no_grad():
                self.W[padding_idx].fill_(0)
            self.register_buffer(
                "_padding_idx_tensor",
                torch.tensor([padding_idx], device=self.W.device, dtype=torch.long),
            )
            if self.W.requires_grad:
                self.W.register_hook(
                    lambda grad: grad.index_fill_(0, self._padding_idx_tensor, 0)
                )

        self.scale_grad_by_freq = scale_grad_by_freq
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.sparse = sparse

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                unique_indices = torch.unique(x)
                norms = torch.linalg.vector_norm(
                    self.W[unique_indices], ord=self.norm_type, dim=-1
                )
                mask = norms > self.max_norm
                self.W[unique_indices[mask]] *= (self.max_norm / norms[mask]).unsqueeze(
                    -1
                )
        if self.sparse:
            out = EmbeddingFunction.apply(self.W, x)
        else:
            out = self.W[x]

        if self.W.requires_grad and self.scale_grad_by_freq:
            freq = torch.bincount(x.flatten(), minlength=self.W.shape[0]).to(
                self.W.dtype
            )
            inverse_freq = 1 / freq.clamp(min=1)
            out.register_hook(lambda grad: inverse_freq[x].unsqueeze(-1) * grad)
        return out
