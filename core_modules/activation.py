import torch
import torch.nn as nn


# Lessons:
# 1. In-place op on a leaf tensor with requires_grad=True (e.g. nn.Parameter, or any user-created tensor with
#    requires_grad=True) raises: "a leaf Variable that requires grad is being used in an in-place operation."
#    Two reasons:
#      1) Graph correctness: backward needs the original input values (ReLU's backward needs to know which entries
#         were <= 0). In-place overwrites them, so the saved tensor on the autograd tape would be corrupted.
#      2) Leaf identity: optimizer.step() updates params via no_grad in-place writes and assumes params stay leaves.
#         Mutating a leaf inside forward bumps its _version mid-graph and breaks that assumption.
# 2. In-place is safe on non-leaf tensors (outputs of previous ops) when no downstream backward needs the
#    pre-mutation values. ReLU qualifies: its backward only needs output > 0, which the in-place result still gives.
#    Counter-example: log_'s backward needs the input value, so log_ on a tensor reused later is unsafe.
# 3. PyTorch tracks each tensor's _version; if a saved tensor's version changes before .backward(), it raises then.
def relu(input, inplace=False):
    return input.clamp_(min=0) if inplace else input.clamp(min=0)


def relu_(input):
    return input.clamp_(min=0)


class ReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return relu(x, self.inplace)


# Customized backward is used to avoid numerical overflow. When x is very small x (negative), regular autograd will
# compute exp(-x) / (1 + exp(-x))^2, leading to overflow. using out * (1 - out) with out in [0, 1] avoids this.
class SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return 1 / (1 + torch.exp(-x))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out)


def sigmoid(input):
    return SigmoidFunction.apply(input)


class Sigmoid(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return sigmoid(input)
