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


def sigmoid(input):
    return 1 / (1 + torch.exp(-input))


class Sigmoid(nn.Module):
    def forward(self, input):
        return sigmoid(input)
