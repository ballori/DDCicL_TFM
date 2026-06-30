import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableSoftThreshold1D(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        # We start with a small value to encourage sparsity, but it can be learned during training.
        # We use shape [1, num_filters, 1] to ensure it works well with 1D tensors
        self.threshold = nn.Parameter(torch.full((1, num_filters, 1), 0.0001))

    def forward(self, x):
        abs_x = torch.abs(x)
        thresh = F.relu(self.threshold) # The threshold must be non-negative
        return torch.sign(x) * F.relu(abs_x - thresh)