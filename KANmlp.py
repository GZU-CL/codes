from torch import nn
import sys
import torch.nn.functional as F

from kan_convolutional.KANLinear import KANLinear


class KAN1(nn.Module):
    def __init__(self, label_num, device: str = 'cpu'):
        super().__init__()
        self.flat = nn.Flatten()
        self.kan1 = KANLinear(
            28 * 28,
            label_num,
            grid_size=10,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0, 1],
        )

    def forward(self, x):
        x = self.flat(x)
        x = self.kan1(x)
        x = F.log_softmax(x, dim=1)
        return x