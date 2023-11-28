import torch
import torch.nn as nn

import numpy as np


class Conv1Layer(nn.Module):
    def __init__(self, in_feats, ws_feats, out_feats, im_size):
        super().__init__()

        # Modulation layer
        self.conv_mod    = nn.Linear(ws_feats, in_feats, bias=True)
        self.conv_weight = nn.Parameter(torch.randn(out_feats, in_feats, 3, 3))

        # Activation layer
        self.act_bias = nn.Parameter(torch.randn(1, out_feats, 1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.act_scale = np.sqrt(2).astype(np.float32)

        self.padding = 1

        self.noise = nn.Parameter(torch.randn(1, 1, im_size, im_size))


    def forward(self, x, ws):
        # Calculate filter from ws
        filt = self.conv_mod(ws)
        filt = torch.unsqueeze(torch.unsqueeze(filt, dim=2), dim=3)
        filt = self.conv_weight * filt

        # Normalize filter
        scale = 1.0 / torch.sqrt((torch.sum(filt * filt, dim=[1, 2, 3]) + 1e-8))
        scale = torch.unsqueeze(scale, dim=1)
        scale = torch.unsqueeze(scale, dim=2)
        scale = torch.unsqueeze(scale, dim=3)
        filt = filt * scale

        # Filter x
        x = torch.nn.functional.conv2d(x, filt, padding=self.padding)

        # Add noise
        x = x + self.noise

        # Activation
        x = x + self.act_bias
        x = self.lrelu(x)
        x = x * self.act_scale
        return x
