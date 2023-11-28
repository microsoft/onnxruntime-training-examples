import torch
import torch.nn as nn

import numpy as np


class Conv0Layer(nn.Module):
    def __init__(self, in_feats, ws_feats, out_feats, im_size):
        super().__init__()

        # Modulation layer
        self.conv_mod    = nn.Linear(ws_feats, in_feats, bias=True)
        self.conv_weight = nn.Parameter(torch.randn(out_feats, in_feats, 3, 3))

        # Blur layer
        self.blur = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, bias=False)

        # Activation layer
        self.act_bias = nn.Parameter(torch.randn(1, out_feats, 1, 1))
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.act_scale = np.sqrt(2).astype(np.float32)

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

        # Flip filter
        filt = torch.flip(filt, dims=[2, 3])

        # Blur x
        x = x.permute([1, 0, 2, 3])
        x = self.blur(x)
        x = x.permute([1, 0, 2, 3])

        # Filter x
        x = torch.nn.functional.conv2d(x, filt)

        # Add noise
        x = x + self.noise

        # Activation
        x = x + self.act_bias
        x = self.lrelu(x)
        x = x * self.act_scale
        return x
