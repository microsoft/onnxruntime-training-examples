import torch
import torch.nn as nn

from conv1 import Conv1Layer


class FirstBlock(nn.Module):
    def __init__(self, in_feats, ws_feats, out_feats, im_size):
        super().__init__()

        self.x = nn.Parameter(torch.randn(1, in_feats, 4, 4))
        self.conv1 = Conv1Layer(in_feats=in_feats, ws_feats=ws_feats, out_feats=out_feats, im_size=im_size)

        # Modulation layer
        self.conv_mod    = nn.Linear(ws_feats, in_feats, bias=True)
        self.conv_weight = nn.Parameter(torch.randn(3, in_feats, 1, 1))

        # Activation layer
        self.bias = nn.Parameter(torch.randn(3))
        self.padding = 0


    def forward(self, ws):
        x = self.conv1(self.x, ws[:,0,:])

        filt = self.conv_mod(ws[:,1,:])
        filt = torch.unsqueeze(torch.unsqueeze(filt, dim=2), dim=3)
        filt = self.conv_weight * filt

        img = torch.nn.functional.conv2d(x, filt, padding=self.padding, bias=self.bias)

        return x, img
