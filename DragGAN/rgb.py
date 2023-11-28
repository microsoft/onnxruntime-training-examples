import torch
import torch.nn as nn


class RGBLayer(nn.Module):
    def __init__(self, in_feats, ws_feats):
        super().__init__()

        # Modulation layer
        self.conv_mod    = nn.Linear(ws_feats, in_feats, bias=True)
        self.conv_weight = nn.Parameter(torch.randn(3, in_feats, 1, 1))

        # Upsample layer
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)

        # Activation layer
        self.bias = nn.Parameter(torch.randn(3))
        self.padding = 0


    def forward(self, x, ws, img):
        # Calculate filter from ws
        filt = self.conv_mod(ws)
        filt = torch.unsqueeze(torch.unsqueeze(filt, dim=2), dim=3)
        filt = self.conv_weight * filt

        # Filter x
        x = torch.nn.functional.conv2d(x, filt, padding=self.padding, bias=self.bias)

        # Upsample output image from previous block
        img = img.permute([1, 0, 2, 3])
        img = self.upsample(img)
        img = img.permute([1, 0, 2, 3])

        img = img + x
        return img
