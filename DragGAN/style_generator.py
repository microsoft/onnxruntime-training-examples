import torch.nn as nn
import torch.nn.functional as FF

from first_block import FirstBlock
from synth_block import SynthBlock


class StyleGenerator(nn.Module):
    def __init__(self, feat_list, ws_feats):
        super().__init__()

        self.block_list = nn.ModuleList()
        for idx in range(len(feat_list)-1):
            im_size = 2**(idx+2)

            if idx == 0:
                block = FirstBlock(
                    feat_list[idx], ws_feats, feat_list[idx+1], im_size)
            else:
                block = SynthBlock(
                    feat_list[idx], ws_feats, feat_list[idx+1], im_size)

            self.block_list.append(block)

    def forward(self, ws):
        block = self.block_list[0]
        x, img = block(ws[:, 0:2, :])

        for (index, block) in enumerate(self.block_list[1::]):
            i = 1 + index * 2
            x, img = block(x, ws[:, i:i+3, :], img)

            if index == 5:
                F = x

        F = FF.interpolate(F, img.shape[-2:], mode='bilinear')

        return img, F
