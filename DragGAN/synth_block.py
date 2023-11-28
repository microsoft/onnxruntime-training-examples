import torch.nn as nn

from conv0 import Conv0Layer
from conv1 import Conv1Layer
from rgb import RGBLayer


class SynthBlock(nn.Module):
    def __init__(self, in_feats, ws_feats, out_feats, im_size):
        super().__init__()
        self.conv0 = Conv0Layer(in_feats=in_feats, ws_feats=ws_feats, out_feats=out_feats, im_size=im_size)
        self.conv1 = Conv1Layer(in_feats=out_feats, ws_feats=ws_feats, out_feats=out_feats, im_size=im_size)
        self.rgb  = RGBLayer(in_feats=out_feats, ws_feats=ws_feats)


    def forward(self, x, ws, img):
        x = self.conv0(x, ws[:,0,:])
        x = self.conv1(x, ws[:,1,:])
        img = self.rgb(x, ws[:,2,:], img)
        return x, img
