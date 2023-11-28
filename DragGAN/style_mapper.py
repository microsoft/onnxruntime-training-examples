import torch.nn as nn

class StyleMapper(nn.Module):
    def __init__(self, num_layers, ws_feats):
        super().__init__()

        self.layer_list = nn.ModuleList()
        self.act_list = nn.ModuleList()
        self.scale = 2 ** 0.5

        for _ in range(num_layers):
            layer = nn.Linear(ws_feats, ws_feats)
            self.layer_list.append(layer)

            act = nn.LeakyReLU(negative_slope=0.2)
            self.act_list.append(act)

    def forward(self, x):
        for layer, act in zip(self.layer_list, self.act_list):
            x = layer(x)
            x = act(x) * self.scale
        return x
