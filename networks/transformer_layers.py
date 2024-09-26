import torch.nn as nn
import copy
from networks.layers import clones, SublayerConnection
import torch


class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = copy.deepcopy(self_attn)
        self.feed_forward = copy.deepcopy(feed_forward)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):    # B x L x D
        """Separately calculate each channel."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x    # B x L x D
