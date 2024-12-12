import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """docstring for Encoder"""

    def __init__(self, config, N, K, use_cuda):
        super(Model, self).__init__()
        self.embed_size = config['embedding_size']
        self.hidden_size = config['hidden_size']

        self.N = N
        self.K = K
        self._cuda = use_cuda

        self.encoder = nn.Linear(self.embed_size, self.hidden_size, bias=False)
        self.relation_net = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=False),
            nn.ReLU()
        )

        # encoder_layer = nn.TransformerEncoderLayer(d_model=2*self.hidden_size, nhead=8,
        #                                            dim_feedforward=4*self.hidden_size)