import os
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import copy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import torch.nn.functional as F
import warnings
from video_dataset import VideoFrameDataset, ImglistToTensor

# Totally naive decoder implementation for testing, takes in frames x embed_dim and outputs frames x h x w x num_classes
# When running a batch, we should rearrange to (b f) x embed_dim before passing in
# Input is of shape b t (h w) m, 
class BaselineFCDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, H, W, num_classes, num_hidden_layers=2,dropout=0.1):
        super(BaselineFCDecoder, self).__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("fc_in", nn.Linear(embed_dim, hidden_dim))

        for i in range(num_hidden_layers):
            self.layers.add_module(f"hidden_{i+1}", nn.Linear(hidden_dim, hidden_dim))
            self.layers.add_module(f"relu_{i+1}", nn.ReLU())
            self.layers.add_module(f"dropout_{i+1}", nn.Dropout(p=dropout))

        self.layers.add_module("fc_out", nn.Linear(hidden_dim,  H * W * num_classes))

        self.reshape_output = nn.Sequential(
            nn.Unflatten(2, (num_classes, H, W)) # Wrong? the tensor is (b n H * W * num_classes) where n is the number of patches
        )

    def forward(self, x):
        self.layers(x)
        return self.reshape_output(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_hidden_layers=2):
        super(Decoder, self).__init__()

        self.decoder = BaselineFCDecoder(input_dim, hidden_dim, num_classes, num_hidden_layers) # What are H and W?

    def forward(self, x):
        x = rearrange(x, 'b f e -> (b t) c h w') # the input is of shape (b t n m), also what is c? in order for the function to work some values need to be specified
        return self.decoder(x)