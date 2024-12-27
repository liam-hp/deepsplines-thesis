import torch
import torch.nn as nn
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath('../DeepSplines'))
from deepsplines.ds_modules import dsnn

class LinearReLU(nn.Module):
  def __init__(self, layers, input_size=8):
    super(LinearReLU, self).__init__()

    self.layers = nn.Sequential()  
    prev=input_size
    for l in layers:
      self.layers.append(nn.Linear(prev, l))
      self.layers.append(nn.ReLU())
      prev=l
    self.layers.append(nn.Linear(prev, 1))

    self.num_params = sum(p.numel() for p in self.parameters())

  def forward(self, x):
    x = self.layers(x)
    return x

class LinearBSpline(dsnn.DSModule):
  def __init__(self, layers, s=5, r=4, init="relu", input_size=8):
    super(LinearBSpline, self).__init__()

    self.fc_ds = nn.ModuleList()
    opt_params = {
            'size': s,
            'range_': r,
            'init': init,
            'save_memory': False
    }
    self.layers = nn.Sequential()

    prev=input_size
    for l in layers:
      self.layers.append(nn.Linear(prev, l))
      self.layers.append(dsnn.DeepBSpline('fc', l, **opt_params))
      prev=l
    self.layers.append(nn.Linear(prev, 1))

    self.num_params = self.get_num_params()
  
  def get_layers(self):
    return self.layers
  
  def forward(self, x):
    x = self.layers(x)
    return x