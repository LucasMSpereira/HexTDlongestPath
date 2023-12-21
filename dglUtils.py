# definitions for implementing the graph transformer

import numpy as np
from copy import deepcopy
import dgl
from dgl.nn import GraphConv
from dgl import function as fn
from torch import nn
import torch.nn.functional as F

class gcn(nn.Module):
    
  """Graph convolutional network"""

  def __init__(self, params: dict) -> None:
    super(gcn, self).__init__()
    
    self.conv1 = GraphConv(params["inDim"], params["hDim"], activation = params["activFunction"])
    self.conv2 = GraphConv(params["hDim"], 1)

  def forward(self, g, inFeature):
    h = self.conv1(g, inFeature)
    return dgl.mean_nodes(g, "h")