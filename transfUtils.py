# definitions for implementing the graph transformer

import numpy as np
from copy import deepcopy
import tensorflow as tf
# import tensorflow.experimental.numpy as tnp
# tnp.experimental_enable_numpy_behavior()
from tensorflow import keras
from keras import layers
import dgl
from dgl import function as fn
from torch import nn

class graphTransformer(nn.Module):
    
    """Graph transformer network"""

    def __init__(self, params: dict) -> None:
      super().__init__()
      # l graph transformer layers
        # input state, shape = 1
        # (1) embedded state, shape = embedDim
        # multi-head attention
          # h_i -> Q, K, V. shape = (nNodes, nHeads, embedDim)
          # (2) (Q, K) -> dot, scale, softmax
          # (2) + V
        # (3) DNN(out_dim -> out_dim)
        # (4) norm(1 + 3)
        # (5) FNN
        # norm(5 + 4)
      # task MLP for binary classification

      # states of node and neighborhood as inputs
      nodeState = keras.Input(shape = (1,), name = "nodeState")
      # embedding layer
      embedState = layers.Embedding(params["flagAmount"] + 4, params["embedDim"])
      # embed both inputs
      nodeEmbed = embedState(nodeState)
      # initial residual connection for node state in first transformer layer
      h = deepcopy(nodeEmbed)
      # sequence of transformer layers
      for headID in range(params["numberLayers"]):
        # last layer output with 'outDim' dimension.
        # outputs of other layers with embedding dimension
        if headID != params["numberLayers"] - 1:
          layerOutDim = params["embedDim"]
        else:
          layerOutDim = params["outDim"]
        # attention. input dimension is always 'embedDim'.
        # output dimension changes in last transformer layer
        head = transfHead(
          layerOutDim // params["numberHeads"],
          params["numberHeads"],
        )(h, params["mapGraph"])
        # 'O' operation
        head = layers.Dense(layerOutDim)(head)
        # add first residual connection in current layer, then normalize
        h = layers.BatchNormalization()(h + head)
        # transformer layer DNN
        hDNN = layers.Dense(layerOutDim * 2, activation = params["transfDNNactiv"])(h)
        hDNN = layers.Dense(layerOutDim)(hDNN)
        # add second residual connection in current layer, then normalize
        h = layers.BatchNormalization()(h + hDNN)
      transfOut = h
      classifyMLP = layers.Dense(1, activation = tf.nn.sigmoid)(transfOut)
      model = keras.Model(inputs = [nodeState], outputs = classifyMLP)
      model.summary()
      return model
      
# attention utilities
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
      return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim = True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
      # clamp for softmax numerical stability
      return {field: tf.math.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
    return func

class transfHead(layers.Layer):
    
  """Head definition for graph transformer layer"""

  def __init__(self, outDim, nHeads) -> None:
    super().__init__()
    self.Q = layers.Dense(outDim * nHeads)
    self.K = layers.Dense(outDim * nHeads)
    self.V = layers.Dense(outDim * nHeads)
    self.nHeads = nHeads
    self.outDim = outDim

  def propagate_attention(self, g):
    # Compute attention score
    g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
    g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

    # Send weighted values to target nodes
    g.send_and_recv(g.edges(), fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
    g.send_and_recv(g.edges(), fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

  def call(self, state, graph):
    # graphEmbed = list(map(self.embedLayer, g.nodes()))
    # build Q, K, V from node state.
    # reshape these matrices into
    # (number of nodes, number of heads, feature dimension).
    # store in respective node features in fields of DGL graph object
    # @tf.autograph.experimental.do_not_convert
    graph.ndata['Q_h'] = tf.reshape(
      self.Q(state), (graph.num_nodes(), self.nHeads, self.outDim)
    ).numpy()
    graph.ndata['K_h'] = tf.reshape(self.K(state), (graph.num_nodes(), self.nHeads, self.outDim))
    graph.ndata['V_h'] = tf.reshape(self.V(state), (graph.num_nodes(), self.nHeads, self.outDim))

    self.propagate_attention(graph)

    return graph.ndata['wV']/graph.ndata['z']