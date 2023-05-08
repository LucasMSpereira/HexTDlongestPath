#%% Imports
import os
os.environ['DGLBACKEND'] = 'tensorflow'
import transfUtils
import keras
from graph_manager import graphManager
import dgl
import random
import numpy as np
import utilities as utils
import networkx as nx
import tensorflow as tf
import data_utils
#%% Definitions
random.seed(100)
rowAmount = colAmount = 10 # map dimensions
ds = data_utils.dataManager(0, rowAmount, colAmount)
goal = "optimalPath" # 'both', 'optimalPath', 'OSPlenth'
dataTrain = ds.TFdata(goal, trainSplit = 1)[0] # dataset
dataSize = len(dataTrain.batch(1))
# for (sampleID, (initialMap, OSPlength, optimalMap)) in enumerate(dataTrain.batch(1)):
for (sampleID, (initialMap, optimalMap)) in enumerate(dataTrain.batch(1)):
  if sampleID % 1000 == 0:
    print(f"""Sample {sampleID + 1}/{dataSize}""")
  # useful format for map definitions
  initialMap = list(np.asarray(initialMap).reshape(-1, 1).transpose()[0])
  optimalMap = list(np.asarray(optimalMap).reshape(-1, 1).transpose()[0])
  # decode to graphManager format
  ds.decodeMapString(initialMap)
  spotDict = ds.decodeMapString(optimalMap)
  # instantiate graphManager object for graph utilities
  graphM = graphManager(utils.listToStr(optimalMap),
    rowAmount, colAmount, *utils.flagsFromDict(spotDict, colAmount)
  )
  # graph representation from string definition
  nxGraph = graphM.binaryToGraph(graphM.mapDefinition[0])
  # change element type of map representations, and combine them
  nodeFeature = []
  for (initialHexType, optimalHexPresence) in zip(initialMap, graphM.mapDefinition[0]):
    nodeFeature.append((int(initialHexType), int(optimalHexPresence)))
  # annotate nodes of networkx graph.
  # initial map is used as input to graph transformer. this map's nodes
  # are annotated with type code: 0=void, 1=full, 2=spawn, 3=flag(s), 4=base
  # optimal map serves as node labels. node features
  # are binary representation: 0=void, 1=full
  nx.set_node_attributes(
    nxGraph,
    {hexID: {
        "initialType": initialHexType, "optimalType": optimalHexPresence
      } for (hexID, (initialHexType, optimalHexPresence)) in enumerate(nodeFeature)
    }
  )
  # dgl graph from networkx. indicate attributes to copy over.
  # makes the graph directed with networkx.Graph.to_directed()
  # (all edges are bidirectional)
  DGLgraph = dgl.from_networkx(nxGraph, ["initialType", "optimalType"])
#%%
model = transfUtils.graphTransformer(
  { # parameters
    "map": 1,
    "flagAmount": 2, # number of flags in maps used in training
    "embedDim": 5, # dimension of state embedding
    "numberLayers": 5, # number of graph transformer layers
    "numberHeads": 5, # number of heads in each graph transformer layer
    "outDim": 15, # output size of DNN in graph transformer layers
    # activation function in first FNN layer of transformer layer
    "transfDNNactiv": "relu"
  }
)
#%% Data
dataTrain, dataVal = ds.TFdata(goal) # data splits
batchSize = 100
# organize both splits in batches
batchesTrain = dataTrain.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
batchesVal = dataVal.batch(batch_size = batchSize,
  num_parallel_calls = tf.data.AUTOTUNE, deterministic = False
)
#%% Instantiate graph transformer model
model.compile(
  optimizer = keras.optimizers.RMSprop(
  learning_rate = hp.Float("lr", min_value = 1e-4, max_value = 1, sampling = "log")),
  loss = keras.losses.MeanSquaredError(),
  metrics = [keras.metrics.MeanAbsoluteError()]
)