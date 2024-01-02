# Script used to generate annotated DGL graph dataset
#%% Imports
import os
from copy import deepcopy
os.environ['DGLBACKEND'] = 'pytorch'
from graph_manager import graphManager
from datetime import datetime
import dgl
import random
import torch
import numpy as np
import utilities as utils
import networkx as nx
import data_utils
random.seed(100)
#%% Setup variables
rowAmount = colAmount = 10 # map dimensions
ds = data_utils.dataManager(0, rowAmount, colAmount)
# Dataset (with possible goals 'both', 'optimalPath', 'OSPlenth')
data = ds.TFdata("both", trainSplit = 1)
ds.generateMap()
dataSize = len(data.batch(1))
#%% Generate dataset
DGLdataset = [] # list of samples
OSPlengthList = [] # list of OSP lengths of each sample
for (sampleID, (initialMap, OSPlength, optimalMap)) in enumerate(data.batch(1)):
# for (sampleID, (initialMap, optimalMap)) in enumerate(data.batch(1)):
  if sampleID % 4000 == 0:
    print(f"""Sample {sampleID}/{dataSize} Time: {datetime.now().strftime("%H:%M:%S")}""")
  # useful format for map definitions
  initialMap = list(np.asarray(initialMap).reshape(-1, 1).transpose()[0])
  optimalMap = list(np.asarray(optimalMap).reshape(-1, 1).transpose()[0])
  # change element type of map representations, and combine them
  nodeFeature = []
  for (initialHexType, optimalHex) in zip(initialMap, optimalMap):
    nodeFeature.append((int(initialHexType), int(optimalHex)))
  # annotate nodes of networkx graph.
  # initial map is used as input. this map's nodes
  # are annotated with type code: 0 = void, 1 = full,
  # 2 = spawn, 3 = flag(s), 4 = base.
  # optimal map serves as node labels. node features
  # have binary representation: 0=void, 1=full
  # graph representation from string definition
  decodedMap = deepcopy(initialMap)
  ds.decodeMapString(decodedMap)
  nxGraph = ds.graphUtils.binaryToGraph(ds.graphUtils.stringToBinary(utils.listToStr(decodedMap)))
  nx.set_node_attributes(
    nxGraph,
    {
      hexID: {
        "initialType": torch.tensor(initialHexType, dtype = torch.int32),
        "optimalType": torch.tensor(optimalHex, dtype = torch.int32)
      } for (hexID, (initialHexType, optimalHex)) in enumerate(nodeFeature)
    }
  )
  # nx.set_edge_attributes(nxGraph, [1], name = "name")
  # build dgl graph from networkx and append to graph list.
  # indicate attributes to copy over. graph is made directed
  # with networkx.Graph.to_directed() (bidirectional edges).
  # dgl.add_self_loop() avoids problem with DGL convolution
  # in graphs with nodes to which no other
  # nodes point (or isolated nodes)
  DGLdataset.append(
    dgl.add_self_loop(
      dgl.from_networkx(
        nx.to_directed(nxGraph),
        ["initialType", "optimalType"]
      )
    )
  )
  if OSPlength.ndim == 0:
    OSPlengthList.append(OSPlength)
  elif OSPlength.ndim == 1:
    OSPlengthList.append(OSPlength[0].numpy())
  else:
    print("ANOTHER DIM")
dgl.save_graphs( # save dataset
  "./DGLgraphData.bin", DGLdataset,
  {"gLabel": torch.tensor(OSPlengthList)}
)
#%% Check dataset file
graphList, label = dgl.load_graphs("./DGLgraphData.bin")
print(f"type(graphList): {type(graphList)}")
print(f"len(graphList): {len(graphList)}")
[print(g) for g in graphList[:5]]
print(f"\ntype(label): {type(label)}")
print(f"'label' keys: {label.keys()}")
print(f"type(label['gLabel']): {type(label['gLabel'])}")
print(f"label['gLabel'].size(): {label['gLabel'].size()}")
print(label['gLabel'][:5])