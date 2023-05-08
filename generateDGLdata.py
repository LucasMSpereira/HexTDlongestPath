# Script used to generate annotated DGL graph
# dataset for graph transformer

import os
os.environ['DGLBACKEND'] = 'tensorflow'
from graph_manager import graphManager
from datetime import datetime
import dgl
import random
import numpy as np
import utilities as utils
import networkx as nx
import data_utils

random.seed(100)
rowAmount = colAmount = 10 # map dimensions
ds = data_utils.dataManager(0, rowAmount, colAmount)
goal = "optimalPath" # 'both', 'optimalPath', 'OSPlenth'
data = ds.TFdata(goal, trainSplit = 1)[0] # dataset
dataSize = len(data.batch(1))
DPGdataset = []
for (sampleID, (initialMap, optimalMap)) in enumerate(data.batch(1)):
  if sampleID % 1000 == 0:
    print(f"""Sample {sampleID}/{dataSize} Time: {datetime.now().strftime("%H:%M:%S")}""")
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
  # build dgl graph from networkx and append to graph list.
  # indicate attributes to copy over.
  # makes the graph directed with networkx.Graph.to_directed()
  # (bidirectional edges)
  DPGdataset.append(dgl.from_networkx(nxGraph, ["initialType", "optimalType"]))

dgl.save_graphs("./DGLgraphData.bin", DPGdataset)