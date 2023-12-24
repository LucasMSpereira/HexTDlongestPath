# utilities for processing graphs and maps

import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import itertools
import utilities as utils
from hexalattice.hexalattice import *

class graphManager():
  """
  Collects methods and variables related to
  graphs. Nodes and lengths are stored as properties of this
  class for each section (spawn-flag-base consecutive pairs)
  """
  def __init__(
    self, initialMapString: str,
    numRow: int, numCol: int,
    rowsOfFlags: list, colsOfFlags: list
  ):
    # dictionary to store spots (spawn-flag-base)
    self.spot = {"spawn": [], "flag": [], "base": []}
    # store length of each map
    self.mapLength = []
    # number of rows and columns of full map
    self.nRow = numRow; self.nCol = numCol
    # get hexagon IDs of flags
    for (flagRow, flagCol) in zip(rowsOfFlags, colsOfFlags):
      self.spot["flag"].append(
        flagRow * self.nCol + flagCol
      )
    # list to store map definitions as binary lists
    self.mapDefinition = [
      self.stringToBinary(initialMapString, updateSpot = True)
    ]
    initialMapGraph = self.binaryToGraph(self.mapDefinition[0])
    # nodes of each section
    pathsInfo = self.allPaths(initialMapGraph)
    sectionLength = [len(sec) for sec in pathsInfo]
    # error in case of disconnection in initial map
    if math.prod(sectionLength) == 0:
      x = []
      for (section, l) in enumerate(sectionLength):
        if l == 0:
          x.append(section)
      raise Exception(f"Initial map has disconnection in section(s) {x}.")
    else: # total path length
       self.storeLength(self.totalSteps(pathsInfo))

  def allPaths(self, mapGraph):
    """Get paths for each section"""
    intermediatePath = []
    # A* in each section:
    for (source, target) in utils.pairwise(itertools.chain(*self.spot.values())):
      intermediatePath.append(self.aStarPathing(mapGraph, source, target))
    return intermediatePath

  def aStarPathing(self, _graph, _source, _target):
    """
    A* for shortest path between source and
    target nodes in given graph
    """
    try: # shortest path by A* and its length
      return nx.astar_path(_graph, _source, _target)
    except nx.NetworkXNoPath: # return 0 in case of disconnection
      return []

  def bestMap(self):
    """
    Return index of best map and number of
    steps of shortest path
    """
    # lengths of shortest path in each map
    totalLength = list(map(self.totalSteps, self.mapDefinition))
    # size of longest path
    longest = (0, 0)
    for length in enumerate(totalLength):
      if length[1] > longest[1]:
        longest = length
    return longest

  def binaryToConnections(self, binaryDefinition: list):
    """Hexagon connectivity from binary map definition"""
    # each key x refers to hexagon with ID x.
    # Starts from 0 and goes left to right, top to bottom.
    # respective values in dict contain the IDs of
    # neighbouring hexagons
    connectivity = {}
    hexa = 0
    while hexa < self.nRow * self.nCol:
      connectivity[hexa] = []
      nodeRow = int(math.ceil((hexa + 0.1)/self.nCol))
      if nodeRow % 2 != 0 or hexa == 0: # odd lines
        connectivity[hexa].append(hexa + 1)
        connectivity[hexa].append(hexa - self.nCol)
        connectivity[hexa].append(hexa - self.nCol - 1)
        connectivity[hexa].append(hexa - 1)
        connectivity[hexa].append(hexa + self.nCol - 1)
        connectivity[hexa].append(hexa + self.nCol)
        # first hexagon in odd lines
        if hexa in [self.nCol * i for i in range(self.nRow)]:
          connectivity[hexa][2] = -1
          connectivity[hexa][3] = -1
          connectivity[hexa][4] = -1
        if hexa in [self.nCol + self.nCol * i - 1 for i in range(self.nRow)]:
          connectivity[hexa][0] = -1
        if nodeRow == self.nRow:
          connectivity[hexa][5] = -1
          connectivity[hexa][4] = -1
      else: # even lines
        connectivity[hexa].append(hexa + 1)
        connectivity[hexa].append(hexa - self.nCol + 1)
        connectivity[hexa].append(hexa - self.nCol)
        connectivity[hexa].append(hexa - 1)
        connectivity[hexa].append(hexa + self.nCol)
        connectivity[hexa].append(hexa + self.nCol + 1)
        # left of first hexagon in even lines
        if hexa in [self.nCol * i for i in range(self.nRow)]:
          connectivity[hexa][3] = -1
        # last line of hexagons
        if nodeRow == self.nRow:
          connectivity[hexa][5] = -1
          connectivity[hexa][4] = -1
        # last hexagon in even lines
        if hexa in [self.nCol + self.nCol * i - 1 for i in range(self.nRow)]:
          connectivity[hexa][5] = -1
          connectivity[hexa][1] = -1
          connectivity[hexa][0] = -1
      hexa += 1
    # include holes
    voidHexas = [] # IDs of void hexagons
    for (hexID, value) in enumerate(binaryDefinition):
      if value == 0:
        voidHexas.append(hexID)
    # correct range of IDs (0 -> nHexas - 1).
    # remove references involving void hexagons as well
    for hexa in range(len(connectivity)):
      if hexa in voidHexas: # void hexagons don't reference other hexagons
        connectivity[hexa][:] = -np.ones(len(connectivity[hexa]), int)
      else:
        for hexaID in range(len(connectivity[hexa])):
          # nullify negative references. and no one references void hexagons
          if connectivity[hexa][
            hexaID] < 0 or connectivity[hexa][hexaID] in voidHexas or connectivity[
            hexa][hexaID] > self.nRow * self.nCol - 1:
            connectivity[hexa][hexaID] = -1
    # cleanup incorrect connections (-1)
    for (hexa, _) in enumerate(connectivity):
      for neighbor in reversed(range(len(connectivity[hexa]))):
        if connectivity[hexa][neighbor] == -1:
          connectivity[hexa].pop(neighbor)
    
    return connectivity

  def binaryToGraph(self, binaryDefinition: list):
    """
    Get adjacency graph from binary map representation
    """
    return nx.from_dict_of_lists(self.binaryToConnections(binaryDefinition))
  
  def interpretMLout(self, initialMap: list, predictedOSP):
    """Interpret OSP prediction as map"""
    # mapList = list(map(str, np.asarray(initialMap)[0]))
    optMap = copy.copy(initialMap)
    predictedOSP = np.asarray(predictedOSP)[0]
    for (index, code) in enumerate(initialMap):
      if code == "1":
        if predictedOSP[index] < 0.55:
          optMap[index] = "0"
    return optMap

  def mapCheck(self, mapDefinition: list):
    """
    1) Hexagons can't be added to initial map
    2) Spawn(s), flag(s) and base can't be blocked
    """
    for (hexagonID, code) in enumerate(self.mapDefinition[0]):
      if code == 0:
        mapDefinition[hexagonID] = 0
    
    for s in self.spot.values():
      for ID in s:
        mapDefinition[ID] = 1
    
    return mapDefinition

  def plotGraph(self, mapID: int):
    """Plot graph of a certain map"""
    nx.draw(self.binaryToGraph(self.mapDefinition[mapID]))

  def plotMesh(self, mapID: int):
    """Plot hexagon lattice"""
    mapSize = len(self.mapDefinition[mapID])
    # initial full map with shades of grey
    colors = np.zeros([mapSize, 3]) + 0.3 * np.array([
      np.fromiter(
        itertools.repeat(np.random.rand(), times = 3),
        float
      ) for _ in range(mapSize)
    ])
    totalPath = []
    spotSection = self.allPaths(self.binaryToGraph(self.mapDefinition[mapID]))
    for (secNum, section) in enumerate(spotSection):
      if secNum == 0 or secNum == len(section) - 1:
        for hexa in section:
          totalPath.append(hexa)
      else: # intermediate section
        for hexa in range(1, len(section) - 1):
          totalPath.append(section[hexa])
    # void hexagons in white
    for hexa in range(mapSize):
      if self.mapDefinition[mapID][hexa] == 0:
        colors[hexa, :] = 1
    # shortest path in shades of green
    for hexa in totalPath:
      colors[hexa, :] = [
        0, 1 - 0.45 * np.random.rand(), 0
      ]
    # spawn in red
    for spawn in self.spot["spawn"]:
      colors[spawn, :] = [0.8, 0, 0]
    # flags in orange
    for (flagNum, hexa) in enumerate(self.spot["flag"]):
      colors[hexa, :] = list(map(
        lambda x: x * (1 - 0.5 * flagNum / len(self.spot["flag"])),
        [1, 128 / 255, 0]
      ))
    # base in blue
    colors[self.spot["base"], :] = [0, 0, 0.8]
    
    _, ax = plt.subplots()
    create_hex_grid(
      nx = self.nCol, ny = self.nRow, do_plot = True,
      face_color = colors,
      edge_color = (1, 1, 1),
      h_ax = ax,
      plotting_gap = 0.05)

    ax.invert_yaxis()
    plt.show()

  def storeLength(self, length: int):
    """Store length of entire path"""
    self.mapLength.append(length)

  def storeMap(self, mapDef: list):
    """Store binary representation of map"""
    self.mapDefinition.append(list(map(int, mapDef)))

  def stringToBinary(self, mapString: str):
    """
    Get binary representation from (decoded)
    string of integers describing hexagons:
    0 - void; 1 - full
    2 - spawn; 3 - flag(s)
    4 - player's base
    """
    # binary list indicating free hexagons
    binaryList = []
    # populate binaryList and store ID of
    # 'spots' (spawn, flags, base)
    for (index, code) in enumerate(mapString):
      if int(code) == 0: # void hexagon
        binaryList.append(0)
      elif int(code) == 1: # free hexagon
        binaryList.append(1)
      elif int(code) == 2: # spawn
        binaryList.append(1)
        self.spot["spawn"].append(index)
      elif int(code) == 3: # flag
        binaryList.append(1)
      elif int(code) == 4: # player's base
        binaryList.append(1)
        self.spot["base"].append(index)
    return binaryList

  def totalSteps(self, binaryMap: list) -> int:
    """
    Return total number of steps in shortest path
    for given binary representation of map
    """
    mapGraph = self.binaryToGraph(binaryMap)
    # nodes of each section
    pathsInfo = self.allPaths(mapGraph)
    sectionLength = [len(sec) for sec in pathsInfo]
    # return 0 in case of path disconnection
    if math.prod(sectionLength) == 0:
      return 0
    else:
      # total path length
      # (discounting hexagon repetition among sections)
      return sum(sectionLength) - (len(sectionLength) - 1)
