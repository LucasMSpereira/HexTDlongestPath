from hexalattice.hexalattice import *
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import math
import itertools

class graphManager():
  """
  Collects methods and variables related to
  graphs. Nodes and lengths are stored as properties of this
  class for each section (spawn-flag-base consecutive pairs)
  """
  def __init__(
    self, initialMapString: str,
    numRow: int, numCol: int,
    rowsOfFlags: list, colsOfFlags: int
  ):
    self.flagRows = rowsOfFlags # rows of each flag
    self.flagCols = colsOfFlags # columns of each flag
    # dictionary to store spots (spawn-flag-base)
    self.spot = {"spawn": [], "flag": [], "base": []}
    # store length of each map
    self.mapLength = []
    # number of rows and columns of full map
    self.nRow = numRow; self.nCol = numCol
    # list to store map definitions as binary lists
    self.mapDefinition = [self.stringToBinary(initialMapString)]
    initialMapGraph = self.binaryToMap(self.mapDefinition[0])
    # nodes and length of each section
    pathsInfo = self.allPaths(initialMapGraph)
    sectionLength = [sec[1] for sec in pathsInfo]
    # error in case of disconnection in initial map
    if math.prod(sectionLength) == 0:
      x = []
      for (section, l) in enumerate(sectionLength):
        if l == 0:
          x.append(section)
      raise Exception(f"Initial map has disconnection in section(s) {x}.")
    else: # total path length
       self.mapLength.append(sum(sectionLength))

  def aStarPathing(self, _graph, _source, _target):
    """
    A* for shortest path between source and
    target nodes in given graph
    """
    try: # shortest path by A* and its length
      p = nx.astar_path(_graph, _source, _target)
      return p, len(p)
    except nx.NetworkXNoPath: # return 0 in case of disconnection
      return [], 0

  def binaryToMap(self, binaryDefinition: list):
    """
    Get adjacency graph from binary map representation
    """
    # adjacency between hexagons. list of lists.
    # list x refers to hexagon with ID x, and
    # contains the IDs of its neighbors
    connectivity = {}
    hexa = -1
    while hexa < self.nRow * self.nCol:
      hexa += 1
      connectivity[hexa] = []
      nodeRow = int(math.ceil(hexa/self.nCol))
      if nodeRow % 2 != 0: # odd lines
        connectivity[hexa].append(hexa + 1)
        connectivity[hexa].append(hexa - self.nCol)
        connectivity[hexa].append(hexa - self.nCol - 1)
        connectivity[hexa].append(hexa - 1)
        connectivity[hexa].append(hexa + self.nCol - 1)
        connectivity[hexa].append(hexa + self.nCol)
        # first hexagon in odd lines
        if hexa in [1 + self.nCol * i for i in range(self.nRow)]:
          connectivity[hexa][2] = 0
          connectivity[hexa][3] = 0
          connectivity[hexa][4] = 0
        if hexa in [self.nCol + self.nCol * i for i in range(self.nRow)]:
          connectivity[hexa][0] = 0 # right of last hexagon in odd lines
        if nodeRow == self.nRow:
          connectivity[hexa][4] = 0
          connectivity[hexa][5] = 0
      else: # even lines
        connectivity[hexa].append(hexa + 1)
        connectivity[hexa].append(hexa - self.nCol + 1)
        connectivity[hexa].append(hexa - self.nCol)
        connectivity[hexa].append(hexa - 1)
        connectivity[hexa].append(hexa + self.nCol)
        connectivity[hexa].append(hexa + self.nCol + 1)
        # left of first hexagon in even lines
        if hexa in [1 + self.nCol * i for i in range(self.nRow)]:
          connectivity[hexa][3] = 0
        # last line of hexagons
        if nodeRow == self.nRow:
          connectivity[hexa][4] = 0; connectivity[hexa][5] = 0
        # last hexagon in even lines
        if hexa in [self.nCol + self.nCol * i for i in range(self.nRow)]:
          connectivity[hexa][0] = 0; connectivity[hexa][1] = 0; connectivity[hexa][5] = 0
    # include holes
    voidHexas = [] # IDs of void hexagons
    for (hexID, value) in enumerate(binaryDefinition):
      if value == 0:
        voidHexas.append(hexID)
    # replace negative IDs with zeros.
    # remove references involving void hexagons as well
    for hexa in range(len(connectivity)):
      if hexa in voidHexas: # void hexagons don't reference other hexagons
        for hexaID in range(len(connectivity[hexa])):
          connectivity[hexa][hexaID] = 0
      else:
        for hexaID in range(len(connectivity[hexa])):
          # nullify negative references. and no one references void hexagons
          if connectivity[hexa][hexaID] < 0 or connectivity[hexa][hexaID] in voidHexas:
            connectivity[hexa][hexaID] = 0
    # return graph
    return nx.from_dict_of_lists(connectivity)
  
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
    nx.draw(self.binaryToMap(self.mapDefinition[mapID]))

  def plotMesh(self, mapID: int):
    """Plot hexagon lattice"""
    colors = np.zeros([len(self.mapDefinition[mapID]), 3])
    for spawn in self.spot["spawn"]:
      colors[spawn, :] = 0
    for flag in self.spot["flag"]:
      colors[flag, :] = 0.4
    colors[self.spot["base"], :] = 0.8
    hex_centers, _ = create_hex_grid(
      nx = self.nCol, ny = self.nRow, do_plot = True,
      face_color = colors,
      edge_color = (0.25, 0.25, 0.25),
      plotting_gap = 0.1)

    plt.show()

  def storeLength(self, length: int):
    """Store length of path"""
    self.sectionLength.append(length)

  def storeMap(self, mapDef: list):
    """Store map"""
    self.mapDefinition.append(mapDef)

  def storePath(self, nodeList: list):
    """Store node list"""
    self.sectionPath.append(nodeList)

  def stringToBinary(self, mapString: str):
    """
    In instantiation of class, get binary representation
    from string of integers describing hexagons:
    0 - void; 1 - full
    2 - spawn; 3 - flag
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
        self.spot["flag"].append(index)
      elif int(code) == 4: # player's base
        binaryList.append(1)
        self.spot["base"].append(index)
    return binaryList

  def allPaths(self, mapGraph):
    """
    Get paths for each section and
    their respective lengths
    """
    intermediatePath = []
    # A* in each section:
    for (source, target) in pairwise(itertools.chain(*self.spot.values())):
      intermediatePath.append(self.aStarPathing(mapGraph, source, target))
    return intermediatePath

# available in standard module itertools from version 3.10
def pairwise(iterable):
  # pairwise('ABCDEFG') --> AB BC CD DE EF FG
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)