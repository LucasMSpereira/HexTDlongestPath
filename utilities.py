# general utilities

import copy
from pathlib import WindowsPath
from hexalattice.hexalattice import *
import h5py
import random
import pygad
import statistics
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math
import itertools
import tensorflow as tf

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
    self.mapDefinition = [self.stringToBinary(initialMapString)]
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
    """
    Get paths for each section and
    their respective lengths
    """
    intermediatePath = []
    # A* in each section:
    for (source, target) in pairwise(itertools.chain(*self.spot.values())):
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
    colors = np.zeros([mapSize, 3]) + np.array([np.fromiter(itertools.repeat(np.random.rand(), times = 3), float) for _ in range(mapSize)]) * 0.3
    step = -1
    totalPath = []
    spotSection = self.allPaths(self.binaryToGraph(self.mapDefinition[mapID]))
    for (secNum, section) in enumerate(spotSection):
      if secNum == 0 or secNum == len(section) - 1:
        for hexa in section:
          step += 1
          totalPath.append(hexa)
      else: # intermediate section
        for hexa in range(1, len(section) - 1):
          step += 1
          totalPath.append(section[hexa])
    # void hexagons in white
    for hexa in range(mapSize):
      if self.mapDefinition[mapID][hexa] == 0:
        colors[hexa, :] = 1
    # shortest path in shades of green
    for (stepID, hexa) in enumerate(totalPath):
      colors[hexa, :] = [0, 1 - 0.45 * np.random.rand(), 0]
    # spawn in red
    for spawn in self.spot["spawn"]:
      colors[spawn, :] = [0.8, 0, 0]
    # flags in orange
    for (flagNum, hexa) in enumerate(self.spot["flag"]):
      colors[hexa, :] = list(map(lambda x: x * (1 - 0.5 * flagNum / len(self.spot["flag"])), [1, 128 / 255, 0]))
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

def optimize(
  population: int, generations: int, sampleObject,
  _pathLength, callback = None
) -> int:
  """Run EA for given population size and number of generations"""
  ga_instance = pygad.GA(
    fitness_func = _pathLength,
    num_generations = generations,
    callback_generation = callback,
    sol_per_pop = population,
    num_parents_mating = round(0.2 * population) if population > 10 else 1,
    keep_elitism = 0,
    num_genes = sampleObject.nRow * sampleObject.nCol,
    gene_space = [0, 1],
    parent_selection_type = "rank",
    crossover_type = "single_point",
    mutation_type = "random",
    mutation_percent_genes = 10,
    suppress_warnings = True,
    save_solutions = True
  )
  ga_instance.run() # run optimization
  return ga_instance.best_solution()[1]

# available in standard module itertools from version 3.10 onwards
def pairwise(iterable):
  # pairwise('ABCDEFG') --> AB BC CD DE EF FG
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)

def listToStr(l: list):
  return ''.join(map(str, l))

def flagsRowsAndCols(initialMapString: str, nCol: int, printInfo = True):
  """Print rows and columns of flags"""
  flagPos = {"row": [], "col": []}
  for (index, code) in enumerate(initialMapString):
    if int(code) == 3:
      row = int(math.ceil((index + 0.1) / nCol))
      flagPos["row"].append(row - 1)
      flagPos["col"].append(index - (row - 1) * nCol)
  if printInfo:
    for (flagNum, _) in enumerate(flagPos["row"]):
      print(f"""
      Flag in row {flagPos["row"][flagNum]} column {flagPos["col"][flagNum]}
      """)
  else:
    return flagPos

class dataManager():
  
  """Utilities for dataset generation and processing"""
  
  def __init__(self, amountOfSamples: int = 0, nRow: int = 0, nCol: int = 0):
    self.nRow = nRow # number of rows in maps
    self.nCol = nCol # number of cols in maps
    self.graphUtils = 0
    self.attempt = 0
    if amountOfSamples != 0:
      self.nSample = amountOfSamples # number of samples to be generated
      # create hdf5 file to store data
      self.fileID = h5py.File(
        f"""{str(WindowsPath("C:/Users/kaoid/Desktop/HexTDdataset"))}\\{str(
          random.randint(0, 9000)
        )}_{str(self.nRow)}r{str(self.nCol)}c{str(amountOfSamples)}.hdf5""",
        'w'
      )
      # dataset to store initial map string
      self.initialDS = self.fileID.create_dataset(
        "initial_string", (amountOfSamples,), dtype = h5py.string_dtype(length = self.nRow * self.nCol)
      )
      # dataset to store optimal map string
      self.optimalDS = self.fileID.create_dataset(
        "optimal_string", (amountOfSamples,), dtype = h5py.string_dtype(length = self.nRow * self.nCol)
      )
      # dataset to store length of optimal shortest path (OSP)
      self.ospDS = self.fileID.create_dataset("OSP_length", (amountOfSamples,), dtype = 'int')
  
  def conciseMapEncoding(self, mapIndex: int) -> str:
    """
    Alternative map representation that encodes
    map topology (0s and 1s) and spots (spawn-flags-base)
    """
    for spotClass in self.graphUtils.spot.keys():
      for (pos, spotID) in enumerate(self.graphUtils.spot[spotClass]):
        if spotClass == "spawn":
          self.graphUtils.mapDefinition[mapIndex][spotID] = 2
        elif spotClass == "flag":
          self.graphUtils.mapDefinition[mapIndex][spotID] = 3 + pos
        elif spotClass == "base":
          self.graphUtils.mapDefinition[mapIndex][spotID] = 3 + len(self.graphUtils.spot["flag"])
    return listToStr(self.graphUtils.mapDefinition[mapIndex])

  def decodeMapString(self, encodedMapString: list) -> list:
    """
    Decode map string from HDF5 file back to graphManager format.
    Also returns 'spot' dictionary.
    """
    spot = {"spawn": [], "flag": [], "base": []}
    flagDict = {}
    intMap = list(map(int, encodedMapString))
    maxCode = max(intMap)
    for (hexID, code) in enumerate(intMap):
      if code == 2: # spawn
        spot["spawn"].append(hexID)
      elif code == maxCode: # player's base
        encodedMapString[hexID] = "4"
        spot["base"].append(hexID)
      elif code != 0 and code != 1: # flags
        encodedMapString[hexID] = "3"
        flagDict[code] = hexID
    flagKey = list(flagDict.keys())
    flagKey.sort()
    spot["flag"] = [flagDict[i] for i in flagKey]
    return spot

  def generateDataset(self):
    """Generate dataset with requested amount of samples"""
    for sample in range(self.nSample):
      OSPlength = 0
      while OSPlength == 0: # filter problematic OSP
        print(f"Generating sample {sample + 1}")
        self.generateMap() # create random map
        # index and OSP length of optimal map
        optimalMapIndex, OSPlength = self.optimizeSample()
      self.saveDataPoint(sample, optimalMapIndex, OSPlength)
    self.fileID.close()

  def generateMap(self):
      """Generate random map for current sample"""
      mapCheck = False
      # generate random conectivity and make verifications
      while not mapCheck:
        connection, spotID, binary = self.randomConnectivity(self.nRow, self.nCol)
        # discard map in case of initial string problem
        if connection == 0:
          break
        # discard map in case of disconnection
        if self.graphUtils.totalSteps(binary) == 0:
          break
        # discard map in case of spot adjacency
        for (reference, neighbor) in itertools.permutations(spotID, r = 2):
          if neighbor in connection[reference]:
            break
        else:
          mapCheck = True
      try:
        # attempt to create graph representation of map
        nx.from_dict_of_lists(connection)
      except: # discard map in case of isolated hexagon
        return self.generateMap()

  def optimizeSample(self):
    """
    Optimize map of current sample. Return index of
    this sample's optimal map, and OSP length
    """
    def pathLength(binaryMap: list, solution_idx) -> int:
      """
      Objective to be maximized. Receives binary list
      indicating hexagons that are absent (0) or
      present (1). Returns length of shortest path
      """
      # guarantee a few basic map properties
      binaryMap = self.graphUtils.mapCheck(binaryMap)
      # return total size of path (or zero in case of disconnection)
      return self.graphUtils.totalSteps(binaryMap)

    def callback_gen(ga_instance):
      """callback function"""
      # store current best solution
      self.graphUtils.storeMap(ga_instance.best_solution()[0])
      # store path length of current best solution
      self.graphUtils.storeLength(ga_instance.best_solution()[1])

    # run evolutionary optimization for 100 generations with
    # population size of 50 individuals
    optimize(50, 100, self.graphUtils, pathLength, callback = callback_gen)
    # return index and OSP length of best map
    return self.graphUtils.bestMap()

  def readHDF5file(self, hdf5Name, decode = True):
    """
    From hdf5 file, retrieve initial and optimal map definitions as used by
    graphManager objects, OSP lengths, flag rows and columns, and map dimensions.
    'decode' kwarg controls if map strings are decoded
    """
    # read data from file
    filePath = str(WindowsPath(".")) + "\\" + hdf5Name
    with h5py.File(filePath, 'r') as f:
      osp = list(f['OSP_length'])
      initStr = list(map(list, list(f['initial_string'].asstr())))
      optStr = list(map(list, list(f['optimal_string'].asstr())))
    if not decode:
      encodedInitStr = copy.deepcopy(initStr)
      encodedOptimalStr = copy.deepcopy(optStr)
    # get number of rows and columns in map
    numRow = self.nRow
    numCol = self.nCol
    basicInitStr, basicOptimalStr, flagRow, flagCol = [], [], [], []
    # iterate in samples
    for (initialSampleStr, optimalSampleStr) in zip(initStr, optStr):
      # get initial map string in basic format
      sampleSpot = self.decodeMapString(initialSampleStr)
      basicInitStr.append(listToStr(initialSampleStr))
      # get optimal map string in basic format
      self.decodeMapString(optimalSampleStr)
      basicOptimalStr.append(listToStr(optimalSampleStr))
      # get sorted positions of flags
      sampleFlagRow, sampleFlagCol = [], []
      for flagID in sampleSpot["flag"]:
        row = int(math.ceil((flagID + 0.1) / numCol))
        sampleFlagRow.append(row - 1)
        sampleFlagCol.append(flagID - (row - 1) * numCol)
      flagRow.append(sampleFlagRow)
      flagCol.append(sampleFlagCol)
    return {
      "initStr": basicInitStr if decode else encodedInitStr,
      "optimalStr": basicOptimalStr if decode else encodedOptimalStr,
      "numRow": numRow,
      "numCol": numCol,
      "flagRow": flagRow,
      "flagCol": flagCol,
      "osp": osp
    }

  def randomConnectivity(self, nRow: int, nCol: int, flagAmount: int = 2, density: float = 0.8):
    """Create random connectivity for current sample"""
    self.attempt += 1
    mapSize = nRow * nCol
    fullHexa = math.ceil(mapSize * density)
    # define basic map topology
    randomMap = list(np.ones(fullHexa, dtype = int))
    randomMap.extend(list(np.zeros(mapSize - fullHexa, dtype = int)))
    random.shuffle(randomMap)
    # choose spawn, flag(s) and player's base locations
    indices = random.sample(range(0, mapSize), k = flagAmount + 2)
    for (pos, index) in enumerate(indices):
      if pos == 0: # spawn in first index
        randomMap[index] = 2
      elif pos == len(indices) - 1: # player's base in last index
        randomMap[index] = 4
      else: # intermediary indices recieve flags
        randomMap[index] = 3
    mapString = listToStr(randomMap)
    try:
      self.graphUtils = graphManager(mapString, nRow, nCol, 
        *list(flagsRowsAndCols(
          mapString, nCol, printInfo = False
      ).values()))
    except:
      return 0, 0, 0
    # return connectivity for map generated, and indices
    return self.graphUtils.binaryToConnections(
      self.graphUtils.mapDefinition[0]
    ), indices, self.graphUtils.mapDefinition[0]

  def saveDataPoint(self, sampleIndex: int, optMapIndex: int, ospLength: int):
    """Store sample in dataset"""
    # store concise encoding of initial map string
    self.initialDS[sampleIndex] = self.conciseMapEncoding(0)
    # store concise encoding of optimal map string
    self.optimalDS[sampleIndex] = self.conciseMapEncoding(optMapIndex)
    # store length of OSP
    self.ospDS[sampleIndex] = ospLength

  def studyHDF5file(self, hdf5Name):
    """
    Check HDF5 data file for problems
    """
    # read data from file
    filePath = str(WindowsPath("C:/Users/kaoid/Desktop/HexTDdataset")) + "\\" + hdf5Name
    with h5py.File(filePath, 'r') as f:
      osp = list(f['OSP_length'])
      initStr = list(map(list, list(f['initial_string'].asstr())))
      optStr = list(map(list, list(f['optimal_string'].asstr())))
    # get number of rows and columns in map
    rIndex = hdf5Name.find("r")
    cIndex = hdf5Name.find("c")
    numCol = int(hdf5Name[rIndex + 1 : cIndex])
    quart = statistics.quantiles(osp)
    print(f"""OSP lengths ({len(osp)} total):
      Minimum: {min(osp)}
      First quartile: {quart[0]}
      Second quartile: {quart[1]}
      Third quartile: {quart[2]}
      Maximum: {max(osp)}
      Mean: {statistics.mean(osp):.3E}
      Standard deviation: {statistics.stdev(osp):.3E}
    """)
    basicInitStr, basicOptimalStr = [], []
    # iterate in samples
    for (sample, (initialSampleStr, optimalSampleStr)) in enumerate(zip(initStr, optStr)):
      # get initial map string in basic format
      if len(listToStr(initialSampleStr)) * len(listToStr(optimalSampleStr)) == 0:
          raise Exception(f"Sample {sample} in \"{hdf5Name}\".hdf5 has empty map definition.")
      sampleSpot = self.decodeMapString(initialSampleStr)
      for pos in sampleSpot.values():
        if len(pos) == 0:
          raise Exception(f"Sample {sample} in \"{hdf5Name}\".hdf5 has problematic 'spot' dict.")
      basicInitStr.append(listToStr(initialSampleStr))
      # get optimal map string in basic format
      self.decodeMapString(optimalSampleStr)
      basicOptimalStr.append(listToStr(optimalSampleStr))
      # get sorted positions of flags
      flagList = []
      for flagID in sampleSpot["flag"]:
        row = int(math.ceil((flagID + 0.1) / numCol))
        flagList.append(row - 1)
        flagList.append(flagID - (row - 1) * numCol)
      if len(flagList) == 0:
        raise Exception(f"Sample {sample} in \"{hdf5Name}\".hdf5 has problematic flag positioning.")

  def TFdata(self, fileName = "dataset.hdf5"):
    """
    Create tensorflow Dataset object from HDF5 dataset file
    """
    dataDict = self.readHDF5file(fileName, decode = False)
    tfDataset = 0
    for (index, (initStr, optimalStr, osp)) in enumerate(zip(dataDict["initStr"], dataDict["optimalStr"], dataDict["osp"])):
      initStr = list(map(int, initStr))
      optimalStr = list(map(int, optimalStr))
      if index == 0:
        tfDataset = tf.data.Dataset.from_tensors((initStr, optimalStr, osp))
      else:
        tfDataset = tfDataset.concatenate(tf.data.Dataset.from_tensors((initStr, optimalStr, osp)))
    return tfDataset