# utilities for processing data

import copy
from pathlib import WindowsPath
import h5py
import random
import statistics
import numpy as np
import networkx as nx
import math
import itertools
import tensorflow as tf

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

  def TFdata(
      self, modelOutput: str, trainSplit: float = 0.9, fileName = "dataset.hdf5"
  ):
    """
    From HDF5 dataset file, create tensorflow Dataset objects
    for training/validation
    """
    fileDict = self.readHDF5file(fileName, decode = False)
    numberOfSamples = len(fileDict["osp"])
    # shuffled indices to access dataset in random order
    indices = list(range(0, numberOfSamples))
    random.shuffle(indices)
    trainInitStr, trainLabel = [], []
    valInitStr, valLabel = [], []
    lengthAndOSP = ([], [])
    trainIndex = math.ceil(numberOfSamples * trainSplit)
    if modelOutput != "both":
      print(f"""
        {trainIndex} samples for training
        {numberOfSamples - trainIndex} samples for validation
      """)
    # extract ML inputs and labels in shuffled order
    for (sampleNumber, index) in enumerate(indices):
      initStr, optimalStr, osp = fileDict["initStr"][index], fileDict["optimalStr"][index], fileDict["osp"][index]
      if sampleNumber <= trainIndex: # sample goes to training split
        if modelOutput == "both":
          continue
        elif modelOutput == "optimalPath":
          trainLabel.append(list(map(int, optimalStr)))
        elif modelOutput == "OSPlength":
          trainLabel.append(osp)
        trainInitStr.append(list(map(int, initStr)))
      else: # sample goes to validation split
        valInitStr.append(list(map(int, initStr)))
        if modelOutput == "both":
          lengthAndOSP[0].append(osp)
          lengthAndOSP[1].append(list(map(int, optimalStr)))
        elif modelOutput == "optimalPath":
          valLabel.append(list(map(int, optimalStr)))
        elif modelOutput == "OSPlength":
          valLabel.append(osp)
    if modelOutput == "both":
      return tf.data.Dataset.from_tensor_slices((valInitStr, *lengthAndOSP))
    else:
      # return tf Datasets for training and validation
      return (
        tf.data.Dataset.from_tensor_slices((trainInitStr, trainLabel)),
        tf.data.Dataset.from_tensor_slices((valInitStr, valLabel))
      )