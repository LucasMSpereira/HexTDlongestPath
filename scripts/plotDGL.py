#%% Script to test and plot samples from
# both dataset formats
import utilities as utils
import random
import data_utils
#%% Get data
# dataManager instance for data utils
rowAmount = colAmount = 10
ds = data_utils.dataManager(0, rowAmount, colAmount)
# get dgl data (graphs)
batchLoader = ds.readDGLdataset(trainPercent = 1, batchSize = 1)
nSamples = 10 # number of plots from each dataset version
#%% Test HDF5
encodedHDF5 = ds.readHDF5file("dataset.hdf5", decode = False)
# decodedHDF5 = ds.readHDF5file("dataset.hdf5", decode = True)
ds.generateMap() # assign graphUtils property
gm = ds.graphUtils # alias
for _ in range(nSamples):
  m = random.choice(encodedHDF5["optimalStr"])
  m = utils.strToList(m)
  gm.spot = ds.decodeMapString(m)
  m = utils.listToStr(m)
  gm.mapDefinition[0] = gm.stringToBinary(m)
  gm.plotMesh(0)
#%% Plots from HDF5
print("\n")
ds.generateMap() # reset graphUtils property
gm = ds.graphUtils
iterBatches = iter(batchLoader)
for plotCount in range(nSamples):
  m = next(iterBatches)
  mapStr = []
  # iterate in current graph
  for i, n in enumerate(m[0].nodes):
    if i == m[0].num_nodes():
      break
    # build map string from node attribute
    mapStr.append(str(n.data['optimalType'].item()))
    # mapStr.append(str(n.data['initialType'].item()))
  mapStr = utils.strToList(mapStr)
  gm.spot = ds.decodeMapString(mapStr)
  gm.mapDefinition[0] = gm.stringToBinary(mapStr)
  gm.plotMesh(0)