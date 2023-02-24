#%%
# Script used to generate dataset HDF5 files and analyse them

import time
import random
import utilities
from pathlib import WindowsPath
# from importlib.machinery import SourceFileLoader
# utilities = SourceFileLoader("utilities", str(WindowsPath(".\\utilities.py"))).load_module()
# import utilities
#%%
# initialize dataset object
sampleNum = 0
ds = utilities.dataset(
  amountOfSamples = sampleNum, nRow = 10, nCol = 10
)
#%%
if sampleNum != 0:
  t1 = time.time()
  ds.generateDataset()
  t2 = time.time()
#%%
if sampleNum != 0:
  print(f"""
  Total time: {round(t2 - t1)} s ({round((t2 - t1) / 3600, ndigits = 1)} h)
  Average time per sample: {round((t2 - t1) / sampleNum, ndigits = 1)} s
  Average attempts per sample: {round(ds.attempt / sampleNum, ndigits = 1)}
  """)
#%%
if sampleNum == 0:
  for (index, hdf5File) in enumerate(WindowsPath("C:/Users/kaoid/Desktop/HexTDdataset").iterdir()):
    # print maps of k sample(s) from current file
    print(index + 1, hdf5File.name)
    init, opt, nRow, nCol, flagRow, flagCol, osp = ds.readHDF5file(hdf5File.name)
    for sample in random.sample(range(1000), k = 1):
      initMapObj = utilities.graphManager(
        init[sample], nRow, nCol,
        flagRow[sample], flagCol[sample]
      )
      initMapObj.plotMesh(0)
      print(initMapObj.bestMap()[1])
      optMapObj = utilities.graphManager(
        opt[sample], nRow, nCol,
        flagRow[sample], flagCol[sample]
      )
      optMapObj.plotMesh(0)
      print(optMapObj.bestMap()[1])
      print(osp[sample])
    # look for problems in dataset file
    ds.studyHDF5file(hdf5File.name)