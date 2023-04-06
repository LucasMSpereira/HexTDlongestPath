#%%
# Script used to combine HDF5 files into one, and analyse this final file

import h5py
from collections import Counter
import utilities
import data_utils
from pathlib import WindowsPath
#%%
amountOfSamples = 20000
nRow = nCol = 10
fileID = h5py.File(
  f"""{str(WindowsPath("C:/Users/kaoid/Desktop/HexTDdataset"))}\{nRow}r{nCol}c{amountOfSamples}.hdf5""",
  'w'
)
# dataset to store initial map string
initialDS = fileID.create_dataset(
  "initial_string", (amountOfSamples,),
  dtype = h5py.string_dtype(length = nRow * nCol)
)
# dataset to store optimal map string
optimalDS = fileID.create_dataset(
  "optimal_string", (amountOfSamples,), dtype = h5py.string_dtype(length = nRow * nCol)
)
# dataset to store length of optimal shortest path (OSP)
ospDS = fileID.create_dataset("OSP_length", (amountOfSamples,), dtype = 'int')
#%%
osp, initStr, optStr = [], [], []
for (index, hdf5File) in enumerate(WindowsPath("C:/Users/kaoid/Desktop/HexTDdataset").iterdir()):
  if hdf5File.name != "10r10c20000.hdf5":
    with h5py.File(hdf5File, 'r') as f:
        osp.extend(list(f['OSP_length']))
        print(len(list(f['OSP_length'])))
        initStr.extend(list(map(list, list(f['initial_string'].asstr()))))
        print(len(list(map(list, list(f['initial_string'].asstr())))))
        optStr.extend(list(map(list, list(f['optimal_string'].asstr()))))
        print(len(list(map(list, list(f['optimal_string'].asstr())))))
#%%
for sample in range(amountOfSamples):
  # store initial map string
  initialDS[sample] = utilities.listToStr(initStr[sample])
  # store optimal map string
  optimalDS[sample] = utilities.listToStr(optStr[sample])
  # store length of OSP
  ospDS[sample] = osp[sample]
fileID.close()
#%%
f1 = h5py.File(WindowsPath("C:/Users/kaoid/Desktop/HexTDdataset/10r10c20000.hdf5"), 'r')
osp = list(f1['OSP_length'])
initStrFile = list(map(list, list(f1['initial_string'].asstr())))
optStrFile = list(map(list, list(f1['optimal_string'].asstr())))
countOSP = dict(Counter(osp))
l = list(countOSP.keys())
l.sort()
for k in l:
  print(k, countOSP[k])
ds = data_utils.dataManager(
  amountOfSamples = 0, nRow = 10, nCol = 10
)
ds.studyHDF5file("10r10c20000.hdf5")
