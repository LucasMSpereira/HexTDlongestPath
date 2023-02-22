#%%
import h5py
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
  with h5py.File(hdf5File, 'r') as f:
      osp.extend(list(f['OSP_length']))
      initStr.extend(list(map(list, list(f['initial_string'].asstr()))))
      optStr.extend(list(map(list, list(f['optimal_string'].asstr()))))
#%%
for sample in range(amountOfSamples):
  # store initial map string
  initialDS[sample] = initStr[sample]
  # store optimal map string
  optimalDS[sample] = optStr[sample]
  # store length of OSP
  ospDS[sample] = osp[sample]
fileID.close()