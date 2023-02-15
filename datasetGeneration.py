#%%
import time
from pathlib import WindowsPath
import utilities
#%%
# initialize dataset object
sampleNum = 200
ds = utilities.dataset(
  filePath = WindowsPath("C:/Users/kaoid/Desktop/HexTDdataset"),
  amountOfSamples = sampleNum, nRow = 10, nCol = 10
)
#%%
t1 = time.time()
ds.generateDataset()
t2 = time.time()
#%%
print(f"""
Total time: {round(t2 - t1)} s ({round((t2 - t1) / 3600, ndigits = 1)} h)
Average time per sample: {round((t2 - t1) / sampleNum, ndigits = 1)} s
Average attempts per sample: {round(ds.attempt / sampleNum, ndigits = 1)}
""")