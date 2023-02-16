#%%
import time
import utilities
#%%
# initialize dataset object
sampleNum = 10
ds = utilities.dataset(
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
init, opt, nRow, nCol, flagRow, flagCol = ds.readHDF5file("1785_10r10c4")
#%%
for sample in range(sampleNum):
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