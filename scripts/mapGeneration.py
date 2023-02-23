#%%
# Script used to generate first maps

import time
import datetime
import utilities
#%%
# initialize dataset object
ds = utilities.dataset(
  filePath = "path",
  amountOfSamples = 0, nRow = 10, nCol = 10
)
randomMap = ds.generateMap()
#%%
def pathLength(binaryMap: list, solution_idx) -> int:
  """
  Objective to be maximized. Receives binary list
  indicating hexagons that are absent (0) or
  present (1). Returns length of shortest path
  """
  # guarantee a few basic map properties
  binaryMap = ds.graphUtils.mapCheck(binaryMap)
  # return total size of path (or zero in case of disconnection)
  return ds.graphUtils.totalSteps(binaryMap)

def callback_gen(ga_instance):
  """callback function"""
  print(f"""
  Gen: {
    ga_instance.generations_completed
  } Longest path: {
    ga_instance.best_solution()[1]
  } Time: {datetime.datetime.now().strftime("%H:%M:%S")}
  """)
  # store current best solution
  ds.graphUtils.storeMap(ga_instance.best_solution()[0])
  # store path length of current best solution
  ds.graphUtils.storeLength(ga_instance.best_solution()[1])

t1 = time.time()
utilities.optimize(50, 100, ds.graphUtils, pathLength, callback = callback_gen)
t2 = time.time()
print(round(t2 - t1, ndigits = 1), "seconds")
best = ds.graphUtils.bestMap()
ds.graphUtils.plotMesh(0)
ds.graphUtils.plotMesh(best[0])
print(best[1], "steps")