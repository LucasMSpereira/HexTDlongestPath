#%%
# Script used to study statistics of map optimization performance

import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import datetime
import utilities
import graph_manager
#%%
mapRows = 10
mapCols = 10
sample = graph_manager.graphManager(
  # initial map string
  "4111011003111111101011011111101111101110111111111111111111110111011111011111101101011111113001101112",
  mapRows, mapCols, # n° of rows and cols
  [0, 9], # rows of each flag
  [9, 0] # columns of each flag
)
#%%
def pathLength(binaryMap: list, solution_idx) -> int:
  """
  Objective to be maximized. Receives binary list
  indicating hexagons that are absent (0) or
  present (1). Returns length of shortest path
  """
  # guarantee a few basic map properties
  binaryMap = sample.mapCheck(binaryMap)
  # return total size of path (or zero in case of disconnection)
  return sample.totalSteps(binaryMap)

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
  sample.storeMap(ga_instance.best_solution()[0])
  # store path length of current best solution
  sample.storeLength(ga_instance.best_solution()[1])
# %%
popSizes = [80 * i for i in range(1, 4)]
numOfGenerations = [150 * i for i in range(1, 4)]
numOfSolutions = []; longestPath = []
for (index, (pop, gen)) in enumerate(itertools.product(popSizes, numOfGenerations)):
  print(f"{round(index / 9 * 100)}%")
  longestPath.append(
    utilities.optimize(pop, gen, sample, pathLength)
  )
  numOfSolutions.append(pop * gen)
#%%
corr = scipy.stats.pearsonr(longestPath, numOfSolutions)
fig, ax = plt.subplots()
a, b = np.polyfit(numOfSolutions, longestPath, deg = 1)
y = a * np.array(numOfSolutions) + b
nonZeroPaths = len(list(itertools.filterfalse(lambda x: x != 0, longestPath)))
print(
  f"""
  {round(
    nonZeroPaths / len(longestPath) * 100
  )}% of runs didn't find a path.
  Pearson correlation coefficient of {round(corr.correlation, ndigits = 3)}.
  Average of {a:.2E} steps\solution, which
  leads to 1 extra step for every {round(1/a)} solutions."""
)
ax.plot(numOfSolutions, y)
ax.set_xlabel("Total solutions tried")
ax.set_ylabel("Longest path")
plt.scatter(numOfSolutions, longestPath)