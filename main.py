#%%
# import pygad
import datetime
import utilities
#%%
sample = utilities.graphManager(
  # initial map string
  "4111011003111111101011011111101111101110111111111111111111110111011111011111101101011111113001101112",
  10, 10, # n° of rows and cols
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
  # map string -> adjacency matrix -> graph
  mapGraph = sample.binaryToGraph(binaryMap)
  # nodes of each section
  pathsInfo = sample.allPaths(mapGraph)
  # return total size of path (or zero in case of disconnection)
  return sample.totalSteps(pathsInfo)

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

# ga_instance = pygad.GA(
#   fitness_func = pathLength,
#   callback_generation = callback_gen
# )
# ga_instance.run()