#%%
import pygad
import datetime
import utilities
#%%
mapRows = 10
mapCols = 10
sample = utilities.graphManager(
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

#%%
ga_instance = pygad.GA(
  fitness_func = pathLength,
  on_generation = callback_gen,
  num_generations = 500,
  num_parents_mating = 4,
  sol_per_pop = 8,
  num_genes = mapRows * mapCols,
  gene_space = [0, 1],
  parent_selection_type = "sss",
  keep_parents = 1,
  crossover_type = "single_point",
  mutation_type = "random",
  mutation_percent_genes = 10,
  save_solutions = True
)
ga_instance.run()
ga_instance.plot_fitness()
ga_instance.plot_new_solution_rate()
best = sample.bestMap()
print(best)
sample.plotMesh(best[1])
# %%
