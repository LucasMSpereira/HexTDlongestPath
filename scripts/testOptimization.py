#%%
# Script used to test map optimization parameters

import time
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

populationSize = 40
ga_instance = pygad.GA(
  fitness_func = pathLength,
  # on_generation = callback_gen,
  num_generations = 150,
  sol_per_pop = populationSize,
  num_parents_mating = round(0.1 * populationSize) if populationSize > 10 else 1,
  keep_elitism = 0,
  num_genes = sample.nRow * sample.nCol,
  gene_space = [0, 1],
  parent_selection_type = "rank",
  crossover_type = "single_point",
  mutation_type = "random",
  mutation_percent_genes = 10,
  suppress_warnings = True,
  save_solutions = True,
  parallel_processing = 8
)
t1 = time.time()
ga_instance.run() # run optimization
t2 = time.time()
# development of fitness value over optimization
ga_instance.plot_fitness()
# Number of new solutions per generation
ga_instance.plot_new_solution_rate()
# index and number of steps needed in best map
best = sample.bestMap()
# number of steps needed in initial map
initial = sample.totalSteps(sample.mapDefinition[0])
print(f"""
{round(t2 - t1, ndigits = 1)} s.
Best map takes {best[0]} steps.
Initial map takes {initial} steps.
Improvement of {round((best[0]/initial - 1) * 100, ndigits = 1)}%.
{ga_instance.best_solution()[1]} steps.
""")
#%%
# store current best solution
sample.storeMap(ga_instance.best_solution()[0])
# store path length of current best solution
sample.storeLength(ga_instance.best_solution()[1])
sample.plotMesh(0)
sample.plotMesh(sample.bestMap()[1])