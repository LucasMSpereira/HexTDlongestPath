import datetime
import utilities
import pygad

### test optimization ###

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
  # store current best solution
  sample.storeMap(ga_instance.best_solution()[0])
  # store path length of current best solution
  sample.storeLength(ga_instance.best_solution()[1])

populationSize = 20
ga_instance = pygad.GA(
    fitness_func = pathLength,
    num_generations = 10,
    callback_generation = callback_gen,
    sol_per_pop = populationSize,
    num_parents_mating = round(0.2 * populationSize),
    keep_elitism = 0,
    num_genes = sample.nRow * sample.nCol,
    gene_space = [0, 1],
    parent_selection_type = "rank",
    crossover_type = "single_point",
    mutation_type = "random",
    mutation_percent_genes = 10,
    suppress_warnings = True,
    save_solutions = True
  )

ga_instance.run() # run optimization
def testNumberOfMaps():
  assert len(sample.mapDefinition) == 11

# index and number of steps needed in best map
best = sample.bestMap()

# number of steps needed in initial map
initial = sample.totalSteps(sample.mapDefinition[0])
def testPathLengths():
  assert best[1] >= initial