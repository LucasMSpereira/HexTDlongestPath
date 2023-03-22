# general utilities

import pygad
import math
import itertools

def optimize(
  population: int, generations: int, sampleObject,
  _pathLength, callback = None
) -> int:
  """Run EA for given population size and number of generations"""
  ga_instance = pygad.GA(
    fitness_func = _pathLength,
    num_generations = generations,
    callback_generation = callback,
    sol_per_pop = population,
    num_parents_mating = round(0.2 * population) if population > 10 else 1,
    keep_elitism = 0,
    num_genes = sampleObject.nRow * sampleObject.nCol,
    gene_space = [0, 1],
    parent_selection_type = "rank",
    crossover_type = "single_point",
    mutation_type = "random",
    mutation_percent_genes = 10,
    suppress_warnings = True,
    save_solutions = True
  )
  ga_instance.run() # run optimization
  return ga_instance.best_solution()[1]

# available in standard module itertools from version 3.10 onwards
def pairwise(iterable):
  # pairwise('ABCDEFG') --> AB BC CD DE EF FG
  a, b = itertools.tee(iterable)
  next(b, None)
  return zip(a, b)

def listToStr(l: list):
  """
  Turn all elements in a list into strings. Then concatenate
  them, and return the single resulting string.
  """
  return ''.join(map(str, l))

def flagsRowsAndCols(initialMapString: str, nCol: int, printInfo = True):
  """Print rows and columns of flags"""
  flagPos = {"row": [], "col": []}
  for (index, code) in enumerate(initialMapString):
    if int(code) == 3:
      row = int(math.ceil((index + 0.1) / nCol))
      flagPos["row"].append(row - 1)
      flagPos["col"].append(index - (row - 1) * nCol)
  if printInfo:
    for (flagNum, _) in enumerate(flagPos["row"]):
      print(f"""
      Flag in row {flagPos["row"][flagNum]} column {flagPos["col"][flagNum]}
      """)
  else:
    return flagPos