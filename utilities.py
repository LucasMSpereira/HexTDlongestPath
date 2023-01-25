import math

def checkInitialMap(mapString):
  """
  Check if hexagons were added to initial map, which
  is in scope when pathLength() is defined.
  If that's the case, remove these hexagons
  """

def pathLength(mapString, solution_idx):
  """
  Objective to be maximized
  """
  # check if hexagons were added to initial map, which
    # is in scope when checkInitialMap() is defined
  # if that's the case, remove these hexagons
  mapString = checkInitialMap(mapString)
  # list of stops on the path (spawn-flags-base)
  stop = intermediateDestinies(mapString)
  # map string -> adjacency matrix -> graph
  mapGraph = stringToMap(mapString)
  # list with nodes in each section (spawn-flag-base consecutive pair)
  sectionLength = []
  # list with length of each section
  sectionPath = []
  # A* in each section:
  for (source, target) in stop:
    section = aStarPathing(mapGraph, source, target)
    sectionPath.append(section[0])
    sectionLength.append(section[1])
  # return 0 in case of path disconnection
  if math.prod(sectionLength) == 0:
    return 0
  else: # total path length
    return sum(sectionLength)

def aStarPathing(_graph, _source, _target):
  """
  A* for shortest path
  """
  try: # shortest path by A* and its length
    p = nx.astar_path(_graph, _source, _target)
    return p, len(p) - 1
  except nx.NetworkXNoPath: # return 0 in case of no path
    return [], 0