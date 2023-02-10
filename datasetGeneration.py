#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy
import itertools
import datetime
import utilities
#%%
# initialize dataset object with dummy graphManager object
ds = utilities.dataset()
randomMap = ds.generateMap(10, 10)
ds.graphUtils.plotMesh(0)