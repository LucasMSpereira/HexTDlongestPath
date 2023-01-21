import numpy
import matplotlib.pyplot as plt
from hexalattice.hexalattice import *

hex_centers, _ = create_hex_grid(nx=5,
                                 ny=5,
                                 do_plot=True)
                                 
plt.show()