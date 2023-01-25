#%%
import numpy
import utilities
import matplotlib.pyplot as plt
import networkx as nx
from hexalattice.hexalattice import *
#%%
G = nx.Graph()
G.add_edge("A", "B")
G.add_edge("B", "D")
G.add_edge("A", "C")
G.add_edge("C", "D")
G.add_edge("D", "E")
nx.draw(G)
nx.to_numpy_array(G)
#%%


print(aStarPathing(G, "A", "D"))
#%%
hex_centers, _ = create_hex_grid(nx=5, ny=5, do_plot=True)
plt.show()