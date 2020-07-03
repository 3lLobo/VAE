import networkx as nx 
import gmatch4py as gm
import matplotlib.pyplot as plt

ged = gm.GraphEditDistance(1,2,1,1)
# ged.set_attr_graph_used("theme","color") # Edge colors and node themes attributes will be used.

g1=nx.complete_bipartite_graph(5,4) 
g2=nx.complete_bipartite_graph(10,8)

# Visualize the graphs for general understanding
plt.subplot(121)
nx.draw_spectral(g1, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw_spectral(g2, with_labels=True, font_weight='bold')


plt.show()
plt.savefig('matchie.png')

# Let the magic happen. Match the two graphs.
result=ged.compare([g1,g2],None) 
# print(nx.adjacency_matrix(g1))
print(result)
print(ged.similarity(result))
