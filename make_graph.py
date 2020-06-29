import networkx as nx
import matplotlib.pyplot as plt

lollipop = nx.lollipop_graph(2, 3)
plt.subplot(121)
nx.draw(lollipop, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw_spectral(lollipop, with_labels=True, font_weight='bold')


plt.show()
plt.savefig('graphie.png')

print(nx.adjacency_matrix(lollipop))
graph = lollipop
graph.add_node(6)
graph.add_edge(6,1)
graph.add_attribute(6, 'a')
print(nx.nodes(graph))
print(nx.adjacency_matrix(graph))
print(nx.adjacency_data(graph))
print(nx.attr_matrix(graph))
print(nx.attribute_mixing_matrix(graph, '3'))
