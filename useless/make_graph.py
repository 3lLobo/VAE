import networkx as nx
import matplotlib.pyplot as plt

lollipop = nx.lollipop_graph(2, 3)
plt.subplot(121)
nx.draw(lollipop, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw_spectral(lollipop, with_labels=True, font_weight='bold')


plt.show()
plt.savefig('graphie.png')

# print(nx.adjacency_matrix(lollipop

graph = lollipop
graph.add_node(6)
graph.add_edge(6,1)

FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)], peep='Pink')

for (u, v, wt) in FG.edges.data('weight'):
    if wt < 0.9: 
        print('(%d, %d, %.3f)' % (u, v, wt))
FG.nodes[1]['peep'] = 'Poop'      
FG.add_edges_from([(1,3),(2,5)], meep='Peep')
print(FG.nodes.data())
print(FG.edges.data())

print(FG.edge_attr_dict_factory())
print(FG.node_attr_dict_factory())
