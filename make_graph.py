import networkx as nx
import matplotlib.pyplot as plt

lollipop = nx.lollipop_graph(3, 5)
plt.subplot(121)
nx.draw(lollipop, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw_spectral(lollipop, with_labels=True, font_weight='bold')


plt.show()
plt.savefig('graphie.png')