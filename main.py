import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from cdt.data import AcyclicGraphGenerator
from CD_CSG import CD_CSG
import random

np.random.seed(0)
random.seed(0)
generator = AcyclicGraphGenerator('polynomial', noise='gaussian', noise_coeff=0.4, npoints=500, nodes=8)
data, graph1 = generator.generate()

ground_truth = np.array(nx.adj_matrix(graph1).todense())
graph = graph1.to_undirected()

cs = CD_CSG()
'''
# predict from a graph
DAG, adj = cs.predict_graph(data, graph)
'''
# predict from an adjacent matrix
A = np.array(nx.adj_matrix(graph).todense())
DAG, adj = cs.predict_adj(data, A)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.title('ground truth')
nx.draw(graph1, pos=nx.circular_layout(graph1), node_color='g', edge_color='r', with_labels=True, font_size=18, width=2, node_size=1000)
plt.subplot(122)
plt.title('CD-CSG result')
nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True, font_size=18, width=2, node_size=1000)
plt.show()

print(ground_truth)
print(adj)