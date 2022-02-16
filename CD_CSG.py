from cdt.data import AcyclicGraphGenerator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from GenLearning import CSG_model


class CD_CSG():
    """Causal discovery via causal star graphs.

    **Description**: Causal discovery via causal star graphs （CD-CSG） is causal discovery
    framework to learn causal directed acyclic graphs (DAGs).
    It bases on the generalized learning and identify the causal directions through
    finding the asymmetry in the forward and backward model of CD-CSG.

    **Data Type**: Continuous

    Example:
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from cdt.data import AcyclicGraphGenerator
        >>> from GenLearning import CSG_model
        >>> generator = AcyclicGraphGenerator('polynomial', noise='gaussian', noise_coeff=0.4, npoints=500, nodes=8)
        >>> data, graph1 = generator.generate()
        >>> graph = graph1.to_undirected() # get the causal skeleton of graph1

        >>> obj = CD_CSG()
        >>> # This example uses the predict_graph() method
        >>> DAG, adj = obj.predict_graph(data, graph)

        >>> # This example uses the predict_adj() method
        >>> A = np.array(nx.adj_matrix(graph).todense()) # get the adjacent matrix of the graph
        >>> DAG, adj = obj.predict_adj(data, A)

        >>> # To view the result
        >>> plt.figure(figsize=(8, 4))
        >>> plt.subplot(121)
        >>> plt.title('ground truth')
        >>> nx.draw(graph1, pos=nx.circular_layout(graph1), node_color='g', edge_color='r', with_labels=True, font_size=18, width=2, node_size=1000)
        >>> plt.subplot(122)
        >>> plt.title('CSG result')
        >>> nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True, font_size=18, width=2, node_size=1000)
        >>> plt.show()
    """

    def predict_adj(self, data, ADJ):
        """Predict from an adjacent matrix.

        Args:
            data (np.ndarray): A np.array of variables in the causal skeleton.
            ADJ (np.ndarray): The adjacent matrix of the causal skeleton.
        Returns:
            output(nx.DiGraph): The predicted causal DAG.
            adj(np.ndarray): The adjacent matrix of the causal DAG.
        """
        graph = nx.from_numpy_matrix(ADJ)
        data = np.array(data)
        adj = self.get_causalstargraph(data, graph)
        output = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
        return output, adj

    def predict_graph(self, data, graph):
        """Predict from an undirected graph.

        Args:
            data (np.ndarray): A np.array of variables in the causal skeleton.
            graph (np.ndarray): The undirected graph of the causal skeleton.
        Returns:
            output(nx.DiGraph): The predicted causal DAG.
            adj(np.ndarray): The adjacent matrix of the causal DAG.
        """
        data = np.array(data)
        adj = self.get_causalstargraph(data, graph)
        output = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
        return output, adj

    def get_stargraph(self, graph, n):
        g = graph.to_undirected()
        A = np.array(nx.adj_matrix(g).todense())
        adj = np.array(np.where(A[n, :] == 1)).reshape(-1, )
        return adj

    def get_causalstargraph(self, data, graph):
        D = data.shape[1]
        adj = np.zeros([D, D])
        for i in range(D):
            adj_i = self.get_stargraph(graph, i)
            csg = np.c_[data[:, adj_i], data[:, i]]
            causal_dir = CSG_model(csg)
            adj[adj_i, i] = causal_dir
            adj[i, adj_i] = -causal_dir + 1
        return adj
