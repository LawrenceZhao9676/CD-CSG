# Causal Discovery via Causal Star Graphs
## data available:
Real-world Data: The real-world datasets includes abalone data, auto MPG data, cardiac arrhythmia data, concrete compressive strength data, Deutscher Wetterdienst data and ozone data. All data and ground truth are in folder .\real-world data.    
Synthetic Data: The generation of synthetic data are based on cdt.data.AcyclicGraphGenerator and we overwrite it with only additive noise.  


## code available:
CD-CSG.py: Causal discovery via causal star graphs （CD-CSG） is causal discovery framework to learn causal directed acyclic graphs (DAGs). It bases on the generalized learning and identify the causal directions through finding the asymmetry in the forward and backward model of CD-CSG.  
GenLearning.py: Generalized learning and the forward and backward model in CD-CSG.   
main.py: An example of running CD-CSG.

### dependencies 
    - Keras                    2.4.3 
    - tensorflow               2.2.0 
    - numpy                    1.18.5 
    - networkx                 2.4 
    - cdt                      0.5.21
The code is tested with python 3.7 on Windows 10. 
GPU: NVIDIA GeForce RTX 2080Ti 

**Data Type**: Continuous

## Example:
### main.py  
```python
        >>> import numpy as np
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from cdt.data import AcyclicGraphGenerator
        >>> from CD_CSG import CD_CSG
        >>> import random
        
        >>> generator = AcyclicGraphGenerator('polynomial', noise='gaussian', noise_coeff=0.4, npoints=500, nodes=8)
        >>> data, graph1 = generator.generate()
        >>> graph = graph1.to_undirected() # get the causal skeleton of graph1
    
        >>> obj = CD_CSG()
```
```python
        # This example uses the predict_graph() method
        >>> DAG, adj = obj.predict_graph(data, graph)
```
```python        
        # This example uses the predict_adj() method
        >>> A = np.array(nx.adj_matrix(graph).todense()) # get the adjacent matrix of the graph
        >>> DAG, adj = obj.predict_adj(data, A)
```
```python        
        # To view the result
        >>> plt.figure(figsize=(8, 4))
        >>> plt.subplot(121)
        >>> plt.title('ground truth')
        >>> nx.draw(graph1, pos=nx.circular_layout(graph1), node_color='g', edge_color='r', with_labels=True, font_size=18, width=2, node_size=1000)
        >>> plt.subplot(122)
        >>> plt.title('CSG result')
        >>> nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True, font_size=18, width=2, node_size=1000)
        >>> plt.show()
```
---

#### predict_graph(data, graph):   
    Predict from undirected graph of a causal skeleton using CSG.  
    Args:  
        data (np.ndarray): A np.array of variables in the causal skeleton.  
        graph (np.ndarray): The undirected graph of the causal skeleton.  
    Returns:  
        output(nx.DiGraph): The predicted causal DAG.  
        adj(np.ndarray): The adjacent matrix of the causal DAG.  

#### predict_adj(data, adj):    
    Predict from adjacent matrix of a causal skeleton using CSG.  
    Args:  
        data (np.ndarray): A np.array of variables in the causal skeleton.  
        ADJ (np.ndarray): The adjacent matrix of the causal skeleton.  
    Returns:  
        output(nx.DiGraph): The predicted causal DAG.  
        adj(np.ndarray): The adjacent matrix of the causal DAG.  




### License
This project is licensed under the MIT License - see the LICENSE file for details.
