import leidenalg
import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt

# Generate an Erdos-Renyi graph with 100 nodes and a 0.1 probability of edge creation
G = ig.Graph.Erdos_Renyi(n=100, p=0.1)

# Apply the Leiden algorithm to find communities
part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition)

# Convert igraph graph to NetworkX graph
G_nx = nx.Graph(part.graph.get_edgelist())

# Draw the NetworkX graph using matplotlib
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G_nx, seed=42)  # For consistent layout
nx.draw(G_nx, pos, with_labels=True, node_size=100, node_color='skyblue', edge_color='gray')

plt.show()