import numpy as np
import networkx as nx
from tools import analyse_spectrum, plot_degree_distribution
import random


def generate_graph(n, ccs, d):
    # Calculate the size of each connected component
    component_size = n // ccs

    # Initialize an empty graph using NetworkX
    G = nx.Graph()

    # Generate k connected components, each being a random d-regular graph
    for i in range(ccs):
        # Generate a random d-regular graph of size 'component_size'
        component = nx.random_regular_graph(d, component_size)
        # Relabel nodes to ensure unique labels across different components
        mapping = {node: node + i * component_size for node in component.nodes()}
        nx.relabel_nodes(component, mapping, copy=False)
        # Add the component to the main graph
        G = nx.compose(G, component)

    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)
    print("Spectrum before planting:")
    analyse_spectrum(adjacency_matrix)

    # Randomly split the vertices into 3 equal sets V1, V2, V3
    nodes = list(G.nodes())
    random.shuffle(nodes)
    V1 = nodes[:n // 3]
    V2 = nodes[n // 3: 2 * n // 3]
    V3 = nodes[2 * n // 3:]

    # Remove all edges within each set V1, V2, V3
    for V in [V1, V2, V3]:
        for i in range(len(V)):
            for j in range(i + 1, len(V)):
                adjacency_matrix[V[i]][V[j]] = 0
                adjacency_matrix[V[j]][V[i]] = 0

    return adjacency_matrix


if __name__ == "__main__":
    adjacency_matrix = generate_graph(n=1000, ccs=1, d=42)
    print("Spectrum after planting:")
    analyse_spectrum(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)
