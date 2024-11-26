from tools import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def generate_graph(n, k, d):
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for _ in range(k):
        # Split the vertices into three equal sets V1, V2, V3
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        split_size = n // 3
        V1 = nodes[:split_size]
        V2 = nodes[split_size:2 * split_size]
        V3 = nodes[2 * split_size:]

        # Define probabilities
        p_within_V2 = d / (k * (n - 1))
        p_within_V1 = d / (10 * k * (n - 1))
        p_within_V3 = d / (10 * k * (n - 1))
        p_between_V2_V1 = d / (k * (n - 1))
        p_between_V2_V3 = d / (k * (n - 1))
        p_between_V1_V3 = (19 * d) / (10 * k * (n - 1))

        # Add edges within V2
        for u in V2:
            for v in V2:
                if u < v and np.random.rand() < p_within_V2:
                    G.add_edge(u, v)

        # Add edges within V1
        for u in V1:
            for v in V1:
                if u < v and np.random.rand() < p_within_V1:
                    G.add_edge(u, v)

        # Add edges within V3
        for u in V3:
            for v in V3:
                if u < v and np.random.rand() < p_within_V3:
                    G.add_edge(u, v)

        # Add edges between V2 and V1
        for u in V2:
            for v in V1:
                if np.random.rand() < p_between_V2_V1:
                    G.add_edge(u, v)

        # Add edges between V2 and V3
        for u in V2:
            for v in V3:
                if np.random.rand() < p_between_V2_V3:
                    G.add_edge(u, v)

        # Add edges between V1 and V3
        for u in V1:
            for v in V3:
                if np.random.rand() < p_between_V1_V3:
                    G.add_edge(u, v)

    return G


def plot_degree_distribution(graph):
    degrees = [degree for node, degree in graph.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    n = 2997  # Total number of vertices (should be divisible by 3)
    k = 3
    d = 500

    G = generate_graph(n, k, d)
    print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    plot_degree_distribution(G)
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    analyse_spectrum(adjacency_matrix)
