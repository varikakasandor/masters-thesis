import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

from tools import analyse_spectrum, plot_degree_distribution_from_adjacency_matrix


def generate_bipartite_graph(n, d):
    # Generate a bipartite graph by connecting each vertex in A to d random neighbors in B
    nodes_per_side = n // 2
    A = list(range(nodes_per_side))
    B = list(range(nodes_per_side, n))
    G = nx.Graph()
    G.add_nodes_from(A, bipartite=0)
    G.add_nodes_from(B, bipartite=1)

    for u in A:
        neighbors = np.random.choice(B, size=d, replace=False)
        for v in neighbors:
            G.add_edge(u, v)

    return G, A, B


def modify_bipartite_graph(G, A, B):
    # Randomly select n/6 vertices from each side
    n = len(G.nodes())
    to_remove_A = np.random.choice(A, size=n // 6, replace=False)
    to_remove_B = np.random.choice(B, size=n // 6, replace=False)

    # Remove all edges between selected vertices
    for u in to_remove_A:
        for v in to_remove_B:
            if G.has_edge(u, v):
                G.remove_edge(u, v)

    return G


def plot_spectrum(adjacency_matrix):
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(adjacency_matrix)

    # Plot the spectrum
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title('Spectrum of the Graph')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    n = 3000  # Total number of vertices (must be even)
    d = 800  # Degree for the initial bipartite graph

    # Step 1: Generate the initial bipartite graph
    G, A, B = generate_bipartite_graph(n, d)

    # Step 2: Modify the bipartite graph
    G = modify_bipartite_graph(G, A, B)

    # Step 3: Analyze the spectrum
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    # print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    analyse_spectrum(adjacency_matrix)
    plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)

