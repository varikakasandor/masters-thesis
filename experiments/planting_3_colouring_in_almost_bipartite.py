import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tools import *


def generate_bipartite_graph(n, d, d_prime):
    # Split nodes into two equal parts (sequential split)
    V1 = list(range(n // 2))
    V2 = list(range(n // 2, n))

    # Generate d'-regular graph within each part
    G1 = nx.random_regular_graph(d_prime, len(V1))
    G2 = nx.random_regular_graph(d_prime, len(V2))

    # Create adjacency matrix for the complete graph
    adjacency_matrix = np.zeros((n, n))
    adjacency_matrix[:n // 2, :n // 2] = nx.to_numpy_array(G1)
    adjacency_matrix[n // 2:, n // 2:] = nx.to_numpy_array(G2)

    # Generate bipartite edges between V1 and V2 using a random graph
    p = (d - d_prime) / len(V2)  # Probability for bipartite edges
    B = nx.algorithms.bipartite.generators.random_graph(len(V1), len(V2), p)

    # Add the bipartite edges to the adjacency matrix
    for u, v in B.edges:
        adjacency_matrix[u, v] = 1
        adjacency_matrix[v, u] = 1

    return adjacency_matrix, V1, V2


def split_and_remove_edges_for_three_coloring(adjacency_matrix, n):
    # Split vertices into 3 equal random parts
    nodes = list(range(n))
    random.shuffle(nodes)
    part_size = n // 3
    V1 = nodes[:part_size]
    V2 = nodes[part_size:2 * part_size]
    V3 = nodes[2 * part_size:]

    # Remove edges within each part using NumPy indexing for speedup
    for part in [V1, V2, V3]:
        part_indices = np.ix_(part, part)
        adjacency_matrix[part_indices] = 0

    return adjacency_matrix


def print_eigenvalues(adjacency_matrix):
    eigenvalues = np.linalg.eigh(adjacency_matrix)[0]
    print(f"First eigenvalue: {eigenvalues[-1]:.4f}")  # Largest eigenvalue
    print(f"Second eigenvalue: {eigenvalues[-2]:.4f}")
    print(f"Fifth smallest eigenvalue: {eigenvalues[4]:.4f}")
    print(f"Fourth smallest eigenvalue: {eigenvalues[3]:.4f}")
    print(f"Third smallest eigenvalue: {eigenvalues[2]:.4f}")
    print(f"Penultimate eigenvalue: {eigenvalues[1]:.4f}")
    print(f"Last eigenvalue: {eigenvalues[0]:.4f}")  # Smallest eigenvalue


if __name__ == "__main__":
    n = 3000  # Total number of nodes, must be divisible by 2
    d = 400  # Parameter d (degree of regularity between parts)
    d_prime = 5  # Parameter d' (degree of regularity within each part)

    # Generate the bipartite graph with d' regular graphs within each part and (d - d') regular bipartite edges
    adjacency_matrix, V1, V2 = generate_bipartite_graph(n, d, d_prime)

    # Phase 0: Initial graph
    print("Eigenvalues after phase 0:")
    print_eigenvalues(adjacency_matrix)
    plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)

    # Phase 1: Split into 3 parts and remove intra-part edges for three-color planting
    adjacency_matrix = split_and_remove_edges_for_three_coloring(adjacency_matrix, n)
    print("\nEigenvalues after phase 1:")
    print_eigenvalues(adjacency_matrix)
    plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)

    # Further analysis or visualization can be added here if necessary.