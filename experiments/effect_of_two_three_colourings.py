import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from tools import analyse_spectrum, plot_degree_distribution


def generate_initial_regular_graph(n, d):
    # Generate a random (3d/2)-regular graph
    degree = (3 * d) // 2
    if degree >= n:
        raise ValueError("Degree must be less than the number of nodes for a valid regular graph.")
    G = nx.random_regular_graph(degree, n)
    adjacency_matrix = nx.to_numpy_array(G)
    return adjacency_matrix


def split_and_remove_edges(adjacency_matrix, n):
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
    n = 3000  # Total number of nodes, must be divisible by 3
    d = 1000  # Parameter d

    # Generate the initial (3d/2)-regular graph
    adjacency_matrix = generate_initial_regular_graph(n, d)

    # Phase 0: Initial graph
    print("Eigenvalues after phase 0:")
    print_eigenvalues(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)

    # Phase 1: Split into 3 parts and remove intra-part edges
    adjacency_matrix = split_and_remove_edges(adjacency_matrix, n)
    print("\nEigenvalues after phase 1:")
    print_eigenvalues(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)

    # Phase 2: Split into 3 parts again and remove intra-part edges
    adjacency_matrix = split_and_remove_edges(adjacency_matrix, n)
    print("\nEigenvalues after phase 2:")
    print_eigenvalues(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)
