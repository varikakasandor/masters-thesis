import numpy as np
import networkx as nx
import random
from tools import plot_degree_distribution_from_adjacency_matrix


def generate_d_regular_graph(n, d):
    """Generates a d-regular random graph."""
    G = nx.random_regular_graph(d, n)
    return nx.to_numpy_array(G)


def plant_independent_set(adjacency_matrix, independent_set):
    """Removes edges within the specified independent set."""
    part_indices = np.ix_(independent_set, independent_set)
    adjacency_matrix[part_indices] = 0
    return adjacency_matrix


def print_selected_eigenvalues(adjacency_matrix):
    """Prints specific eigenvalues of the adjacency matrix."""
    eigenvalues = np.linalg.eigh(adjacency_matrix)[0]
    print(f"Largest eigenvalue: {eigenvalues[-1]:.4f}")
    print(f"Second largest eigenvalue: {eigenvalues[-2]:.4f}")
    print(f"Smallest eigenvalue: {eigenvalues[0]:.4f}")


def random_independent_set(n, size):
    """Generates a random independent set of the specified size."""
    return random.sample(range(n), size)


if __name__ == "__main__":
    n = 4000  # Number of nodes
    d = 700  # Degree of the d-regular graph
    iter = 30  # Number of iterations

    # Generate initial d-regular graph
    adjacency_matrix = generate_d_regular_graph(n, d)

    print("Initial eigenvalues:")
    print_selected_eigenvalues(adjacency_matrix)

    # Sequentially plant independent sets and report eigenvalues
    for i in range(iter):
        print(f"\nIteration {i + 1}:")

        # Plant random independent set
        independent_set = random_independent_set(n, n // 3)
        adjacency_matrix = plant_independent_set(adjacency_matrix, independent_set)

        # Print eigenvalues
        print_selected_eigenvalues(adjacency_matrix)

    # Optionally plot the degree distribution at the end
    plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)
