import numpy as np
from tools import analyse_spectrum, plot_degree_distribution
import random


def generate_graph(n, c, d):
    # Sizes of the parts
    size_A1 = int(c * n)
    size_B1 = int(c * n)
    size_A2 = int((1 / 2 - c) * n)
    size_B2 = size_A2
    total_size = size_A1 + size_B1 + size_A2 + size_B2

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((total_size, total_size))

    # Indices for parts
    A1_indices = range(0, size_A1)
    B1_indices = range(size_A1, size_A1 + size_B1)
    A2_indices = range(size_A1 + size_B1, size_A1 + size_B1 + size_A2)
    B2_indices = range(size_A1 + size_B1 + size_A2, total_size)

    # Between A1 and B2: random bipartite graph
    prob_A1_B2 = d / size_B2  # Probability of edge existence
    for i in A1_indices:
        for j in B2_indices:
            if random.random() < prob_A1_B2:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # Since the graph is undirected

    # Between B1 and A2: random bipartite graph (analogous to A1-B2)
    prob_B1_A2 = d / size_A2
    for i in B1_indices:
        for j in A2_indices:
            if random.random() < prob_B1_A2:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    # Between A2 and B2: random bipartite graph
    expected_degree_A2_B2 = d * (1 / 2 - 2 * c) / (1 / 2 - c)
    prob_A2_B2 = expected_degree_A2_B2 / size_B2
    for i in A2_indices:
        for j in B2_indices:
            if random.random() < prob_A2_B2:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix


if __name__ == "__main__":
    adjacency_matrix = generate_graph(n=3000, c=1/25, d=1000)
    analyse_spectrum(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)
