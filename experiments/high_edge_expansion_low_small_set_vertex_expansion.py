import numpy as np
from tools import analyse_spectrum, plot_degree_distribution
import random

def generate_graph(n, c, d):
    # Sizes of the parts
    size_A = int(c * n)
    size_B = int(3 * c * n)
    size_C = int((1 - 4 * c) * n)
    total_size = size_A + size_B + size_C

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((total_size, total_size))

    # Indices for parts
    A_indices = range(0, size_A)
    B_indices = range(size_A, size_A + size_B)
    C_indices = range(size_A + size_B, total_size)

    # Between A and B: random bipartite graph
    prob_A_B = d / (3 * c * n)  # Probability of edge existence
    for i in A_indices:
        for j in B_indices:
            if random.random() < prob_A_B:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # Since the graph is undirected

    # Between B and C: random bipartite graph
    prob_B_C = 2 * d / (3 * (1 - 4 * c) * n)
    for i in B_indices:
        for j in C_indices:
            if random.random() < prob_B_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    # Within C: random graph
    prob_C_C = (1 - 6 * c) * d / (((1 - 4 * c) ** 2) * n)
    for i in C_indices:
        for j in C_indices:
            if i < j and random.random() < prob_C_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix

if __name__ == "__main__":
    adjacency_matrix = generate_graph(n=3000, c=34/3000, d=100)
    analyse_spectrum(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)
