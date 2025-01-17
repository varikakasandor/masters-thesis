import numpy as np
import random
from tools import analyse_spectrum, plot_degree_distribution_from_adjacency_matrix


def generate_d_regular_graph(n, d, alpha):
    # Sizes of the parts
    size_A = n // 3
    size_B = n // 3
    size_C = n // 6
    size_D = n // 6

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n))

    # Indices for parts
    A_indices = range(0, size_A)
    B_indices = range(size_A, size_A + size_B)
    C_indices = range(size_A + size_B, size_A + size_B + size_C)
    D_indices = range(size_A + size_B + size_C, n)

    # Edge probabilities
    prob_A_B = (d / 2) / size_A
    prob_A_C = (alpha * d) / size_A
    prob_A_D = ((1 - alpha) * d) / size_A
    prob_B_C = ((1 - alpha) * d) / size_B
    prob_B_D = (alpha * d) / size_B
    prob_C_D = 0

    # Add edges between parts with given probabilities
    for i in A_indices:
        for j in B_indices:
            if random.random() < prob_A_B:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in A_indices:
        for j in C_indices:
            if random.random() < prob_A_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in A_indices:
        for j in D_indices:
            if random.random() < prob_A_D:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in B_indices:
        for j in C_indices:
            if random.random() < prob_B_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in B_indices:
        for j in D_indices:
            if random.random() < prob_B_D:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix


if __name__ == "__main__":
    n = 4000
    d = 1000
    alpha = 0.51  # Example parameter

    adjacency_matrix = generate_d_regular_graph(n, d, alpha)
    analyse_spectrum(adjacency_matrix)
    plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)
