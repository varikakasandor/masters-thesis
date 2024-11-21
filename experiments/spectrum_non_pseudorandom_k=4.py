import numpy as np
import random
import matplotlib.pyplot as plt
from tools import analyse_spectrum, plot_degree_distribution


def generate_quad_partition_graph(n, c, d):
    # Check if n is divisible by 4
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4 for this construction.")

    # Shuffle and split vertices into 4 equal random parts
    nodes = list(range(n))
    # random.shuffle(nodes)
    part_size = n // 4
    V1 = nodes[:part_size]
    V2 = nodes[part_size:2 * part_size]
    V3 = nodes[2 * part_size:3 * part_size]
    V4 = nodes[3 * part_size:]

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n))

    # Define expected degree probabilities
    p_12 = (2 - c) * d / (4 * part_size)  # Expected degree between V_i and V_{i+1}
    p_13 = c * d / (2 * part_size)  # Expected degree between V_i and V_{i+2}

    # Add edges between consecutive parts (V1-V2, V2-V3, V3-V4, V4-V1)
    parts = [(V1, V2), (V2, V3), (V3, V4), (V4, V1)]
    for part_A, part_B in parts:
        for i in part_A:
            for j in part_B:
                if random.random() < p_12:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1

    # Add edges between alternating parts (V1-V3, V2-V4)
    cross_parts = [(V1, V3), (V2, V4)]
    for part_A, part_B in cross_parts:
        for i in part_A:
            for j in part_B:
                if random.random() < p_13:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1

    return adjacency_matrix, V1, V2, V3, V4


if __name__ == "__main__":
    n = 2000  # Total number of nodes, must be divisible by 4
    c = 1 / 50  # Parameter c
    d = 500  # Parameter d

    # Generate the graph
    adjacency_matrix, V1, V2, V3, V4 = generate_quad_partition_graph(n, c, d)

    # Analyse the spectrum after generating the graph
    print("Spectrum after generating the graph:")
    analyse_spectrum(adjacency_matrix)

    # Plot the degree distribution
    plot_degree_distribution(adjacency_matrix)

    # Compute and plot the eigenvector corresponding to the most negative eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
    most_negative_eigenvalue_index = np.argmin(eigenvalues)
    most_negative_eigenvector = eigenvectors[:, most_negative_eigenvalue_index]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(most_negative_eigenvector)), most_negative_eigenvector, marker='o', linestyle='-', markersize=2)
    plt.xlabel("Index")
    plt.ylabel("Eigenvector Value")
    plt.title("Eigenvector Corresponding to the Most Negative Eigenvalue")
    plt.show()
