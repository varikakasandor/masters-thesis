import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

from tools import analyse_spectrum, plot_degree_distribution_from_adjacency_matrix, analyse_2nd_largest_eigenvalue


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


def plot_alpha_vs_eigenvalue(n, d, num_grid_points):
    alphas = np.linspace(0.5, 0.75, num_grid_points)
    second_largest_eigenvalues = []

    for alpha in alphas:
        adjacency_matrix = generate_d_regular_graph(n, d, alpha)
        second_largest = analyse_2nd_largest_eigenvalue(adjacency_matrix)
        second_largest_eigenvalues.append(second_largest)

    plt.plot(alphas, second_largest_eigenvalues, marker='o')
    plt.title(f"Relationship Between Alpha and Lambda_2 for n={n}, d={d}")
    plt.xlabel("Alpha")
    plt.ylabel("Normalised 2nd Largest Eigenvalue")
    plt.grid(True)
    plt.show()


def find_elbow_points(n, num_grid_points, num_ds, d_min, d_max):
    ds = np.linspace(d_min, d_max, num_ds, dtype=int)
    elbow_points = []

    for d in tqdm(ds):
        alphas = np.linspace(0.5, 0.75, num_grid_points)
        second_largest_eigenvalues = []

        for alpha in tqdm(alphas):
            adjacency_matrix = generate_d_regular_graph(n, d, alpha)
            second_largest = analyse_2nd_largest_eigenvalue(adjacency_matrix)
            second_largest_eigenvalues.append(second_largest)

        # Find the "elbow" where eigenvalue starts increasing significantly
        diffs = np.diff(second_largest_eigenvalues)
        elbow_index = np.argmax(diffs > 0.01)  # Threshold to detect change
        elbow_alpha = alphas[elbow_index] if elbow_index < len(alphas) else None
        elbow_points.append((d, elbow_alpha))

    # Plot elbow points
    ds, elbow_alphas = zip(*[(d, alpha) for d, alpha in elbow_points if alpha is not None])
    plt.plot(ds, elbow_alphas, marker='o')
    plt.title("Elbow Points for Different d Values")
    plt.xlabel("d")
    plt.ylabel("Alpha at Elbow Point")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    n = 1000
    NUM_GRID_POINTS = 40
    NUM_DS = 30
    D_MIN = 200
    D_MAX = 500

    find_elbow_points(n, NUM_GRID_POINTS, NUM_DS, D_MIN, D_MAX)

    """d = 500
    NUM_GRID_POINTS = 30
    plot_alpha_vs_eigenvalue(n, d, NUM_GRID_POINTS)"""

    """alpha = 0.51  # Example parameter
    adjacency_matrix = generate_d_regular_graph(n, d, alpha)
    analyse_spectrum(adjacency_matrix)
    plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)"""
