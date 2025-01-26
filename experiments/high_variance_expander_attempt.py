from multiprocessing import Pool

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from tools import analyse_spectrum, plot_degree_distribution_from_adjacency_matrix, analyse_2nd_largest_eigenvalue
import random


def generate_graph(n, c, d):
    assert n % 9 == 0
    if c > 1 / 24:
        print(f"WARNING: c<=1/24 doesn't hold!")
    # Sizes of the parts
    size_A, size_B, size_C = n // 3, n // 3, n // 3
    size_A1, size_A3 = int((2 * c) * (n // 3)), int((2 * c) * (n // 3))
    size_A2 = size_A - size_A1 - size_A3

    # Indices for parts
    A1_indices = range(0, size_A1)
    A2_indices = range(size_A1, size_A1 + size_A2)
    A3_indices = range(size_A1 + size_A2, size_A)
    B_indices = range(size_A, size_A + size_B)
    C_indices = range(size_A + size_B, n)

    adjacency_matrix = np.zeros((n, n))

    # Between A1 and B: random bipartite graph
    prob_A1_B = d / size_B  # Probability of edge existence
    for i in A1_indices:
        for j in B_indices:
            if random.random() < prob_A1_B:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # Since the graph is undirected

    # Between A2 and B: random bipartite graph
    prob_A2_B = d / (2 * size_B)  # Probability of edge existence
    for i in A2_indices:
        for j in B_indices:
            if random.random() < prob_A2_B:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # Since the graph is undirected

    # Between A2 and C: random bipartite graph
    prob_A2_C = d / (2 * size_C)  # Probability of edge existence
    for i in A2_indices:
        for j in C_indices:
            if random.random() < prob_A2_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # Since the graph is undirected

    # Between A3 and C: random bipartite graph
    prob_A3_C = d / size_C  # Probability of edge existence
    for i in A3_indices:
        for j in C_indices:
            if random.random() < prob_A3_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # Since the graph is undirected

    # Between B and C: random bipartite graph
    prob_B_C = d / (2 * size_C)  # Probability of edge existence
    for i in B_indices:
        for j in C_indices:
            if random.random() < prob_B_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1  # Since the graph is undirected

    return adjacency_matrix


def process_c(args):
    n, c, d = args
    adjacency_matrix = generate_graph(n, c, d)
    return analyse_2nd_largest_eigenvalue(adjacency_matrix)


def plot_c_vs_expansion(n, d, num_grid_points):
    cs = np.linspace(0, 1 / 20, num_grid_points)

    # Prepare arguments for parallel processing
    args = [(n, c, d) for c in cs]

    with Pool() as pool:
        second_largest_eigenvalues = list(tqdm(pool.imap(process_c, args), total=len(cs)))

    plt.plot(cs, second_largest_eigenvalues, marker='o')
    plt.title(f"Relationship Between c and Lambda_2 for n={n}, d={d}")
    plt.xlabel("c")
    plt.ylabel("Normalised 2nd Largest Eigenvalue")
    plt.grid(True)

    # Save the plot
    plt.savefig(f"c_vs_expansion_n{n}_d{d}.png")
    plt.show()


if __name__ == "__main__":
    n = 5004
    d = 1500
    num_grid_points = 50
    plot_c_vs_expansion(n, d, num_grid_points)
    # adjacency_matrix = generate_graph(n=4005, c=0, d=1000)
    # analyse_spectrum(adjacency_matrix)
    # plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)
