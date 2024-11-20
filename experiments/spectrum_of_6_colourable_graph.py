import numpy as np
import networkx as nx
from tools import analyse_spectrum, plot_degree_distribution
import random
import matplotlib.pyplot as plt

def generate_tripartite_graph(n, p1, p2):
    # Check if n is divisible by 3
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3 for this construction.")

    # Shuffle and split vertices into 3 equal random parts
    nodes = list(range(n))
    random.shuffle(nodes)
    part_size = n // 3
    part_A = nodes[:part_size]
    part_B = nodes[part_size:2 * part_size]
    part_C = nodes[2 * part_size:]

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n))

    # Add edges between the parts A, B, C with probability p1
    for i in part_A:
        for j in part_B:
            if random.random() < p1:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1
        for j in part_C:
            if random.random() < p1:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in part_B:
        for j in part_C:
            if random.random() < p1:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    # Add edges between sets {1, 2, ..., n/2} and {n/2+1, ..., n} with probability p2
    half_n = n // 2
    for i in range(0, half_n):
        for j in range(half_n, n):
            if random.random() < p2:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix, part_A, part_B, part_C

def calculate_degree_variance_table(adjacency_matrix, part_A, part_B, part_C, n):
    half_n = n // 2
    set_A1 = [node for node in part_A if node < half_n]
    set_A2 = [node for node in part_A if node >= half_n]
    set_B1 = [node for node in part_B if node < half_n]
    set_B2 = [node for node in part_B if node >= half_n]
    set_C1 = [node for node in part_C if node < half_n]
    set_C2 = [node for node in part_C if node >= half_n]

    sets = {
        "A1": set_A1,
        "A2": set_A2,
        "B1": set_B1,
        "B2": set_B2,
        "C1": set_C1,
        "C2": set_C2
    }

    variance_table = np.zeros((6, 6))
    set_names = list(sets.keys())

    for i, set1_name in enumerate(set_names):
        set1 = sets[set1_name]
        for j, set2_name in enumerate(set_names):
            set2 = sets[set2_name]
            degrees = []
            for node in set1:
                degree = sum(adjacency_matrix[node][neighbor] for neighbor in set2)
                degrees.append(degree)
            variance_table[i][j] = np.var(degrees)

    return variance_table

if __name__ == "__main__":
    n = 2997  # Total number of nodes, must be divisible by 3
    p1 = 1 / 50  # Probability for edges between parts A, B, C
    p2 = 1 / 5  # Probability for edges between {1, 2, ..., n/2} and {n/2+1, ..., n}

    # Generate the graph
    adjacency_matrix, part_A, part_B, part_C = generate_tripartite_graph(n, p1, p2)

    # Analyse the spectrum before planting
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

    # Calculate and print the variance table for degrees into different parts
    variance_table = calculate_degree_variance_table(adjacency_matrix, part_A, part_B, part_C, n)
    print("Variance of degree of vertices between different parts:")
    print(variance_table)
