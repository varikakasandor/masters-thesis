import numpy as np
import networkx as nx
from tools import analyse_spectrum, plot_degree_distribution
import random
import scipy.linalg
import matplotlib.pyplot as plt


def generate_graph_with_independent_set(n, d, c):
    # Generate a random d-regular graph of size 'n'
    G = nx.random_regular_graph(d, n)

    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)
    print("Spectrum before planting:")
    analyse_spectrum(adjacency_matrix)

    # Calculate the size of the independent set
    independent_set_size = int(c * n)

    # Remove all edges within the independent set
    for i in range(independent_set_size):
        for j in range(i + 1, independent_set_size):
            adjacency_matrix[i][j] = 0
            adjacency_matrix[j][i] = 0

    return adjacency_matrix


if __name__ == "__main__":
    adjacency_matrix = generate_graph_with_independent_set(n=1000, d=500, c=0.48)
    print("Spectrum after planting:")
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
    analyse_spectrum(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)

    # Plot the eigenvector corresponding to the most negative eigenvalue
    most_negative_eigenvalue_index = np.argmin(eigenvalues)
    most_negative_eigenvector = eigenvectors[:, most_negative_eigenvalue_index]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(most_negative_eigenvector)), most_negative_eigenvector, marker='o', linestyle='-', markersize=2)
    plt.xlabel("Index")
    plt.ylabel("Eigenvector Value")
    plt.title("Eigenvector Corresponding to the Most Negative Eigenvalue")
    plt.show()
