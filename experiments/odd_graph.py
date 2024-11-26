import itertools
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from tools import analyse_spectrum, plot_degree_distribution


def generate_odd_graph(k):
    # Generate nodes as all k-element subsets of a (2k-1)-element set
    base_set = list(range(2 * k + 1))
    nodes = list(itertools.combinations(base_set, k))

    # Create a graph where nodes are k-element subsets and edges exist if subsets are disjoint
    G = nx.Graph()
    G.add_nodes_from(nodes)

    for u, v in itertools.combinations(nodes, 2):
        if set(u).isdisjoint(set(v)):
            G.add_edge(u, v)

    return nx.to_numpy_array(G)


def add_random_regular_edges(adjacency_matrix, d_prime):
    # Convert adjacency matrix to graph
    G = nx.from_numpy_array(adjacency_matrix)
    n = len(adjacency_matrix)

    # Generate a d'-regular random graph and add its edges to G, avoiding duplicate edges
    random_regular_graph = nx.random_regular_graph(d_prime, n)
    for u, v in random_regular_graph.edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    return nx.to_numpy_array(G)


def print_eigenvalues(adjacency_matrix):
    eigenvalues = np.linalg.eigh(adjacency_matrix)[0]
    print(f"First eigenvalue: {eigenvalues[-1]:.4f}")  # Largest eigenvalue
    print(f"Second eigenvalue: {eigenvalues[-2]:.4f}")
    if len(eigenvalues) >= 5:
        print(f"Fifth smallest eigenvalue: {eigenvalues[4]:.4f}")
    if len(eigenvalues) >= 4:
        print(f"Fourth smallest eigenvalue: {eigenvalues[3]:.4f}")
    if len(eigenvalues) >= 3:
        print(f"Third smallest eigenvalue: {eigenvalues[2]:.4f}")
    if len(eigenvalues) >= 2:
        print(f"Penultimate eigenvalue: {eigenvalues[1]:.4f}")
    print(f"Last eigenvalue: {eigenvalues[0]:.4f}")  # Smallest eigenvalue


def plot_smallest_eigenvector(adjacency_matrix):
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)
    smallest_eigenvector = eigenvectors[:, 0]  # Corresponding to the smallest eigenvalue

    # Plot the eigenvector
    plt.figure()
    plt.plot(smallest_eigenvector, marker='o')
    plt.title("Eigenvector Corresponding to the Smallest Eigenvalue")
    plt.xlabel("Node Index")
    plt.ylabel("Eigenvector Component Value")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    k = 5  # Parameter k defining the Odd Graph O_k
    d_prime = 3  # Degree of the random regular graph to add

    # Generate the Odd Graph O_k
    adjacency_matrix = generate_odd_graph(k)

    # Phase 0: Initial graph
    print("Eigenvalues after generating Odd Graph:")
    print_eigenvalues(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)
    plot_smallest_eigenvector(adjacency_matrix)

    # Phase 1: Add edges of a random d'-regular graph
    adjacency_matrix = add_random_regular_edges(adjacency_matrix, d_prime)
    print("\nEigenvalues after adding random d'-regular graph edges:")
    print_eigenvalues(adjacency_matrix)
    plot_degree_distribution(adjacency_matrix)

    # Further analysis or visualization can be added here if necessary.
