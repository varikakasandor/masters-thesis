from itertools import product
import networkx as nx

from tools import *


def create_partitioned_graph(n, r, d, bipartite_creator):
    # Assertions for valid inputs
    assert n % r == 0, "n must be divisible by r (equal partition sizes)."
    assert d % (r - 1) == 0, "d must be divisible by (r-1) to ensure d/(r-1)-regularity."

    partition_size = n // r  # The size of each partition

    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Partition the vertices into r sets
    partitions = [list(range(i * partition_size, (i + 1) * partition_size)) for i in range(r)]

    # Connect the partitions using the provided bipartite_creator function
    for i in range(r):
        for j in range(i + 1, r):
            # Get the vertices in partitions i and j
            part_i = partitions[i]
            part_j = partitions[j]

            # Create a bipartite graph between part_i and part_j using bipartite_creator
            bipartite_adj = bipartite_creator(partition_size, d // (r - 1))

            # Map the bipartite adjacency matrix to the full adjacency matrix
            for u in range(partition_size):
                for v in range(partition_size):
                    if bipartite_adj[u, v] == 1:
                        adjacency_matrix[part_i[u], part_j[v]] = 1
                        adjacency_matrix[part_j[v], part_i[u]] = 1

    return adjacency_matrix


# Example bipartite creator function that generates a random d-regular bipartite graph
def random_bipartite_d_regular(partition_size, d):
    # Create an empty adjacency matrix for the bipartite graph
    bipartite_adj = np.zeros((partition_size, partition_size), dtype=int)

    # Ensure each vertex in both partitions has exactly d neighbors
    # This is a simple random matching mechanism
    for i in range(partition_size):
        neighbors = np.random.choice(partition_size, size=d, replace=False)
        bipartite_adj[i, neighbors] = 1

    return bipartite_adj


def random_d_regular_graph(n, d):
    """
    Generate a random d-regular graph with n vertices.
    """
    assert n * d % 2 == 0, "n*d must be even for a d-regular graph."

    # Use NetworkX to generate a random d-regular graph
    G = nx.random_regular_graph(d, n)

    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)

    return adjacency_matrix


def plant_3_colouring(adjacency_matrix):
    """
    Plant a balanced 3-colouring by partitioning the vertices into 3 clusters
    and removing all intra-cluster edges.
    """
    n = adjacency_matrix.shape[0]
    # Randomly assign vertices to one of the 3 clusters
    clusters = np.random.permutation(n) % 3  # Values will be 0, 1, or 2

    # Remove intra-cluster edges by setting entries to 0 within each cluster
    for i in range(n):
        for j in range(i + 1, n):  # Only need to check upper triangle for symmetry
            if clusters[i] == clusters[j]:
                adjacency_matrix[i, j] = 0
                adjacency_matrix[j, i] = 0

    return adjacency_matrix


def create_3_coloured_d_regular_graph(n, d):
    """
    Create a random d-regular graph with n vertices and then plant a balanced
    3-colouring by removing intra-cluster edges.
    """
    # Step 1: Create a random d-regular graph
    adjacency_matrix = random_d_regular_graph(n, d)

    # Step 2: Plant a 3-colouring by removing intra-cluster edges
    adjacency_matrix = plant_3_colouring(adjacency_matrix)

    return adjacency_matrix


def generate_arora_chlamtac_graph(d):
    """
    Generate the graph described by Arora and Chlamtac where:
    - Vertices are labeled by vectors in {1, 2, 3}^d.
    - Two vertices x and y are adjacent if and only if they differ in all coordinates.
    """
    # Step 1: Generate all vertices as vectors from {1, 2, 3}^d
    vertices = list(product([1, 2, 3], repeat=d))
    n = len(vertices)  # Total number of vertices, which is 3^d

    # Step 2: Initialize an adjacency matrix of size n x n
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Step 3: Add edges between vertices that differ in all coordinates
    for i in range(n):
        for j in range(i + 1, n):
            if all(vertices[i][k] != vertices[j][k] for k in range(d)):
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix


if __name__ == "__main__":
    # analyse_spectrum(create_partitioned_graph(1000, 4, 99, random_bipartite_d_regular))
    # analyse_spectrum(create_3_coloured_d_regular_graph(1000, 100))
    arora_graph = generate_arora_chlamtac_graph(8)
    analyse_spectrum(arora_graph)
    analyse_spectrum(plant_3_colouring(arora_graph))
