import numpy as np
import networkx as nx

from experiments.tools import plot_degree_distribution


def generate_bipartite_graph(a_size, b_size, degree_a):
    """Generates a random bipartite graph with specified degree on side A."""
    print("Generating bipartite graph...")
    G = nx.bipartite.random_graph(a_size, b_size, degree_a / b_size)
    # Extract adjacency matrix in (a_size + b_size) x (a_size + b_size) format
    adjacency_matrix = nx.to_numpy_array(G)
    # Extract the top-left and bottom-right blocks for the bipartite graph
    return adjacency_matrix[:a_size, a_size:]  # Only keep the A-B connections


def generate_partitioned_graph(n, d, x, c):
    """Generates the desired d-regular graph with partitions A and B."""
    print("Calculating sizes of partitions A and B...")
    # Sizes of A and B
    size_a = int(x * n)
    size_b = n - size_a

    print(f"Partition A size: {size_a}, Partition B size: {size_b}")

    # Split A into A1 and A2
    size_a1 = size_a // 2
    size_a2 = size_a - size_a1

    print(f"Partition A1 size: {size_a1}, Partition A2 size: {size_a2}")

    # Degrees within A1-A2 and between A and B
    degree_a1_a2 = int(c * d)
    degree_bipartite_a = int((1 - c) * d)
    degree_b = int((1 - x * (1 - c) / (1 - x)) * d)

    print(f"Degrees: A1-A2={degree_a1_a2}, Bipartite={degree_bipartite_a}, B={degree_b}")

    # Generate bipartite graph between A1 and A2 (c*d degree)
    print("Generating bipartite graph between A1 and A2...")
    adjacency_a1_a2 = generate_bipartite_graph(size_a1, size_a2, degree_a1_a2)

    # Generate the subgraph within B ((1-x(1-c)/(1-x))*d-regular graph)
    print("Generating subgraph within B...")
    graph_b = nx.random_regular_graph(degree_b, size_b)
    adjacency_b = nx.to_numpy_array(graph_b)

    # Generate the bipartite graph between A and B
    print("Generating bipartite connections between A and B...")
    adjacency_bipartite = generate_bipartite_graph(size_a, size_b, degree_bipartite_a)

    # Combine into full adjacency matrix
    print("Combining subgraphs into full adjacency matrix...")
    adjacency_matrix = np.zeros((n, n))
    adjacency_matrix[:size_a1, size_a1:size_a] = adjacency_a1_a2  # Connect A1 to A2
    adjacency_matrix[size_a1:size_a, :size_a1] = adjacency_a1_a2.T  # Connect A2 to A1
    adjacency_matrix[size_a:, size_a:] = adjacency_b  # Add B subgraph
    adjacency_matrix[:size_a, size_a:] = adjacency_bipartite  # Connect A to B
    adjacency_matrix[size_a:, :size_a] = adjacency_bipartite.T  # Connect B to A

    print("Graph generation complete.")
    return adjacency_matrix


if __name__ == "__main__":
    n = 2000  # Total number of nodes
    d = 30  # Overall degree
    x = 60 / 2000  # Proportion of vertices in A
    c = 0.9  # Fraction of edges within A
    assert x * n / 2 >= d
    # Generate the graph
    print("Starting graph generation...")
    adjacency_matrix = generate_partitioned_graph(n, d, x, c)

    # Check eigenvalues
    print("Calculating eigenvalues...")
    eigenvalues = np.linalg.eigh(adjacency_matrix)[0]
    print(f"Largest eigenvalue: {eigenvalues[-1]:.4f}")
    print(f"Second largest eigenvalue: {eigenvalues[-2]:.4f}")
    print(f"Smallest eigenvalue: {eigenvalues[0]:.4f}")

    G = nx.from_numpy_array(adjacency_matrix)
    plot_degree_distribution(G)
    graph_connected_message = "The graph is connected" if nx.is_connected(G) else "The graph is NOT connected"
    print(graph_connected_message)
