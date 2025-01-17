import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt


def generate_partitioned_graph(n, d):
    """
    Generates a graph with vertices split into V_{ia} and V_{ib}.

    - V_{ia}: Degree d/2 to V_{i+1}, d/6 to V_{i-1}.
    - V_{ib}: Degree d/2 to V_{i-1}, d/6 to V_{i+1}.
    - No edges within a V_i.
    """
    print("Splitting vertices into V_{ia} and V_{ib}...")
    vertices = list(range(n))
    random.shuffle(vertices)
    size = n // 6

    # Partition vertices
    V1a, V1b = vertices[:size], vertices[size:2 * size]
    V2a, V2b = vertices[2 * size:3 * size], vertices[3 * size:4 * size]
    V3a, V3b = vertices[4 * size:5 * size], vertices[5 * size:]

    partitions = {
        "V1": V1a + V1b,
        "V2": V2a + V2b,
        "V3": V3a + V3b
    }
    adjacency_matrix = np.zeros((n, n))

    print("Generating bipartite graphs with combined stubs...")

    def connect_combined_stubs(V_from_groups, V_to_groups, degree_from, degree_to):
        """Randomly pair stubs between combined groups ensuring balanced bipartite matching."""
        stubs_from = []
        stubs_to = []

        for V_from, deg in zip(V_from_groups, degree_from):
            stubs_from += list(np.repeat(V_from, deg))
        for V_to, deg in zip(V_to_groups, degree_to):
            stubs_to += list(np.repeat(V_to, deg))

        assert len(stubs_from) == len(stubs_to), "Stub mismatch! Check degree distributions."
        random.shuffle(stubs_from)
        random.shuffle(stubs_to)

        for u, v in zip(stubs_from, stubs_to):
            adjacency_matrix[u, v] = 1
            adjacency_matrix[v, u] = 1  # Symmetry

    # V1 <-> V2 connections
    connect_combined_stubs([V1a, V1b], [V2a, V2b], [d // 2, d // 6], [d // 6, d // 2])
    # V2 <-> V3 connections
    connect_combined_stubs([V2a, V2b], [V3a, V3b], [d // 2, d // 6], [d // 6, d // 2])
    # V3 <-> V1 connections
    connect_combined_stubs([V3a, V3b], [V1a, V1b], [d // 2, d // 6], [d // 6, d // 2])

    print("Graph generation complete.")
    return adjacency_matrix, partitions


def plot_degree_distribution(adjacency_matrix, partitions):
    """Plots degree distributions for the three main classes V1, V2, and V3 -> Vj (j != i)."""
    print("Plotting degree distributions...")
    keys = list(partitions.keys())
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i != j:
                Vi = partitions[keys[i]]
                Vj = partitions[keys[j]]
                degrees = [sum(adjacency_matrix[u, v] for v in Vj) for u in Vi]
                plt.figure()
                plt.hist(degrees, bins=20, edgecolor='black')
                plt.title(f"Degree Distribution ({keys[i]} -> {keys[j]})")
                plt.xlabel("Degree")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()


if __name__ == "__main__":
    n = 3000  # Total vertices
    d = 1500  # Degree
    c = 0.05  # Fraction of vertices rerouted

    print("Starting graph generation...")
    adjacency_matrix, partitions = generate_partitioned_graph(n, d)

    print("Converting to NetworkX graph...")
    G = nx.from_numpy_array(adjacency_matrix)

    print("Checking degree distribution...")
    degrees = [degree for node, degree in G.degree()]
    print(f"Average degree: {np.mean(degrees):.2f}")

    print("Checking connectivity...")
    if nx.is_connected(G):
        print("The graph is connected.")
    else:
        print("The graph is NOT connected.")

    # Plot degree distributions
    plot_degree_distribution(adjacency_matrix, partitions)

    # Check eigenvalues
    print("Calculating eigenvalues...")
    eigenvalues = np.linalg.eigh(adjacency_matrix)[0]
    print(f"Largest eigenvalue: {eigenvalues[-1]:.4f}")
    print(f"Second largest eigenvalue: {eigenvalues[-2]:.4f}")
    print(f"Smallest eigenvalue: {eigenvalues[0]:.4f}")

    print("Graph generation complete.")
