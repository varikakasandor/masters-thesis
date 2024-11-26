from tools import *
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt


def plot_degree_distribution(graph):
    degrees = [degree for node, degree in graph.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def create_stochastic_block_graph(n, d, density_matrix):
    # Number of nodes in each part
    num_parts = density_matrix.shape[0]
    nodes_per_part = n // num_parts

    # Assign nodes to parts
    node_groups = {i: list(range(i * nodes_per_part, (i + 1) * nodes_per_part)) for i in range(num_parts)}

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges based on matrix B
    for i in range(num_parts):
        for j in range(num_parts):
            if i <= j:  # Only iterate over upper triangle and diagonal
                prob = density_matrix[i, j] * d / n
                for u in node_groups[i]:
                    for v in node_groups[j]:
                        if u < v:  # Avoid double counting
                            if random.random() < prob:
                                G.add_edge(u, v)
    return G


if __name__ == "__main__":
    # Parameters
    n = 2400  # Total number of nodes
    d = 200  # Average degree, can be adjusted

    # Matrix B
    density_matrix = np.array([
        [3.57583434e-03, 2.42715849e+00, 9.22767189e-02, 1.41958957e+00, 9.25309287e-02, 2.53229827e-02, 4.19833773e-01,
         2.40118690e-02],
        [2.55422924e-02, 6.92711951e-03, 2.77372728e-02, 5.14528089e-03, 8.17989211e-03, 3.31565402e-03, 3.59439103e-02,
         1.56034663e-03],
        [1.87545754e-02, 7.77643161e-01, 7.04511456e-03, 4.89209063e-02, 1.50538965e-01, 1.60215636e-01, 4.56029806e-01,
         1.44790606e-02],
        [1.21405312e-01, 1.79934314e-01, 1.50047085e-01, 5.08182559e-03, 6.79629424e-03, 2.65660359e-02, 4.03551357e-01,
         8.42110121e-02],
        [5.16204346e-02, 1.17298621e-01, 9.66528544e-03, 1.46028321e-02, 1.25295506e-01, 5.72155151e-02, 2.21849838e-02,
         9.36130761e-03],
        [1.38689795e-02, 3.13541694e+00, 8.81019733e-02, 8.79191379e-02, 1.79590517e-01, 4.39026795e-03, 3.43621720e-01,
         1.19935700e-01],
        [3.83143816e-02, 1.72445772e-01, 5.50673281e-01, 6.37928211e-02, 1.23036304e-01, 2.63097219e-02, 1.21070200e-03,
         2.01580332e-02],
        [1.55602946e-01, 9.58000106e-01, 1.75924482e-01, 2.38610106e-01, 6.74645700e-02, 1.51635484e+00, 4.94896571e-02,
         1.31453894e-02]
    ])

    # Create graph
    G = create_stochastic_block_graph(n, d, density_matrix)

    # Plot degree distribution
    plot_degree_distribution(G)

    # Get adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G).todense()

    analyse_spectrum(adjacency_matrix)
