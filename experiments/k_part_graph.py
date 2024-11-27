import random
from tools import *


def create_stochastic_block_graph(n, d, density_matrix):
    # Number of nodes in each part
    num_parts = density_matrix.shape[0]
    nodes_per_part = n // num_parts

    # Assign nodes to parts
    node_groups = {i: list(range(i * nodes_per_part, (i + 1) * nodes_per_part)) for i in range(num_parts)}

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add each edge based on the density matrix at random
    for i in range(num_parts):
        for j in range(num_parts):
            if i <= j:  # Only iterate over upper triangle and diagonal
                prob = density_matrix[i, j] * d * num_parts / n
                for u in node_groups[i]:
                    for v in node_groups[j]:
                        if u < v:  # Avoid double counting
                            if random.random() < prob:
                                G.add_edge(u, v)
    return G


if __name__ == "__main__":
    # Parameters
    n = 2400  # Total number of nodes
    d = 300  # Average degree, can be adjusted

    density_matrix = np.array([
        [0.00000, 0.24210, 0.01345, 0.00000, 0.10719, 0.25079, 0.16445, 0.03303, 0.04908, 0.13990],
        [0.24210, 0.00000, 0.06302, 0.24893, 0.09850, 0.00031, 0.00000, 0.15293, 0.05682, 0.13739],
        [0.01345, 0.06302, 0.00000, 0.18898, 0.15716, 0.22441, 0.03566, 0.07413, 0.00000, 0.24319],
        [0.00000, 0.24893, 0.18898, 0.00000, 0.02606, 0.10079, 0.22050, 0.07598, 0.13027, 0.00849],
        [0.10719, 0.09850, 0.15716, 0.02606, 0.00000, 0.00342, 0.23911, 0.24865, 0.00000, 0.11991],
        [0.25079, 0.00031, 0.22441, 0.10079, 0.00342, 0.00000, 0.10326, 0.18584, 0.12184, 0.00943],
        [0.16445, 0.00000, 0.03566, 0.22050, 0.23911, 0.10326, 0.00000, 0.00000, 0.16334, 0.07369],
        [0.03303, 0.15293, 0.07413, 0.07598, 0.24865, 0.18584, 0.00000, 0.00000, 0.22005, 0.00939],
        [0.04908, 0.05682, 0.00000, 0.13027, 0.00000, 0.12184, 0.16334, 0.22005, 0.00000, 0.25860],
        [0.13990, 0.13739, 0.24319, 0.00849, 0.11991, 0.00943, 0.07369, 0.00939, 0.25860, 0.00000]
    ])

    # Create graph
    G = create_stochastic_block_graph(n, d, density_matrix)

    # Plot degree distribution
    plot_degree_distribution(G)

    graph_connected_message = "The graph is connected" if nx.is_connected(G) else "The graph is NOT connected"
    print(graph_connected_message)

    # Get adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    analyse_spectrum(adjacency_matrix, bottom_print_cnt=8)
