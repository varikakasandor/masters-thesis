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

    # Add edges based on matrix B
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
        [0.00100, 0.28269, 0.14027, 0.18009, 0.05299, 0.32257, 0.01896, 0.00144],
        [0.28269, 0.00100, 0.04290, 0.13618, 0.01317, 0.07256, 0.35141, 0.10009],
        [0.14027, 0.04290, 0.00113, 0.34254, 0.28074, 0.05560, 0.13581, 0.00101],
        [0.18009, 0.13618, 0.34254, 0.00108, 0.00722, 0.02365, 0.04859, 0.26064],
        [0.05299, 0.01317, 0.28074, 0.00722, 0.00113, 0.27551, 0.26610, 0.10313],
        [0.32257, 0.07256, 0.05560, 0.02365, 0.27551, 0.00132, 0.02612, 0.22266],
        [0.01896, 0.35141, 0.13581, 0.04859, 0.26610, 0.02612, 0.00100, 0.15200],
        [0.00144, 0.10009, 0.00101, 0.26064, 0.10313, 0.22266, 0.15200, 0.15903]
    ])

    # Create graph
    G = create_stochastic_block_graph(n, d, density_matrix)

    # Plot degree distribution
    plot_degree_distribution(G)

    graph_connected_message = "The graph is connected" if nx.is_connected(G) else "The graph is NOT connected"
    print(graph_connected_message)

    # Get adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    analyse_spectrum(adjacency_matrix)
