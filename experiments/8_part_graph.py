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
        [0.00000, 0.02670, 0.19414, 0.06657, 0.27480, 0.34947, 0.05822, 0.03012],
        [0.02670, 0.02281, 0.01004, 0.08708, 0.05447, 0.28525, 0.33166, 0.18201],
        [0.19414, 0.01004, 0.05084, 0.30032, 0.07428, 0.06199, 0.29378, 0.01462],
        [0.06657, 0.08708, 0.30032, 0.00003, 0.01096, 0.16554, 0.03216, 0.33737],
        [0.27480, 0.05447, 0.07428, 0.01096, 0.08020, 0.01227, 0.20091, 0.29212],
        [0.34947, 0.28525, 0.06199, 0.16554, 0.01227, 0.00000, 0.03250, 0.09299],
        [0.05822, 0.33166, 0.29378, 0.03216, 0.20091, 0.03250, 0.00000, 0.05079],
        [0.03012, 0.18201, 0.01462, 0.33737, 0.29212, 0.09299, 0.05079, 0.00000]
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
