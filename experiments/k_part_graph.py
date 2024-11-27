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
    density_matrix = np.array([
        [0.00000, 0.14989, 0.09608, 0.06400, 0.01222, 0.09141, 0.00200, 0.19726, 0.04909, 0.14955, 0.08620, 0.00379,
         0.03878, 0.05973],
        [0.14989, 0.00000, 0.00657, 0.03559, 0.12690, 0.03328, 0.07095, 0.00224, 0.13346, 0.02183, 0.08993, 0.09404,
         0.09711, 0.13821],
        [0.09608, 0.00657, 0.00000, 0.00134, 0.11778, 0.06806, 0.06930, 0.02423, 0.15616, 0.05395, 0.08591, 0.04274,
         0.12485, 0.15303],
        [0.06400, 0.03559, 0.00134, 0.00000, 0.09989, 0.06559, 0.00894, 0.08805, 0.13178, 0.04138, 0.13291, 0.02184,
         0.10726, 0.20143],
        [0.01222, 0.12690, 0.11778, 0.09989, 0.00000, 0.14599, 0.10343, 0.14211, 0.06444, 0.04456, 0.01584, 0.00754,
         0.10893, 0.01037],
        [0.09141, 0.03328, 0.06806, 0.06559, 0.14599, 0.00000, 0.03991, 0.02677, 0.01330, 0.12417, 0.15446, 0.14657,
         0.00595, 0.08455],
        [0.00200, 0.07095, 0.06930, 0.00894, 0.10343, 0.03991, 0.00000, 0.13765, 0.03549, 0.18096, 0.14776, 0.05943,
         0.01091, 0.13328],
        [0.19726, 0.00224, 0.02423, 0.08805, 0.14211, 0.02677, 0.13765, 0.00000, 0.04442, 0.00085, 0.05292, 0.16727,
         0.06788, 0.04835],
        [0.04909, 0.13346, 0.15616, 0.13178, 0.06444, 0.01330, 0.03549, 0.04442, 0.00000, 0.13699, 0.08602, 0.12190,
         0.01254, 0.00161],
        [0.14955, 0.02183, 0.05395, 0.04138, 0.04456, 0.12417, 0.18096, 0.00085, 0.13699, 0.00000, 0.00621, 0.05926,
         0.14019, 0.04217],
        [0.08620, 0.08993, 0.08591, 0.13291, 0.01584, 0.15446, 0.14776, 0.05292, 0.08602, 0.00621, 0.00000, 0.02799,
         0.11262, 0.00124],
        [0.00379, 0.09404, 0.04274, 0.02184, 0.00754, 0.14657, 0.05943, 0.16727, 0.12190, 0.05926, 0.02799, 0.00000,
         0.14726, 0.10035],
        [0.03878, 0.09711, 0.12485, 0.10726, 0.10893, 0.00595, 0.01091, 0.06788, 0.01254, 0.14019, 0.11262, 0.14726,
         0.00000, 0.02571],
        [0.05973, 0.13821, 0.15303, 0.20143, 0.01037, 0.08455, 0.13328, 0.04835, 0.00161, 0.04217, 0.00124, 0.10035,
         0.02571, 0.00000]
    ])

    n = density_matrix.shape[0] * 200  # Total number of nodes
    d = 1000  # Average degree, can be adjusted
    assert n % density_matrix.shape[0] == 0

    # Create graph
    G = create_stochastic_block_graph(n, d, density_matrix)

    # Optionally plant a random 3-coloring in it
    G = plant_random_3_coloring(G)

    # Plot degree distribution
    plot_degree_distribution(G)

    graph_connected_message = "The graph is connected" if nx.is_connected(G) else "The graph is NOT connected"
    print(graph_connected_message)

    # Get adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    analyse_spectrum(adjacency_matrix, bottom_print_cnt=8)
