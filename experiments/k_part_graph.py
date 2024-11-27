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
        [0.00000, 0.00500, 0.00624, 0.00416, 0.18582, 0.00004, 0.06185, 0.10593, 0.08334, 0.18027, 0.00991, 0.19307,
         0.00000, 0.00004, 0.16432],
        [0.00500, 0.00000, 0.18426, 0.08820, 0.00004, 0.00000, 0.00005, 0.18330, 0.00001, 0.03980, 0.10567, 0.16827,
         0.00000, 0.10778, 0.11762],
        [0.00624, 0.18426, 0.00000, 0.02759, 0.17709, 0.01047, 0.10945, 0.00000, 0.18047, 0.15582, 0.00001, 0.03400,
         0.11452, 0.00001, 0.00006],
        [0.00416, 0.08820, 0.02759, 0.00000, 0.15026, 0.16166, 0.19926, 0.00000, 0.00012, 0.00001, 0.16047, 0.02982,
         0.00000, 0.00000, 0.17908],
        [0.18582, 0.00004, 0.17709, 0.15026, 0.00000, 0.05919, 0.00000, 0.07397, 0.00004, 0.00009, 0.04440, 0.00004,
         0.10568, 0.20336, 0.00001],
        [0.00004, 0.00000, 0.01047, 0.16166, 0.05919, 0.00000, 0.00000, 0.17342, 0.14564, 0.17068, 0.00011, 0.15799,
         0.00707, 0.07746, 0.03608],
        [0.06185, 0.00005, 0.10945, 0.19926, 0.00000, 0.00000, 0.00000, 0.17085, 0.00696, 0.13424, 0.00005, 0.07017,
         0.09819, 0.14892, 0.00000],
        [0.10593, 0.18330, 0.00000, 0.00000, 0.07397, 0.17342, 0.17085, 0.00000, 0.12649, 0.00001, 0.01819, 0.00003,
         0.14793, 0.00001, 0.00020],
        [0.08334, 0.00001, 0.18047, 0.00012, 0.00004, 0.14564, 0.00696, 0.12649, 0.00000, 0.00000, 0.15342, 0.00013,
         0.00002, 0.14647, 0.15688],
        [0.18027, 0.03980, 0.15582, 0.00001, 0.00009, 0.17068, 0.13424, 0.00001, 0.00000, 0.00000, 0.19813, 0.00003,
         0.03794, 0.03122, 0.05174],
        [0.00991, 0.10567, 0.00001, 0.16047, 0.04440, 0.00011, 0.00005, 0.01819, 0.15342, 0.19813, 0.00000, 0.08832,
         0.14460, 0.07669, 0.00002],
        [0.19307, 0.16827, 0.03400, 0.02982, 0.00004, 0.15799, 0.07017, 0.00003, 0.00013, 0.00003, 0.08832, 0.00000,
         0.14015, 0.11795, 0.00002],
        [0.00000, 0.00000, 0.11452, 0.00000, 0.10568, 0.00707, 0.09819, 0.14793, 0.00002, 0.03794, 0.14460, 0.14015,
         0.00000, 0.00000, 0.20388],
        [0.00004, 0.10778, 0.00001, 0.00000, 0.20336, 0.07746, 0.14892, 0.00001, 0.14647, 0.03122, 0.07669, 0.11795,
         0.00000, 0.00000, 0.09008],
        [0.16432, 0.11762, 0.00006, 0.17908, 0.00001, 0.03608, 0.00000, 0.00020, 0.15688, 0.05174, 0.00002, 0.00002,
         0.20388, 0.09008, 0.00000]
    ])

    n = density_matrix.shape[0] * 200  # Total number of nodes
    d = 1000  # Average degree, can be adjusted
    assert n % density_matrix.shape[0] == 0

    # Create graph
    G = create_stochastic_block_graph(n, d, density_matrix)

    # Optionally plant a random 3-coloring in it
    # G = plant_random_3_coloring(G)

    # Plot degree distribution
    plot_degree_distribution(G)

    graph_connected_message = "The graph is connected" if nx.is_connected(G) else "The graph is NOT connected"
    print(graph_connected_message)

    # Get adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    analyse_spectrum(adjacency_matrix, bottom_print_cnt=8)
