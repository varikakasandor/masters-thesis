import random
import networkx as nx
import matplotlib.pyplot as plt


def generate_d_regular_bipartite_graph(n, d):
    if d * n % 2 != 0:
        raise ValueError("The product of d and n must be even to generate a valid bipartite graph.")

    # Create stubs
    left_stubs = [(i, 'left') for i in range(n) for _ in range(d)]
    right_stubs = [(i, 'right') for i in range(n) for _ in range(d)]

    # Shuffle the stubs
    random.shuffle(left_stubs)
    random.shuffle(right_stubs)

    # Create a bipartite graph
    G = nx.Graph()
    G.add_nodes_from(range(n), bipartite=0)  # left nodes
    G.add_nodes_from(range(n, 2 * n), bipartite=1)  # right nodes

    # Pair stubs
    while left_stubs and right_stubs:
        left_node = left_stubs.pop()
        right_node = right_stubs.pop()
        G.add_edge(left_node[0], right_node[0] + n)

    # Check for multiple edges or self-loops
    if len(G.edges()) != d * n:
        raise ValueError("Graph generation failed, try again.")

    return G


if __name__ == "__main__":

    # Parameters
    n = 1000  # Number of vertices on each side
    d = 2  # Degree of each vertex

    # Generate the graph
    try:
        G = generate_d_regular_bipartite_graph(n, d)

        # Plotting the generated graph
        pos = nx.bipartite_layout(G, nodes=list(range(n)))
        nx.draw(G, pos, with_labels=True, node_color=['skyblue' if i < n else 'lightgreen' for i in G.nodes()])
        plt.show()
    except ValueError as e:
        print(e)
