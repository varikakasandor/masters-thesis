from tools import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def create_almost_bipartite(n, eps, d, d_prime):
    # Assertions to ensure divisibility conditions
    assert eps * n % 1 == 0, "eps * n must be an integer"
    assert (1 - eps) * n % 2 == 0, "(1 - eps) * n must be divisible by 2"
    assert (int(eps * n * d)) % (int((1 - eps) * n)) == 0, "eps * n * d must be divisible by (1 - eps) * n"

    # Split the vertices into three parts: C, A, B
    C_size = int(eps * n)
    AB_size = (int((1 - eps) * n)) // 2

    vertices = np.random.permutation(n)
    C = vertices[:C_size]
    A = vertices[C_size:(C_size + AB_size)]
    B = vertices[(C_size + AB_size):]

    # Create empty graph with n nodes
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add a random d'-regular graph within A
    G_A = nx.random_regular_graph(d_prime, len(A))
    G.add_edges_from([(A[u], A[v]) for u, v in G_A.edges()])

    # Add a random d'-regular graph within B
    G_B = nx.random_regular_graph(d_prime, len(B))
    G.add_edges_from([(B[u], B[v]) for u, v in G_B.edges()])

    # Add a regular bipartite graph between A and B
    d_between_AB = d - d_prime - (int(eps * n * d)) // (int((1 - eps) * n))
    G_AB = create_biregular(len(A), len(B), d_between_AB, d_between_AB)
    add_bipartite_subgraph(G, G_AB, A, B)

    # Add a regular bipartite graph between A and C
    d_AC = d_BC = d - d_prime - d_between_AB
    G_AC = create_biregular(len(A), len(C), d_AC, d // 2)
    add_bipartite_subgraph(G, G_AC, A, C)

    # Add a regular bipartite graph between B and C
    G_BC = create_biregular(len(B), len(C), d_BC, d // 2)
    add_bipartite_subgraph(G, G_BC, B, C)
    return G, A, B, C


def create_union_of_almost_bipartites(n, eps, k, d, d_prime):
    graphs = []
    for _ in range(k):
        G, _, _, _ = create_almost_bipartite(n, eps, d, d_prime)
        graphs.append(G)
    final_graph = nx.Graph()
    for G in graphs:
        final_graph = nx.compose(final_graph, G)

    return final_graph

def plot_degree_distribution(graph):
    degrees = [degree for node, degree in graph.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    n = 1000
    eps = 0.01
    k = 1
    d = 2 * 99
    d_prime = 4


    graph = create_union_of_almost_bipartites(n, eps, k, d, d_prime)
    plot_degree_distribution(graph)
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    analyse_spectrum(adjacency_matrix)


    """G, A, B, C = create_almost_bipartite(n, eps, d, d_prime)
    print(calculate_lambda_n_bound(G, A, B, C, d))
    adjacency_matrix = nx.adjacency_matrix(G).todense()
    analyse_spectrum(adjacency_matrix)"""