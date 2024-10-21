from matplotlib import pyplot as plt

from tools import *
import networkx as nx


def union_of_almost_bipartites(n, eps, k, d, d_prime):
    graphs = []
    for _ in range(k):
        # Split the vertices into three parts: C, A, B
        C_size = int(eps * n)
        AB_size = int((1 - eps) * n / 2)

        vertices = np.random.permutation(n)
        C = vertices[:C_size]
        A = vertices[C_size:C_size + AB_size]
        B = vertices[C_size + AB_size:]

        # Create empty graph with n nodes
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Add a random d'-regular graph within A
        G_A = nx.random_regular_graph(d_prime, len(A))
        G.add_edges_from([(A[u], A[v]) for u, v in G_A.edges()])

        # Add a random d'-regular graph within B
        G_B = nx.random_regular_graph(d_prime, len(B))
        G.add_edges_from([(B[u], B[v]) for u, v in G_B.edges()])

        # Add a random bipartite graph between A and B with the given degree
        d_between_AB = d - d_prime - int(eps * n * d / ((1 - eps) * n))
        G_AB = nx.algorithms.bipartite.random_graph(len(A), len(B), d_between_AB / len(B))
        G.add_edges_from([(A[u], B[v - len(A)]) for u, v in G_AB.edges()])

        # Add a random bipartite graph between A and C with the given degree
        p_between_AC = d / ((1 - eps) * n)
        G_AC = nx.algorithms.bipartite.random_graph(len(A), len(C), p_between_AC)
        G.add_edges_from([(A[u], C[v - len(A)]) for u, v in G_AC.edges()])  # if u < len(A) and v >= len(A)

        # Add a random bipartite graph between B and C with the given degree
        p_between_BC = d / ((1 - eps) * n)
        G_BC = nx.algorithms.bipartite.random_graph(len(B), len(C), p_between_BC)
        G.add_edges_from([(B[u], C[v - len(B)]) for u, v in G_BC.edges()])

        graphs.append(G)

    # Union of k graphs
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
    # Example usage
    n = 1000
    eps = 0.02
    k = 1
    d = 50
    d_prime = 10
    graph = union_of_almost_bipartites(n, eps, k, d, d_prime)
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    plot_degree_distribution(graph)
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    analyse_spectrum(adjacency_matrix)
