import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import random
import matplotlib.pyplot as plt



def generate_random_bipartite_graph(n1, n2, d):
    """Generate a random d-regular bipartite graph between two parts of size n1 and n2."""
    G = nx.Graph()
    G.add_nodes_from(range(n1), bipartite=0)
    G.add_nodes_from(range(n1, n1 + n2), bipartite=1)

    # Stub matching for bipartite graph
    left_stubs = list(np.repeat(range(n1), d))
    right_stubs = list(np.repeat(range(n1, n1 + n2), d))

    random.shuffle(left_stubs)
    random.shuffle(right_stubs)

    while left_stubs and right_stubs:
        u = left_stubs.pop()
        v = right_stubs.pop()
        G.add_edge(u, v)

    return G


def generate_graph(n, k, d):
    part_size = n // (2 * k)
    parts = []
    G = nx.Graph()
    offset = 0

    # Create and store node indices for each part
    for i in range(2 * k):
        part = list(range(offset, offset + part_size))
        parts.append(part)
        offset += part_size
        G.add_nodes_from(part)

    # Add bipartite graphs between union of (V_{i1} ∪ V_{i2}) and (V_{j1} ∪ V_{j2})
    bipartite_degree = d // (2 * (k - 1))
    for i in range(k):
        for j in range(i + 1, k):
            U = parts[2 * i] + parts[2 * i + 1]
            V = parts[2 * j] + parts[2 * j + 1]
            B = generate_random_bipartite_graph(len(U), len(V), bipartite_degree)
            mapping = dict(zip(B.nodes, U + V))
            B = nx.relabel_nodes(B, mapping)
            G.add_edges_from(B.edges)

    # Add bipartite graphs between V_{i1} and V_{(i+1)2} (cycling around at k)
    for i in range(k):
        B = generate_random_bipartite_graph(part_size, part_size, d // 2)
        mapping = dict(zip(B.nodes, parts[i * 2] + parts[((i + 1) % k) * 2 + 1]))
        B = nx.relabel_nodes(B, mapping)
        G.add_edges_from(B.edges)

    return G


if __name__ == "__main__":

    # Parameters
    n = 1000
    k = 5
    d = 80

    G = generate_graph(n, k, d)
    A = nx.to_scipy_sparse_array(G, format='csr')

    # Compute two largest eigenvalues of the adjacency matrix
    A = A.astype(np.float64)
    vals = eigsh(A, k=2, which='LA', return_eigenvectors=False)
    lambda_1, lambda_2 = sorted(vals, reverse=True)

    print(lambda_1, lambda_2)

    degrees = [deg for _, deg in G.degree()]
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left', rwidth=0.8)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.show()
