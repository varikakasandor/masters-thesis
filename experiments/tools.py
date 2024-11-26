from collections import Counter
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def analyse_spectrum(adjacency_matrix, print_info=True, concise=True, bottom_print_cnt=3):
    if print_info and not concise:
        print(
            f"Number of vertices: {adjacency_matrix.shape[0]}, Max degree: {np.max(np.sum(adjacency_matrix, axis=0))}")

    # Calculate the eigenvalues
    eigenvalues, _ = np.linalg.eigh(adjacency_matrix)

    # Reverse the eigenvalues to have them in descending order
    eigenvalues = eigenvalues[::-1]

    # Round the eigenvalues to a reasonable precision to avoid floating point issues
    rounded_eigenvalues = np.round(eigenvalues, decimals=10)

    # Count the multiplicities of each eigenvalue
    eigenvalue_multiplicities = Counter(rounded_eigenvalues)

    # Print out the eigenvalues and their multiplicities
    if print_info and not concise:
        print("Eigenvalue : Multiplicity")
        for eigenvalue, multiplicity in eigenvalue_multiplicities.items():
            print(f"{eigenvalue} : {multiplicity}")

    # Find the smallest and largest eigenvalues
    if print_info and concise:
        largest_eigenvalues = rounded_eigenvalues[:2]  # 2 largest eigenvalues
        smallest_eigenvalues = rounded_eigenvalues[-bottom_print_cnt:]  # bottom_print_cnt smallest eigenvalues
        concise_output = [float(largest_eigenvalues[0]), float(largest_eigenvalues[1]), '...',
                          *[float(val) for val in smallest_eigenvalues]]
        concise_output_str = ', '.join(str(x) if x != '...' else '...' for x in concise_output)
        print(f"Spectrum: [{concise_output_str}]")

    return eigenvalues


def add_bipartite_subgraph(G, G_bip, X, Y):
    for u, v in G_bip.edges():
        node_X = X[u]
        node_Y = Y[v - len(X)]
        G.add_edge(node_X, node_Y)


import networkx as nx


def calculate_lambda_n_bound(G, A, B, C, d):
    # Step 1: Calculate e(A, A) - Number of edges within set A
    e_AA = G.subgraph(A).number_of_edges()

    # Step 2: Calculate e(B, B) - Number of edges within set B
    e_BB = G.subgraph(B).number_of_edges()

    # Step 3: Calculate e(A ∪ B, C) - Number of edges between set (A ∪ B) and set C
    A_union_B = set(A).union(set(B))
    e_AuB_C = 0
    for u in A_union_B:
        for v in C:
            if G.has_edge(u, v):
                e_AuB_C += 1

    # Step 4: Calculate |A ∪ B| - Number of nodes in set A ∪ B
    num_AuB = len(A_union_B)

    # Step 5: Calculate the bound
    bound = (2 * ((2 * e_AA + 2 * e_BB + e_AuB_C) / (d * num_AuB)) - 1) * d

    return bound


def create_biregular(a, b, d_a, d_b):
    """
    Creates a bipartite graph with two sets of vertices A and B.
    Set A has 'a' vertices, each with degree 'd_a'.
    Set B has 'b' vertices, each with degree 'd_b'.
    The graph will be constructed manually without randomization and guarantees no multiedges.

    Parameters:
    a (int): Number of vertices in set A.
    b (int): Number of vertices in set B.
    d_a (int): Degree of each vertex in set A.
    d_b (int): Degree of each vertex in set B.

    Returns:
    A bipartite graph as a NetworkX graph object.
    """
    if a * d_a != b * d_b:
        raise ValueError("The total number of stubs must match: a * d_a == b * d_b")

    # Create an empty bipartite graph
    G = nx.Graph()

    # Add nodes for the two partitions A and B
    G.add_nodes_from(range(a), bipartite=0)  # Set A
    G.add_nodes_from(range(a, a + b), bipartite=1)  # Set B

    # Connect vertices in a round-robin fashion to avoid multiedges
    for i in range(a):
        for j in range(d_a):
            # Connect vertex i in set A to vertex (i + j) % b in set B
            G.add_edge(i, a + (i + j) % b)

    # Step 3: Generate random permutations for each partition
    permuted_A = np.random.permutation(range(a))  # Permutation of nodes in set A
    permuted_B = np.random.permutation(range(a, a + b))  # Permutation of nodes in set B

    # Step 4: Create a mapping from original nodes to permuted nodes within each partition
    mapping = {original: permuted for original, permuted in zip(range(a), permuted_A)}
    mapping.update({original: permuted for original, permuted in zip(range(a, a + b), permuted_B)})

    # Step 5: Relabel the nodes in the graph with the new randomized labels within partitions
    G = nx.relabel_nodes(G, mapping)

    return G


def create_random_bipartite_d_regular(n, d):
    # Creates a random bipartite d-regular graph with n vertices in each partition.

    if d > n:
        raise ValueError("d cannot be greater than n for a bipartite d-regular graph.")

    # Generate a random bipartite d-regular graph using NetworkX
    B = nx.random_regular_graph(d, n * 2)  # Create a regular graph over 2*n vertices
    A = np.zeros((n, n), dtype=int)

    # Add edges only between the two partitions
    for u, v in B.edges():
        if u < n and v >= n:
            A[u, v - n] = 1
        elif v < n and u >= n:
            A[v, u - n] = 1

    return A


def find_first_jump(eigenvalues, threshold=1):
    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - eigenvalues[i - 1]) > threshold:
            return i, float(eigenvalues[i - 1]), float(eigenvalues[i])
    return None


def plot_degree_distribution_from_adjacency_matrix(adjacency_matrix):
    degrees = np.sum(adjacency_matrix, axis=0)
    plt.figure()
    plt.hist(degrees, bins=30, edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()


def plot_degree_distribution(graph):
    degrees = [degree for node, degree in graph.degree()]
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black', alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()