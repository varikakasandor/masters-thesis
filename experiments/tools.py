from collections import Counter
import random

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def plot_density_matrix(A, filename):
    # Set values below 0.05 to 0
    A[A < 0.05] = 0

    # Create a graph with A.shape[0] nodes
    G = nx.Graph()
    num_nodes = A.shape[0]
    G.add_nodes_from(range(num_nodes))

    # Add edges with weights corresponding to non-zero values in A
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Iterate through upper triangle to avoid duplicate edges
            if A[i, j] != 0:
                G.add_edge(i, j, weight=A[i, j])

    # Set up position and draw the graph
    pos = nx.spring_layout(G)  # You can change layout if desired

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Draw edges with varying thickness based on weight
    edges = G.edges(data=True)
    for u, v, data in edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=data['weight'] * 10)  # Multiply weight to scale thickness

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.axis('off')
    plt.savefig(filename, format='png')  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory


def adjust_row_sums_to_exactly_one(A, max_iter=1000):
    print("Row sum adjustment started")
    n = A.shape[0]
    row_sums = A.sum(axis=1)
    print(f"Row sums at the beginning: {row_sums.astype(float)}")
    iter_cnt = 0
    while not np.all(row_sums == 1) and iter_cnt < max_iter:
        iter_cnt += 1
        # Find the (i, j) with A[i, j] largest such that sum(A[i, :]) and sum(A[j, :]) are both not yet 1
        min_value = np.inf
        i_min, j_min = -1, -1
        for i in range(n):
            if row_sums[i] != 1:
                for j in range(i + 1, n):
                    if (row_sums[j] - 1) * (row_sums[i] - 1) > 0 and 0 < A[
                        i, j] < min_value:  # i.e. neither of them are 1 and they are wrong from the same side
                        min_value = A[i, j]
                        i_min, j_min = i, j
        if i_min == -1 or j_min == -1 or iter_cnt == max_iter:
            print("Cannot make the densities sum up to exactly 1 in each row")
            break  # No valid (i, j) found, meaning all row sums are 1
        # Calculate how much to subtract from A[i_min, j_min] and A[j_min, i_min]
        subtract_amount = min(abs(row_sums[i_min] - 1), abs(row_sums[j_min] - 1), A[i_min, j_min])
        if row_sums[i_min] < 1:
            subtract_amount *= -1

        # Update A and row sums
        A[i_min, j_min] -= subtract_amount
        A[j_min, i_min] -= subtract_amount
        row_sums[i_min] -= subtract_amount
        row_sums[j_min] -= subtract_amount

    return A


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


def plant_random_3_coloring(G):
    n = len(G.nodes)
    nodes = list(G.nodes)
    random.shuffle(nodes)

    # Split nodes into 3 equal parts
    V1 = set(nodes[:n // 3])
    V2 = set(nodes[n // 3: 2 * n // 3])
    V3 = set(nodes[2 * n // 3:])

    # Remove edges within each part
    for V in [V1, V2, V3]:
        edges_to_remove = [(u, v) for u in V for v in V if u != v and G.has_edge(u, v)]
        G.remove_edges_from(edges_to_remove)

    return G


def analyse_2nd_largest_eigenvalue(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.sort(eigenvalues)
    return eigenvalues[-2] / eigenvalues[-1]
