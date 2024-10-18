import networkx as nx
import numpy as np
from numpy.linalg import eigvalsh
import random


# Function to compute the spectral gap and return lambda_1 and lambda_2
def spectral_info(graph):
    A = nx.adjacency_matrix(graph).todense()  # Get adjacency matrix
    eigenvalues = eigvalsh(A)  # Get eigenvalues
    lambda_1 = eigenvalues[-1]  # Largest eigenvalue
    lambda_2 = eigenvalues[-2]  # Second largest eigenvalue
    spectral_gap = lambda_1 - lambda_2
    return lambda_1, lambda_2, spectral_gap


# Generate a random d-regular graph
def generate_d_regular_graph(n, d):
    return nx.random_regular_graph(d, n)


# Plant a 3-colouring by removing intra-group edges
def plant_3_coloring(graph, n):
    # Randomly split vertices into 3 groups
    groups = {i: np.random.choice([0, 1, 2]) for i in range(n)}

    # Remove intra-group edges
    edges_to_remove = [(u, v) for u, v in graph.edges if groups[u] == groups[v]]
    graph.remove_edges_from(edges_to_remove)

    return graph, groups


# Restore regularity by adding as few edges as possible
def restore_regularity(graph, n, d):
    # Compute the degree deficit for each node
    degree_deficit = {node: d - graph.degree[node] for node in graph.nodes if graph.degree[node] < d}

    # Get a list of nodes with degree deficit
    nodes_with_deficit = [node for node, deficit in degree_deficit.items() for _ in range(deficit)]

    # Randomly pair nodes from different groups to add edges until the graph is regular
    while len(nodes_with_deficit) > 1:
        u = random.choice(nodes_with_deficit)
        nodes_with_deficit.remove(u)
        # Ensure we don't create self-loops or multi-edges
        v = random.choice([v for v in nodes_with_deficit if not graph.has_edge(u, v) and u != v])
        nodes_with_deficit.remove(v)
        graph.add_edge(u, v)

    return graph


# Print spectral information
def print_spectral_info(graph, step_description):
    lambda_1, lambda_2, spectral_gap = spectral_info(graph)
    proportion = lambda_2 / lambda_1
    print(f"Spectral info after {step_description}:")
    print(f"  λ1 (largest eigenvalue)    : {lambda_1:.4f}")
    print(f"  λ2 (second largest eigenvalue): {lambda_2:.4f}")
    print(f"  Spectral gap (λ1 - λ2)      : {spectral_gap:.4f}")
    print(f"  Proportion λ2 / λ1          : {proportion:.4f}\n")


# Parameters
n = 1000  # Number of nodes
d = 100  # Degree

# Step 1: Generate random d-regular graph
graph = generate_d_regular_graph(n, d)
print_spectral_info(graph, "initial d-regular graph")

# Step 2: Plant the first 3-coloring and remove intra-group edges
graph, groups = plant_3_coloring(graph, n)
print_spectral_info(graph, "first 3-coloring")

# Step 3: Restore regularity by adding the minimum number of edges
graph = restore_regularity(graph, n, d)
print_spectral_info(graph, "after restoring regularity")
