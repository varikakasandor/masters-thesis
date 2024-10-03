from collections import Counter
import numpy as np
import networkx as nx

def analyse_spectrum(adjacency_matrix, print_info=True):
    if print_info:
        print(f"Number of vertices: {adjacency_matrix.shape[0]}, Max degree: {np.max(np.sum(adjacency_matrix, axis=0))}")

    # Calculate the eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)

    # Round the eigenvalues to a reasonable precision to avoid floating point issues
    rounded_eigenvalues = np.round(eigenvalues, decimals=6)

    # Count the multiplicities of each eigenvalue
    eigenvalue_multiplicities = Counter(rounded_eigenvalues)

    # Print out the eigenvalues and their multiplicities
    if print_info:
        print("Eigenvalue : Multiplicity")
        for eigenvalue, multiplicity in eigenvalue_multiplicities.items():
             print(f"{eigenvalue} : {multiplicity}")

    smallest_eigenvalue_index = np.argmin(eigenvalues)
    corresponding_eigenvector = eigenvectors[:, smallest_eigenvalue_index]
    # print(f"Smallest Eigenvector: {corresponding_eigenvector}")
    return eigenvalues

"""
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
    
"""


def find_first_jump(eigenvalues, threshold=1):
    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - eigenvalues[i-1]) > threshold:
            return i, float(eigenvalues[i - 1]), float(eigenvalues[i])
    return None