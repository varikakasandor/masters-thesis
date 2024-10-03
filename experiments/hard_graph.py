import random
from tools import *
from scipy.sparse import block_diag
from itertools import product


def create_random_regular_bipartite_graph(k, d):
    """Creates a random d-regular bipartite graph with k vertices on each side."""
    left_vertices = np.arange(k)
    right_vertices = np.arange(k, 2 * k)  # Adjust right set vertices to be distinct
    adjacency_matrix = np.zeros((2 * k, 2 * k), dtype=int)  # Matrix for k left and k right vertices

    for i in range(k):
        neighbors = np.random.choice(right_vertices - k, size=d, replace=False)  # Neighbors from the right set
        adjacency_matrix[i, neighbors + k] = adjacency_matrix[neighbors + k, i] = 1  # +k to adjust indices to the right side

    return adjacency_matrix


def create_high_expansion_high_threshold_rank_graph(n, k, d, d_inter):
    """
    Creates a d-regular graph G of n vertices consisting of two copies of H,
    between which is a random d_inter-regular bipartite graph. Each H consists of
    n/(2k) disjoint copies of a random (d - d_inter)-regular bipartite graph F.

    Parameters:
        n (int): Total number of vertices in G.
        k (int): Number of vertices on each side of bipartite graph F.
        d (int): Degree of the regular graph G.
        d_inter (int): Degree of the inter-regular bipartite graph between H copies.

    Returns:
        adjacency_matrix (np.ndarray): The adjacency matrix of the graph G.
    """
    assert n % (4 * k) == 0, "n must be divisible by 2k"

    n_half = n // 2
    # Number of copies of F in each H
    num_copies_of_F = (n_half) // (2 * k)

    # Create H1 and H2 (each of size n/2) consisting of disjoint copies of F
    H1 = np.zeros((n_half, n_half))
    H2 = np.zeros((n_half, n_half))

    for i in range(num_copies_of_F):
        # Generate a random (d - d_inter)-regular bipartite graph F
        F = create_random_regular_bipartite_graph(k, d - d_inter)
        # Place F in the corresponding block of H1 and H2
        H1[i * (2 * k):(i + 1) * (2 * k), i * (2 * k):(i + 1) * (2 * k)] = F
        H2[i * (2 * k):(i + 1) * (2 * k), i * (2 * k):(i + 1) * (2 * k)] = F

    # Create the random d_inter-regular bipartite graph between H1 and H2
    inter_bipartite_graph = create_random_regular_bipartite_graph(n_half, d_inter)

    # Combine H1, H2, and the inter-bipartite graph into the full adjacency matrix
    adjacency_matrix = inter_bipartite_graph
    # Fill in H1 in the top-left block
    adjacency_matrix[:n_half, :n_half] = H1
    # Fill in H2 in the bottom-right block
    adjacency_matrix[n_half:, n_half:] = H2

    return adjacency_matrix


def iterate_and_analyse(max_samples=100):
    n = 2048  # Fixed number of vertices
    samples = []

    for k in [64]: # range(1, n // 16 + 1):
        if n % (4 * k) != 0:
            continue  # Skip if 2048 is not divisible by 4k

        for d in [64]: #range(1, k):
            for d_inter in range(1, d):
                if d - d_inter >= k:
                    continue  # Skip if d - d_inter is not less than k

                # Append valid combinations to sample from
                samples.append((k, d, d_inter))

    # Randomly sample up to max_samples combinations
    chosen_samples = random.sample(samples, min(max_samples, len(samples)))

    for k, d, d_inter in chosen_samples:
        # Generate the graph and compute the eigenvalues
        graph = create_high_expansion_high_threshold_rank_graph(n, k, d, d_inter)
        eigenvalues = analyse_spectrum(graph, print_info=False)

        # Sort eigenvalues to access the 2nd and penultimate
        sorted_eigenvalues = np.sort(eigenvalues)

        # Print the 2nd and penultimate eigenvalues
        print(
            f"k={k}, d={d}, d_inter={d_inter}, 2nd eigenvalue={sorted_eigenvalues[1]}, penultimate eigenvalue={sorted_eigenvalues[-2]}")


if __name__ == "__main__":
    # analyse_spectrum(create_high_expansion_high_threshold_rank_graph(1024, 16, 64, 56))
    iterate_and_analyse()
