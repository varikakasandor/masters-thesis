from tools import *
from scipy.sparse import block_diag
from itertools import product


def create_complete_bipartite(d):
    """Create the adjacency matrix of a complete bipartite graph K(d/2, d/2)."""
    half_d = d // 2
    F = np.zeros((d, d))
    F[:half_d, half_d:] = 1
    F[half_d:, :half_d] = 1
    return F


def create_block_diagonal_F(n, d):
    """Create the adjacency matrix for H consisting of n/(2d) disjoint copies of F."""
    num_blocks = n // (2 * d)  # Number of disjoint copies of F
    F = create_complete_bipartite(d)
    blocks = [F] * num_blocks
    H = block_diag(blocks).toarray()
    return H


def create_random_bipartite_regular(n_half, d):
    """Create a random d/2-regular bipartite graph adjacency matrix between two halves of size n_half."""
    half_d = d // 2
    left_vertices = np.arange(n_half)
    right_vertices = np.arange(n_half)

    adjacency_matrix = np.zeros((n_half, n_half), dtype=int)

    for i in range(n_half):
        neighbors = np.random.choice(right_vertices, size=half_d, replace=False)
        adjacency_matrix[i, neighbors] = 1

    return adjacency_matrix


def create_high_expansion_high_threshold_rank_graph(n, d):
    """Create the adjacency matrix for the graph G."""
    # Step 1: Create two disjoint copies of H
    H = create_block_diagonal_F(n, d)

    # Step 2: Create a random bipartite d/2-regular graph between two copies of H
    n_half = n // 2
    bipartite_matrix = create_random_bipartite_regular(n_half, d)

    # Construct full adjacency matrix
    adjacency_matrix = np.zeros((n, n))

    # Place H in the top-left and bottom-right blocks
    adjacency_matrix[:n_half, :n_half] = H
    adjacency_matrix[n_half:, n_half:] = H

    # Place the bipartite connections
    adjacency_matrix[:n_half, n_half:] = bipartite_matrix
    adjacency_matrix[n_half:, :n_half] = bipartite_matrix.T  # Symmetric adjacency matrix

    return adjacency_matrix


if __name__ == "__main__":
    analyse_spectrum(create_high_expansion_high_threshold_rank_graph(3000, 250))
    # for d in [x for x in range(10, 1000) if 3000 % (4 * x) == 0]:
    #    evals = analyse_spectrum(create_high_expansion_high_threshold_rank_graph(3000, d))
    #    print(d, find_first_jump(evals), (float(evals[-2]), float(evals[-1])))


"""
def create_bipartite_bipartite_old(n, k, d):
    # n should be divisible by 4k
    if n % (4 * k) != 0:
        raise ValueError("n must be divisible by 4k")

    # F is a complete bipartite graph with k vertices on each side
    F = np.block([
        [np.zeros((k, k)), np.ones((k, k))],
        [np.ones((k, k)), np.zeros((k, k))]
    ])

    # Number of disjoint copies of F in H (which has n/2 vertices)
    copies_of_F = n // (4 * k)

    # Create H as a block diagonal matrix of copies_of_F disjoint copies of F
    H = np.block([[F if i == j else np.zeros_like(F) for j in range(copies_of_F)] for i in range(copies_of_F)])

    # Create G by making two copies of H and connecting them with a complete bipartite graph
    # G_upper = np.block([[H, np.ones_like(H)], [np.ones_like(H), H]])
    bipartite_random = create_random_bipartite_d_regular(H.shape[0], d)
    G_upper = np.block([[H, bipartite_random], [bipartite_random.T, H]])

    return G_upper

"""