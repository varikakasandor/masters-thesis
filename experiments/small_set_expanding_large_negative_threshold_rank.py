import numpy as np
import random


def build_random_bipartite_edges(block_nodes, left_nodes, right_nodes, deg):
    """
    Approximate a 'deg'-regular bipartite subgraph between left_nodes and right_nodes
    by assigning edges randomly from each node on the left to 'deg' nodes on the right.
    Does NOT guarantee exact regularity on the right side!
    Returns a list of (u, v) edges (u in left, v in right).
    """
    edges = []
    right_list = list(right_nodes)
    for u in left_nodes:
        chosen = random.sample(right_list, deg)
        for v in chosen:
            edges.append((u, v))
    return edges


def create_graph(n, r, d, d1):
    """
    Create an adjacency matrix for the described multi-block bipartite-like random graph.
    n: total # of vertices
    r: # of blocks
    d: main degree (sqrt(n))
    d1: smaller degree (ln n)
    """
    A = np.zeros((n, n), dtype=float)

    block_size = n // r
    blocks = [range(i * block_size, (i + 1) * block_size) for i in range(r)]

    # Build within-block bipartite edges
    # We'll split each block into two halves (left, right).
    for b in range(r):
        block = list(blocks[b])
        half = block_size // 2
        left = block[:half]
        right = block[half:]

        deg_within = d - d1
        edges = build_random_bipartite_edges(block, left, right, deg_within)
        for (u, v) in edges:
            A[u, v] = 1
            A[v, u] = 1

    # Build cross-block edges
    # For each pair of blocks (b1, b2), connect them bipartitely with ~ d1/(r-1) edges per node.
    deg_cross = d1 // (r - 1)
    for b1 in range(r):
        for b2 in range(b1 + 1, r):
            block1 = list(blocks[b1])
            block2 = list(blocks[b2])
            edges12 = build_random_bipartite_edges(block1 + block2, block1, block2, deg_cross)
            for (u, v) in edges12:
                A[u, v] = 1
                A[v, u] = 1

    return A


def compute_small_set_vertex_expansion(A, delta, n, trials=20):
    results = []
    for _ in range(trials):
        S = set(random.sample(range(n), int(delta * n)))
        affected = set()
        for v in S:
            neighbors = np.where(A[v] > 0)[0]
            affected.update(neighbors)
        affected.difference_update(S)  # Remove vertices in S
        results.append(len(affected) / len(S))
    return results

if __name__ == "__main__":
    # Example usage
    n = 3000
    r = 5#int(np.log(np.log(n)))  # ~ ln ln n
    d = 200#int(np.sqrt(n))  # ~ sqrt(n)
    d1 = 20#int(np.log(n))  # ~ ln(n)
    delta = 1 / 3
    print(f"Building graph with n={n}, r={r}, d={d}, d1={d1}")
    A = create_graph(n, r, d, d1)

    # Compute eigenvalues of A
    print("Computing eigenvalues...")
    eigvals = np.linalg.eigvals(A)
    eigvals.sort()

    print("Computing affected vertices...")
    results = compute_small_set_vertex_expansion(A, delta, n)
    print("Affected vertices in 20 trials:")
    print(results)
    print(f"Average affected vertices: {np.mean(results)}")

    # print("Eigenvalues (sorted):")
    # print(eigvals)
    # Print the first 10 and last 10 eigenvalues
    print("First 10 eigenvalues:")
    print(eigvals[:10])

    print("Last 10 eigenvalues:")
    print(eigvals[-10:])
