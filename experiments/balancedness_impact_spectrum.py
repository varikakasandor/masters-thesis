import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from tools import analyse_spectrum

def generate_graph(n, alpha, beta, d):
    # Use group sizes: n1 = alpha*n, n2 = beta*n, n3 = n - n1 - n2.
    n1 = int(round(alpha * n))
    n2 = int(round(beta * n))
    n3 = n - n1 - n2
    A = np.zeros((n, n))
    idx1 = np.arange(n1)
    idx2 = np.arange(n1, n1 + n2)
    idx3 = np.arange(n1 + n2, n)
    groups = [idx1, idx2, idx3]
    sizes = [n1, n2, n3]
    # For each pair (i,j), let k be the remaining index.
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        k = 3 - i - j  # since indices are 0,1,2
        # Edge probability between group i and j:
        p = (sizes[i] + sizes[j] - sizes[k]) / (sizes[i] * sizes[j]) * (d / 2)
        p = np.clip(p, 0, 1)
        idx_i = groups[i]
        idx_j = groups[j]
        # Vectorized edge assignment:
        mask = np.random.rand(len(idx_i), len(idx_j)) < p
        A[np.ix_(idx_i, idx_j)] = mask.astype(float)
        A[np.ix_(idx_j, idx_i)] = mask.T.astype(float)
    return A


def compute_second_smallest_eig(alpha, beta, n, d):
    A = generate_graph(n, alpha, beta, d)
    vals = eigvalsh(A)
    return vals[1] / vals[-1]  # normalised smallest eigenvalue


if __name__ == '__main__':
    # Parameters
    n = 4000  # total vertices
    d = 1000  # density parameter
    num_points = 30
    # For a valid triangle (n1, n2, n3 as "side lengths") we need:
    #   n1 <= n2+n3, n2 <= n1+n3, n3 <= n1+n2.
    # With n1 = alpha*n, n2 = beta*n, n3 = n - n1 - n2, this forces:
    #   alpha <= 0.5, beta <= 0.5, and alpha+beta >= 0.5.
    alphas = np.linspace(0, 0.5, num_points)
    betas = np.linspace(0, 0.5, num_points)
    result = np.full((num_points, num_points), np.nan)
    tasks = []
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            if a + b < 0.5 or b < a or 1 - a - b < b:  # skip invalid parameters
                continue
            tasks.append((i, j, a, b))

    # Parallel evaluation over the grid:
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_second_smallest_eig, a, b, n, d): (i, j)
                   for i, j, a, b in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing eigenvalues"):
            i, j = futures[future]
            try:
                result[i, j] = future.result()
            except Exception as e:
                print(f"Error at alpha index {i}, beta index {j}: {e}")

    # Plot heatmap: x-axis alpha, y-axis beta.
    plt.figure(figsize=(6, 5))
    plt.imshow(result.T, origin='lower',
               extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
               aspect='auto')
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('2nd Smallest Eigenvalue of Adjacency Matrix')
    plt.colorbar(label='Eigenvalue')
    plt.show()
