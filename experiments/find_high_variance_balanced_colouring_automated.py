import numpy as np
from scipy.optimize import differential_evolution
from numpy.linalg import eig


def reconstruct_symmetric_matrix_excluding_blocks(flat_upper, k):
    """Reconstruct a symmetric matrix while excluding kxk diagonal blocks."""
    n = 3 * k
    A = np.zeros((n, n))
    upper_indices = []

    for i in range(n):
        for j in range(i, n):
            if i // k != j // k:  # Exclude kxk diagonal blocks
                upper_indices.append((i, j))

    for idx, (i, j) in enumerate(upper_indices):
        A[i, j] = flat_upper[idx]
        A[j, i] = flat_upper[idx]  # Ensure symmetry

    return A


def objective(flat_upper, k, gamma, epsilon):
    """Objective function to optimize the matrix under constraints."""
    n = 3 * k
    A = reconstruct_symmetric_matrix_excluding_blocks(flat_upper, k)

    # Compute eigenvalues
    eigenvalues = np.sort(np.real(eig(A)[0]))[::-1]

    # Constraints
    lambda_2_diff = max(0, eigenvalues[1] - gamma)  # Penalize if lambda_2 > gamma
    sum_constraint = max(0, np.sum(A[1, k:2 * k]) - (np.sum(A[0, k:2 * k]) - epsilon))

    # Row sum constraints (penalize deviation from row sums being 1)
    row_sum_constraint = np.sum((np.sum(A, axis=1) - 1) ** 2)

    # Total objective: penalize deviations
    return 0.95 * row_sum_constraint + 0.001 * sum_constraint + (1 - 0.95 - 0.001) * lambda_2_diff


def optimize_matrix(k, gamma, epsilon):
    """Optimize the 3k x 3k matrix."""
    n = 3 * k
    # Count variables excluding kxk diagonal blocks
    num_vars = 0
    for i in range(n):
        for j in range(i, n):
            if i // k != j // k:
                num_vars += 1

    bounds = [(0, 1) for _ in range(num_vars)]

    result = differential_evolution(
        objective,
        bounds,
        args=(k, gamma, epsilon),
        strategy='best1bin',
        maxiter=20000,
        popsize=15,
        tol=1e-20,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=True,
        workers=-1,
    )

    A = reconstruct_symmetric_matrix_excluding_blocks(result.x, k)
    return A


def analyze_matrix(A, k):
    """Analyze the optimized matrix."""
    eigenvalues = np.sort(np.real(eig(A)[0]))[::-1]
    print("Optimized Matrix:")
    print(A)

    print("\nEigenvalues:")
    print(eigenvalues)

    print("\nRow Sums:")
    print(np.sum(A, axis=1))

    print("\nColumn Sums:")
    print(np.sum(A, axis=0))

    print("\nBlock sum constraints:")
    print(np.sum(A[0, k:2 * k]), np.sum(A[1, k:2 * k]), max(0, np.sum(A[1, k:2 * k]) - (np.sum(A[0, k:2 * k]) - epsilon)))


if __name__ == "__main__":
    k = 4
    gamma = 0.0
    epsilon = 0.2

    optimized_matrix = optimize_matrix(k, gamma, epsilon)
    analyze_matrix(optimized_matrix, k)
