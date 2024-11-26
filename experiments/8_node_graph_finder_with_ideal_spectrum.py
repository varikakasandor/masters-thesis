from tools import *
import numpy as np
from scipy.optimize import differential_evolution
from numpy.linalg import eig

# Desired eigenvalues
desired_eigenvalues = np.array([1, 1 / 6, 1 / 6, 1 / 6, 1 / 6, -1 / 2, -1 / 2, -1 / 2])


def create_symmetric_matrix(flat_upper):
    """
    Reconstruct a symmetric matrix from its upper triangular entries.
    """
    n = 8
    symmetric_matrix = np.zeros((n, n))
    upper_indices = np.triu_indices(n)
    symmetric_matrix[upper_indices] = flat_upper
    symmetric_matrix = symmetric_matrix + symmetric_matrix.T - np.diag(symmetric_matrix.diagonal())
    return symmetric_matrix


def objective(flat_upper):
    """
    Objective function to minimize the difference between the eigenvalues
    of the candidate matrix and the desired eigenvalues, while considering row sums.
    """
    # Reconstruct the symmetric matrix
    A = create_symmetric_matrix(flat_upper)

    # Compute eigenvalues
    eigenvalues, _ = eig(A)

    # Sort eigenvalues by real part for comparison
    eigenvalues_sorted = np.sort(np.real(eigenvalues))
    desired_sorted = np.sort(np.real(desired_eigenvalues))

    # Compute the eigenvalue mismatch
    eigenvalue_diff = np.sum((eigenvalues_sorted - desired_sorted) ** 2)

    # Compute the row sum penalty
    row_sums = np.sum(A, axis=1)
    row_sum_penalty = np.sum((row_sums - 1) ** 2)  # Penalize deviation from sum 1

    # Sparsity-promoting penalty (negative sum of squares)
    sparsity_penalty = 8 - (flat_upper ** 3).sum()

    # Total objective: eigenvalue mismatch + row sum penalty + sparsity penalty
    return eigenvalue_diff + 100 * row_sum_penalty + (1 / 10) * sparsity_penalty


if __name__ == "__main__":

    # Define bounds for each independent variable (all entries > 0)
    bounds = [(0, 1) for _ in range(36)]  # Avoid zero by setting a small lower bound

    # Differential Evolution parameters
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=30000,
        popsize=15,
        tol=1e-20,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=True,
        workers=-1  # Use all available CPU cores
    )

    # Retrieve the optimized upper triangular part
    optimized_upper = result.x
    # Reconstruct the symmetric matrix
    optimized_matrix = create_symmetric_matrix(optimized_upper)
    optimized_matrix = adjust_row_sums_to_exactly_one(optimized_matrix)

    # Compute its eigenvalues
    optimized_eigenvalues, _ = eig(optimized_matrix)

    # Sort for comparison
    optimized_eigenvalues_sorted = np.sort(np.real(optimized_eigenvalues))
    desired_sorted = np.sort(np.real(desired_eigenvalues))

    print("Optimized Symmetric Matrix:")
    for row in optimized_matrix:
        print(["{:.5f}".format(val) for val in row])

    print("\nDesired Eigenvalues:")
    print(desired_sorted)

    print("\nOptimized Eigenvalues:")
    print(optimized_eigenvalues_sorted)

    # Compute and print the difference
    difference = optimized_eigenvalues_sorted - desired_sorted
    print("\nDifference between Optimized and Desired Eigenvalues:")
    print(difference)

    # Check row sums
    print("\nRow Sums:")
    print(np.sum(optimized_matrix, axis=1))
