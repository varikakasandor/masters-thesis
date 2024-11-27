from tools import *
import numpy as np
from scipy.optimize import differential_evolution
from numpy.linalg import eig

# Desired eigenvalues
desired_eigenvalues = np.array([1, 1 / 6, 1 / 6, 1 / 6, 1 / 6, -1 / 2, -1 / 2, -1 / 2])
desired_sorted = np.sort(np.real(desired_eigenvalues))


def create_symmetric_matrix(flat_upper, make_k_colourable=False):
    """
    Reconstruct a symmetric matrix from its upper triangular entries.
    If make_k_colourable is True, all diagonal values are fixed to be 0.
    """
    k = 8
    symmetric_matrix = np.zeros((k, k))
    upper_indices = np.triu_indices(k, k=1) if make_k_colourable else np.triu_indices(k)
    symmetric_matrix[upper_indices] = flat_upper
    symmetric_matrix = symmetric_matrix + symmetric_matrix.T

    if make_k_colourable:
        np.fill_diagonal(symmetric_matrix, 0)

    return symmetric_matrix


def objective(flat_upper, make_k_colourable=False):
    """
    Objective function to minimize the difference between the eigenvalues
    of the candidate matrix and the desired eigenvalues, while considering row sums.
    """
    # Reconstruct the symmetric matrix
    A = create_symmetric_matrix(flat_upper, make_k_colourable=make_k_colourable)

    # Compute eigenvalues
    eigenvalues, _ = eig(A)

    # Sort eigenvalues by real part for comparison
    eigenvalues_sorted = np.sort(np.real(eigenvalues))

    # Compute the eigenvalue mismatch
    eigenvalue_diff = np.sum((eigenvalues_sorted - desired_sorted) ** 2)

    # Compute the row sum penalty
    row_sums = np.sum(A, axis=1)
    row_sum_penalty = np.sum((row_sums - 1) ** 2)  # Penalize deviation from sum 1

    # Sparsity-promoting penalty (negative sum of squares)
    sparsity_penalty = 8 - (flat_upper ** 3).sum()

    # Total objective: eigenvalue mismatch + row sum penalty + sparsity penalty
    return eigenvalue_diff + 100 * row_sum_penalty + (1 / 10) * sparsity_penalty


def optimize_symmetric_matrix(make_k_colourable):
    """
    Optimize the symmetric matrix using differential evolution.
    """
    # Define bounds for each independent variable (all entries > 0)
    bounds = [(0, 1) for _ in range(28)] if make_k_colourable else [(0, 1) for _ in
                                                               range(36)]  # Avoid zero by setting a small lower bound

    # Differential Evolution parameters
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=5000,
        popsize=15,
        tol=1e-20,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=True,
        workers=-1,  # Use all available CPU cores
        args=(make_k_colourable,)  # Pass the make_k_colourable flag to the objective function
    )

    # Retrieve the optimized upper triangular part
    optimized_upper = result.x
    # Reconstruct the symmetric matrix
    optimized_matrix = create_symmetric_matrix(optimized_upper, make_k_colourable=make_k_colourable)
    optimized_matrix = adjust_row_sums_to_exactly_one(optimized_matrix)

    return optimized_matrix


def analyze_optimized_matrix(optimized_matrix):
    """
    Analyze the optimized matrix by computing eigenvalues, printing the matrix, and checking row sums.
    """
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


if __name__ == "__main__":
    # Boolean flag to fix diagonal values to be 0
    make_k_colourable = True

    # Optimize the symmetric matrix
    optimized_matrix = optimize_symmetric_matrix(make_k_colourable)

    # Analyze the optimized matrix
    analyze_optimized_matrix(optimized_matrix)
