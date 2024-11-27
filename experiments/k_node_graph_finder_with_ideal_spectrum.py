from tools import *
import numpy as np
from scipy.optimize import differential_evolution
from numpy.linalg import eig


def create_desired_eigenvalues(k):
    if k < 4:
        raise ValueError("k must be at least 4 for this construction.")

    # Calculate the value of x such that the sum of the array is 0
    x = 0.5 / (k - 4)

    # Create the array with the specified structure
    arr = np.ones(k) * x
    arr[0] = 1
    arr[-3:] = -0.5

    return arr


def create_symmetric_matrix(flat_upper, k, make_k_colourable=False):
    """
    Reconstruct a symmetric matrix from its upper triangular entries.
    If make_k_colourable is True, all diagonal values are fixed to be 0.
    """
    symmetric_matrix = np.zeros((k, k))
    upper_indices = np.triu_indices(k, k=1) if make_k_colourable else np.triu_indices(k)
    symmetric_matrix[upper_indices] = flat_upper
    symmetric_matrix = symmetric_matrix + symmetric_matrix.T

    if make_k_colourable:
        np.fill_diagonal(symmetric_matrix, 0)

    return symmetric_matrix


def objective(flat_upper, desired_eigenvalues, k, make_k_colourable=False):
    """
    Objective function to minimize the difference between the eigenvalues
    of the candidate matrix and the desired eigenvalues, while considering row sums.
    """
    # Reconstruct the symmetric matrix
    A = create_symmetric_matrix(flat_upper, k, make_k_colourable=make_k_colourable)

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
    sparsity_penalty = k - (flat_upper ** 3).sum()

    # Total objective: eigenvalue mismatch + row sum penalty + sparsity penalty
    return 0.1 * (eigenvalue_diff / k) + 0.9 * (row_sum_penalty / k) + 0.0 * (sparsity_penalty / k)


def optimize_symmetric_matrix(desired_eigenvalues, k, make_k_colourable):
    """
    Optimize the symmetric matrix using differential evolution.
    """
    # Define bounds for each independent variable (all entries > 0)
    num_upper_elements = (k * (k + 1)) // 2 if not make_k_colourable else (k * (k - 1)) // 2
    bounds = [(0, 1) for _ in range(num_upper_elements)]

    # Differential Evolution parameters
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',  # 'rand1bin',
        maxiter=5000,
        popsize=15,
        tol=1e-20,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=False,  # True,
        workers=-1,  # Use all available CPU cores
        args=(desired_eigenvalues, k, make_k_colourable)
        # Pass the desired eigenvalues, k, and make_k_colourable flag to the objective function
    )

    # Retrieve the optimized upper triangular part
    optimized_upper = result.x
    # Reconstruct the symmetric matrix
    optimized_matrix = create_symmetric_matrix(optimized_upper, k, make_k_colourable=make_k_colourable)
    optimized_matrix = adjust_row_sums_to_exactly_one(optimized_matrix, max_iter=1000)

    return optimized_matrix


def analyze_optimized_matrix(optimized_matrix, desired_eigenvalues):
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

    k = 14  # You can change k to any value greater than or equal to 4
    # Desired eigenvalues as input
    desired_eigenvalues = create_desired_eigenvalues(k)

    # Optimize the symmetric matrix
    optimized_matrix = optimize_symmetric_matrix(desired_eigenvalues, k, make_k_colourable)

    # Analyze the optimized matrix
    analyze_optimized_matrix(optimized_matrix, desired_eigenvalues)
