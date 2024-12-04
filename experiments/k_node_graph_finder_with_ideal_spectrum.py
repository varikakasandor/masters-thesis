from tools import *
import numpy as np
from scipy.optimize import differential_evolution
from numpy.linalg import eig


def create_desired_eigenvalues(k, gamma, tr, threshold):
    # Create the array with the specified structure
    arr = np.zeros(k)
    arr[0] = 1
    arr[1] = gamma
    arr[-tr:] = -threshold

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


def objective(flat_upper, desired_eigenvalues, k, tr, make_k_colourable=False):
    """
    Objective function to minimize the difference between the eigenvalues
    of the candidate matrix and the desired eigenvalues, while considering row sums.
    """
    # Reconstruct the symmetric matrix
    A = create_symmetric_matrix(flat_upper, k, make_k_colourable=make_k_colourable)

    # Compute eigenvalues
    eigenvalues, _ = eig(A)

    # Sort eigenvalues descending
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    desired_sorted = np.sort(desired_eigenvalues)[::-1]

    # Eigenvalue conditions
    eigenvalue_diff_2 = max(0, eigenvalues_sorted[1] - desired_sorted[1]) ** 2  # Second eigenvalue should match desired
    eigenvalue_diff_tr = np.sum((eigenvalues_sorted[-tr:] - desired_sorted[
                                                            -tr:]) ** 2)  # np.sum(np.maximum(0, eigenvalues_sorted[-tr:] - desired_sorted[-tr:]) ** 2)  # Last "tr" eigenvalues should match desired
    eigenvalue_diff_penalty = (4 / 6) * eigenvalue_diff_2 + (2 / 6) * eigenvalue_diff_tr

    # Compute the row sum penalty
    row_sums = np.sum(A, axis=1)
    row_sum_penalty = (np.sum((row_sums - 1) ** 2))  # Penalize deviation from sum 1

    # Sparsity-promoting penalty
    sparsity_penalty = (k / (tr ** (3 - 1)) - (
                flat_upper ** 3).sum()) / k  # the ideal would be to have tr entries of 1/tr in each row

    # Total objective: eigenvalue mismatch + row sum penalty + sparsity penalty
    return 0.35 * eigenvalue_diff_penalty + 0.64 * row_sum_penalty + 0.01 * sparsity_penalty


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
        tol=1e-4,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=False,  # True,
        workers=-1,  # Use all available CPU cores
        args=(desired_eigenvalues, k, tr, make_k_colourable)
        # Pass the desired eigenvalues, k, and make_k_colourable flag to the objective function
    )

    # Retrieve the optimized upper triangular part
    optimized_upper = result.x
    # Reconstruct the symmetric matrix
    optimized_matrix = create_symmetric_matrix(optimized_upper, k, make_k_colourable=make_k_colourable)
    optimized_matrix = adjust_row_sums_to_exactly_one(optimized_matrix, max_iter=1000)

    return optimized_matrix


def analyze_optimized_matrix(optimized_matrix, desired_eigenvalues, tr):
    """
    Analyze the optimized matrix by computing eigenvalues, printing the matrix, and checking row sums.
    """
    # Compute its eigenvalues and eigenvectors
    optimized_eigenvalues, eigenvectors = eig(optimized_matrix)

    # Sort for comparison
    optimized_eigenvalues_sorted_indices = np.argsort(np.real(optimized_eigenvalues))[::-1]
    optimized_eigenvalues_sorted = np.real(optimized_eigenvalues[optimized_eigenvalues_sorted_indices])
    desired_sorted = np.sort(np.real(desired_eigenvalues))[::-1]

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

    # Extract and print the eigenvectors corresponding to the tr lowest eigenvalues
    print("\nEigenvectors corresponding to the tr lowest eigenvalues:")
    for i in range(1, tr + 1):
        idx = optimized_eigenvalues_sorted_indices[-i]
        print(f"Eigenvector for eigenvalue {optimized_eigenvalues[idx]:.5f}:")
        print(eigenvectors[:, idx])

    # Extract and print the eigenvector corresponding to the second largest eigenvalue
    second_eigenvalue_index = optimized_eigenvalues_sorted_indices[1]
    print("\nEigenvector corresponding to the second largest eigenvalue:")
    print(eigenvectors[:, second_eigenvalue_index])


# Modify analyze_optimized_matrix to save the optimized matrix as a CSV
def save_optimized_matrix(optimized_matrix, filename):
    np.savetxt(filename, optimized_matrix, delimiter=",", fmt="%.5f")


if __name__ == "__main__":
    # Boolean flag to fix diagonal values to be 0
    make_k_colourable = True

    k = 12  # You can change k to any value greater than or equal to 4
    gamma = 0.1  # Second eigenvalue should be at most gamma
    tr = 3  # Number of trailing eigenvalues to analyze
    threshold = 0.52  # Absolute value of the threshold for the last "tr" eigenvalues
    assert 1 * 1.0 - tr * threshold + (
                k - 1 - tr) * gamma >= 0  # as the trace of the graph is 0, the sum of eigenvalues has to be 0

    # Desired eigenvalues as input
    desired_eigenvalues = create_desired_eigenvalues(k, gamma, tr, threshold)

    # Optimize the symmetric matrix
    optimized_matrix = optimize_symmetric_matrix(desired_eigenvalues, k, make_k_colourable)

    # Save the optimized matrix to a CSV file
    core_filename = f"k{k}_gamma{gamma}_tr{tr}_threshold{threshold}"
    matrix_filename = f"./runs/optimized_matrix_{core_filename}.csv"
    save_optimized_matrix(optimized_matrix, matrix_filename)

    # Analyze the optimized matrix
    analyze_optimized_matrix(optimized_matrix, desired_eigenvalues, tr)

    # Plot and save the density matrix graph
    plot_filename = f"./runs/density_matrix_plot_{core_filename}.png"
    plot_density_matrix(optimized_matrix, plot_filename)