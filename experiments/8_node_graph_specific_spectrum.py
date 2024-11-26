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

    # Total objective: eigenvalue mismatch + row sum penalty
    return eigenvalue_diff + 10 * row_sum_penalty  # Adjust weight for penalty if needed


if __name__ == "__main__":

    # Define bounds for each independent variable (all entries > 0)
    bounds = [(1e-3, 10) for _ in range(36)]  # Avoid zero by setting a small lower bound

    # Differential Evolution parameters
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=3000,
        popsize=15,
        tol=1e-1,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=True,
        workers=-1  # Use all available CPU cores
    )

    # Check if the optimization was successful
    if True:
        # Retrieve the optimized upper triangular part
        optimized_upper = result.x

        # Reconstruct the symmetric matrix
        optimized_matrix = create_symmetric_matrix(optimized_upper)

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
    else:
        print("Optimization failed. Try adjusting the parameters or providing a better initial guess.")
