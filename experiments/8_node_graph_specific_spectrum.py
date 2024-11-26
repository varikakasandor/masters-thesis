import numpy as np
from scipy.optimize import differential_evolution
from numpy.linalg import eig

# Desired eigenvalues
desired_eigenvalues = np.array([1, 1 / 6, 1 / 6, 1 / 6, 1 / 6, -1 / 2, -1 / 2, -1 / 2])


def objective(flat_matrix):
    """
    Objective function to minimize the difference between the eigenvalues
    of the candidate matrix and the desired eigenvalues.
    """
    # Reshape the flat array into an 8x8 matrix
    A = flat_matrix.reshape((8, 8))

    # Compute eigenvalues
    eigenvalues, _ = eig(A)

    # Sort eigenvalues by real part for comparison
    eigenvalues_sorted = np.sort(np.real(eigenvalues))
    desired_sorted = np.sort(np.real(desired_eigenvalues))

    # Compute the sum of squared differences
    # Handle possible multiplicities and ordering
    diff = eigenvalues_sorted - desired_sorted
    return np.sum(diff ** 2)


def constraint_positive(flat_matrix):
    """
    Constraint to ensure all matrix entries are positive.
    """
    return flat_matrix


if __name__ == "__main__":

    # Define bounds for each matrix entry (all entries > 0)
    bounds = [(1e-3, 10) for _ in range(64)]  # Avoid zero by setting a small lower bound

    # Differential Evolution parameters
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=6000,
        popsize=15,
        tol=1e-3,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True,
        polish=True,
        workers=-1  # Use all available CPU cores
    )

    # Check if the optimization was successful
    if True:
        # Retrieve the optimized matrix
        optimized_matrix = result.x.reshape((8, 8))

        # Compute its eigenvalues
        optimized_eigenvalues, _ = eig(optimized_matrix)

        # Sort for comparison
        optimized_eigenvalues_sorted = np.sort(np.real(optimized_eigenvalues))
        desired_sorted = np.sort(np.real(desired_eigenvalues))

        print("Optimized Matrix:")
        print(optimized_matrix)

        print("\nDesired Eigenvalues:")
        print(desired_sorted)

        print("\nOptimized Eigenvalues:")
        print(optimized_eigenvalues_sorted)

        # Compute and print the difference
        difference = optimized_eigenvalues_sorted - desired_sorted
        print("\nDifference between Optimized and Desired Eigenvalues:")
        print(difference)
    else:
        print("Optimization failed. Try adjusting the parameters or providing a better initial guess.")
