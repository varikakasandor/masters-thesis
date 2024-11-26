import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.linalg import eigh

# Desired eigenvalues
eigenvalues = np.array([1, 1 / 6, 1 / 6, 1 / 6, -1 / 2, -1 / 2, -1 / 2])

# Initial diagonal matrix
D = np.diag(eigenvalues)


# Objective: minimize the sum of squares of the diagonal entries of A = Q D Q^T
def objective(Q_flat):
    Q = Q_flat.reshape((7, 7))
    # Ensure Q is orthogonal: Q Q^T = I
    orthogonality = np.dot(Q, Q.T) - np.eye(7)
    orthogonality_norm = np.linalg.norm(orthogonality)
    # Compute A
    A = Q @ D @ Q.T
    # Sum of squares of diagonal entries
    diag_norm = np.sum(A.diagonal() ** 2)
    # Penalize deviation from orthogonality
    return diag_norm + 1000 * orthogonality_norm ** 2


if __name__ == "__main__":
    # Step 1: Define the target eigenvalues
    target_eigenvalues = np.array([1, 1 / 6, 1 / 6, 1 / 6, -1 / 2, -1 / 2, -1 / 2])

    # Step 2: Generate random orthogonal matrix P using QR decomposition
    np.random.seed(42)
    random_matrix = np.random.randn(7, 7)
    Q, _ = np.linalg.qr(random_matrix)  # Q is an orthogonal matrix

    # Step 3: Construct the diagonal matrix D with the target eigenvalues
    D = np.diag(target_eigenvalues)

    # Step 4: Construct A = Q * D * Q^T
    A = Q @ D @ Q.T

    # Step 5: Set diagonal entries of A to zero to enforce the condition
    np.fill_diagonal(A, 0)

    # Step 6: Verify the eigenvalues
    computed_eigenvalues, _ = eigh(A)

    print("Target Eigenvalues:", target_eigenvalues)
    print("Computed Eigenvalues:", computed_eigenvalues)
    print("Final Diagonal Entries:", A.diagonal())

    """
    # Step 1: Create an initial symmetric matrix with zero diagonal
    np.random.seed(42)  # For reproducibility
    A = np.random.uniform(-1, 1, (7, 7))
    A = (A + A.T) / 2  # Make symmetric
    np.fill_diagonal(A, 0)  # Set diagonal elements to zero

    # Step 2: Define the target eigenvalues
    target_eigenvalues = np.array([1, 1 / 6, 1 / 6, 1 / 6, -1 / 2, -1 / 2, -1 / 2])

    # Step 3: Define an iteration to adjust the matrix to match the desired eigenvalues
    learning_rate = 0.01
    num_iterations = 10000

    for iteration in range(num_iterations):
        # Calculate current eigenvalues
        current_eigenvalues, _ = eigh(A)

        # Calculate the error
        error = current_eigenvalues - target_eigenvalues

        # Compute the gradient for each element
        grad = np.zeros_like(A)

        # Adjust off-diagonal elements to reduce the error
        for i in range(7):
            for j in range(i + 1, 7):
                A[i, j] -= learning_rate * error[i]  # Adjust based on the error of corresponding eigenvalue
                A[j, i] = A[i, j]  # Maintain symmetry

        # Project back to ensure diagonal elements remain zero
        np.fill_diagonal(A, 0)

        # Convergence check (optional)
        if np.linalg.norm(error) < 1e-6:
            break

    # Step 4: Verify the result
    computed_eigenvalues, _ = eigh(A)

    print("Target Eigenvalues:", target_eigenvalues)
    print("Computed Eigenvalues:", computed_eigenvalues)
    print("Final Diagonal Entries:", A.diagonal())
    """

    """
    # Step 1: Generate a random orthogonal matrix P using QR decomposition
    np.random.seed(42)  # For reproducibility
    random_matrix = np.random.rand(7, 7)
    Q, _ = qr(random_matrix)  # Q is an orthogonal matrix

    # Step 2: Define the diagonal matrix with the given eigenvalues
    eigenvalues = np.array([1, 1 / 6, 1 / 6, 1 / 6, -1 / 2, -1 / 2, -1 / 2])
    D = np.diag(eigenvalues)

    # Step 3: Construct the symmetric matrix A = Q D Q^T
    A = Q @ D @ Q.T

    # Step 4: Ensure the diagonal is zero by adjusting A
    A_no_diag = A - np.diag(np.diag(A))

    # Step 5: Verify the eigenvalues
    computed_eigenvalues, _ = np.linalg.eigh(A_no_diag)

    print("Desired Eigenvalues:", eigenvalues)
    print("Computed Eigenvalues:", computed_eigenvalues)
    print("Diagonal Entries of A_no_diag:", A_no_diag.diagonal())
    """

    """
    # Initial guess: identity matrix
    Q0 = np.eye(7).flatten()

    # Perform optimization
    result = minimize(objective, Q0, method='L-BFGS-B')

    # Extract optimized Q
    Q_opt = result.x.reshape((7, 7))

    # Compute A
    A_opt = Q_opt @ D @ Q_opt.T

    # Zero out diagonal (optional, since optimization aims to minimize it)
    A_opt_zero_diag = A_opt - np.diag(A_opt.diagonal())

    # Verify eigenvalues
    computed_eigenvalues, _ = eigh(A_opt_zero_diag)

    print("Desired Eigenvalues:", eigenvalues)
    print("Computed Eigenvalues:", computed_eigenvalues)
    print("Diagonal Entries:", A_opt_zero_diag.diagonal())
    """