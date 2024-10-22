import numpy as np
from scipy.optimize import minimize

# Fixed parameters n_ij
n = 180  # Given value
n_ij = np.array([[4, 1, 1], [1, 4, 1], [1, 1, 4]]) * n // 18  # Example values such that for each i, sum_j n_ij = n / 3
assert all(np.sum(n_ij, axis=1) == n / 3), "Each row of n_ij must sum to n / 3"
assert all(np.sum(n_ij, axis=0) == n / 3), "Each column of n_ij must sum to n / 3"


# Define the objective function
def objective(p):
    p = p.reshape((3, 3))
    numerator = 0
    denominator = np.sum(n_ij * p)

    for i in range(3):
        for j in range(3):
            for i_prime in range(i + 1, 3):
                for j_prime in range(3):
                    if j_prime == j:
                        continue
                    numerator += n_ij[i, j] * n_ij[i_prime, j_prime] * (
                            p[i, j] + p[i_prime, j_prime] - 2 * p[i, j] * p[i_prime, j_prime])  # * 9 / (4 * n)

    if denominator == 0:
        return -np.inf  # Avoid division by zero
    return -(numerator / denominator)  # Negative for maximization


# Constraint: sum_{ij} p_{ij} * n_{ij} <= n / 2
def constraint(p):
    p = p.reshape((3, 3))
    return (np.sum(n_ij * p)) - (n / 2)


if __name__ == "__main__":

    # Bounds for p_ij: Each variable is between 0 and 1
    bounds = [(0, 1) for _ in range(9)]

    # Initial guess
    p0 = np.full((3, 3), 0.5).flatten()

    # Constraints in the form required by scipy.optimize
    cons = ({'type': 'ineq', 'fun': constraint},)

    # Run the optimization
    result = minimize(objective, p0, bounds=bounds, constraints=cons)

    # Display the results
    if result.success:
        optimized_p = result.x.reshape((3, 3))
        print("Optimized p:")
        print(optimized_p)
        print("Best achievable objective value:", -result.fun)
    else:
        print("Optimization failed:", result.message)
