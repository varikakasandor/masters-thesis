import numpy as np
from scipy.optimize import minimize


# Define the objective function
def objective(p):
    p = p.reshape((3, 3))
    numerator = 0
    denominator = np.sum(p)

    for i in range(3):
        for j in range(3):
            for i_prime in range(i+1, 3):
                if i_prime == i:
                    continue
                for j_prime in range(3):
                    if j_prime == j:
                        continue
                    numerator += p[i, j] + p[i_prime, j_prime] - 2 * p[i, j] * p[i_prime, j_prime]

    if denominator == 0:
        return -np.inf  # Avoid division by zero
    return -(numerator / denominator)  # Negative for maximization


# Constraint: (sum_{ij} p_{ij}) / 9 <= 1/2
def constraint(p):
    return (np.sum(p) / 9) - 0.5


if __name__ == "__main__":

    # Bounds for p_ij: Each variable is between 0 and 1
    bounds = [(0, 1) for _ in range(9)]

    # Initial guess
    p0 = np.full(9, 0.5)

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
