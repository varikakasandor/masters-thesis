import numpy as np
from scipy.optimize import minimize

# Define the 6 possible permutations of (1, 2, 3)
permutations = [
    (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)
]

# Adjustable parameter
sum_squared_threshold = 0.9  # You can easily adjust this value


# Objective function - minimize c
def objective(x):
    c = x[-1]  # c is the last element in the vector
    return c


# Constraints
def constraint1(x):
    c = x[-1]
    w = x[:-1].reshape((3, 3))
    constraints = []

    # Sum of w_ij-s within each P_a cannot be more than c
    for p in permutations:
        P_a = [w[i - 1, p[i - 1] - 1] for i in range(1, 4)]
        constraints.append(c - sum(P_a))

    return np.array(constraints)


def constraint2(x):
    w = x[:-1].reshape((3, 3))
    sum_squared = 0.0

    # Sum over P_a of the squared sum of w_ij-s within that P_a
    for p in permutations:
        P_a = [w[i - 1, p[i - 1] - 1] for i in range(1, 4)]
        sum_squared += (sum(P_a)) ** 2

    return sum_squared - sum_squared_threshold


def constraint_row_sum(x):
    w = x[:-1].reshape((3, 3))
    constraints = []
    # Sum of the w_ij-s for each row must be exactly 1/3
    for i in range(3):
        constraints.append(np.sum(w[i, :]) - 1 / 3)
    return np.array(constraints)


def constraint_col_sum(x):
    w = x[:-1].reshape((3, 3))
    constraints = []
    # Sum of the w_ij-s for each column must be exactly 1/3
    for j in range(3):
        constraints.append(np.sum(w[:, j]) - 1 / 3)
    return np.array(constraints)


if __name__ == "__main__":
    # Bounds for w_ij-s and c
    bounds = [(0, 1) for _ in range(9)] + [(0, None)]  # 0 <= w_ij <= 1 and c >= 0

    # Improved initial guess closer to proposed values
    x0 = np.array([1 / 3, 0, 0, 0, 1 / 6, 1 / 6, 0, 1 / 6, 1 / 6, 2 / 3])  # Initial guess for w_ij-s and c

    # Define the constraints in a format suitable for minimize
    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    con_row_sum = {'type': 'eq', 'fun': constraint_row_sum}
    con_col_sum = {'type': 'eq', 'fun': constraint_col_sum}
    constraints = [con1, con2, con_row_sum, con_col_sum]

    # Solve the optimization problem with improved tolerance
    solution = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,
                        options={'xtol': 1e-12})

    # Extract the optimal values
    w_opt = solution.x[:-1].reshape((3, 3))
    c_opt = solution.x[-1]

    print("Optimal w_ij values:")
    print(w_opt)
    print("\nOptimal c value:")
    print(c_opt)
