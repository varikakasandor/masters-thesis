from itertools import permutations
import cvxpy as cp

if __name__ == "__main__":
    # Define 3 matrices to represent the 3rd axis (k-axis)
    N = 99
    assert N % 2 == 1
    v = [cp.Variable((3, 3), nonneg=True) for _ in range(3)]
    constraints = []

    # Constraints for balancedness
    for i in range(3):
        constraints.append(sum(cp.sum(v[k][i, :]) for k in range(3)) == N // 3)
    for j in range(3):
        constraints.append(sum(cp.sum(v[k][:, j]) for k in range(3)) == N // 3)
    for k in range(3):
        constraints.append(cp.sum(v[k]) == N // 3)

    # Constraints for good agreement with Colouring 1
    constraints.append(cp.sum(v[0][0, 0] + v[1][1, 1] + v[2][2, 2]) >= (N // 2) + 1)
    constraints.append(cp.sum(v[0][0, 0] + v[1][1, 1] + v[2][2, 2]) >= (N // 2) + 1)

    # Constraints for bad agreement between Colourings 0 and 2
    for perm in permutations(list(range(3))):
        constraints.append(cp.sum(v[0][:, perm[0]] + v[1][:, perm[1]] + v[2][:, perm[2]]) <= N // 2)

    objective = cp.Minimize(0) # cp.Minimize(sum(cp.sum_squares(v[k]) for k in range(3))) #
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # Check if a solution exists
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        print("A solution exists!")
        # Access the solution
        for k in range(3):
            print(f"Solution for v[{k}]:")
            print(v[k].value)
    else:
        print("No solution exists.")

    """
    v = cp.Variable((3, 3, 3), nonneg=True)
    constraints = []

    # Constraints for balancedness
    for i in range(3):
        constraints.append(cp.sum(v[i, :, :]) == N // 3)
    for j in range(3):
        constraints.append(cp.sum(v[:, j, :]) == N // 3)
    for k in range(3):
        constraints.append(cp.sum(v[:, :, k]) == N // 3)

    # Constraints for good agreement with Colouring 1
    constraints.append(v[0, 0, :] + v[1, 1, :] + v[2, 2, :] >= (N // 2) + 1)
    constraints.append(v[:, 0, 0] + v[:, 1, 1] + v[:, 2, 2] >= (N // 2) + 1)

    # Constraints for bad agreement between Colourings 0 and 2
    for perm in permutations(list(range(3))):
        constraints.append(v[0, :, perm[0]] + v[1, :, perm[1]] + v[2, :, perm[2]] < (N // 2) + 1)


    objective = cp.Minimize(0)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # Check if a solution exists
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        print("A solution exists!")
        # Access the solution
        solution = v.value
        print("Solution v:")
        print(solution)
    else:
        print("No solution exists.") 
    """
