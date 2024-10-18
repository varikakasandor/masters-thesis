from itertools import permutations
import cvxpy as cp

if __name__ == "__main__":
    N = 99
    assert N % 2 == 1
    assert N % 3 == 0

    GOOD_AGREEMENT_THRESHOLD = (N // 2) + 1
    BAD_AGREEMENT_THRESHOLD = N // 2  # Even N // 3 is possible, so a shared similar colouring enforces nothing

    v = [cp.Variable((3, 3), nonneg=True) for _ in range(3)]  # shared x, y, z
    constraints = []

    # Constraints for balancedness
    for i in range(3):
        constraints.append(sum(cp.sum(v[k][i, :]) for k in range(3)) == N // 3)
    for j in range(3):
        constraints.append(sum(cp.sum(v[k][:, j]) for k in range(3)) == N // 3)
    for k in range(3):
        constraints.append(cp.sum(v[k]) == N // 3)

    # Constraints for good agreement with Colouring 1
    constraints.append(cp.sum(v[0][0, :] + v[1][1, :] + v[2][2, :]) >= GOOD_AGREEMENT_THRESHOLD)
    constraints.append(cp.sum(v[0][:, 0] + v[1][:, 1] + v[2][:, 2]) >= GOOD_AGREEMENT_THRESHOLD)

    # Constraints for bad agreement between Colourings 0 and 2
    for perm in permutations(list(range(3))):
        constraints.append(
            sum((v[k][0, perm[0]] + v[k][1, perm[1]] + v[k][2, perm[2]]) for k in range(3)) <= BAD_AGREEMENT_THRESHOLD)

    objective = cp.Minimize(0)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # Check if a solution exists
    if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
        # Iterate through the 1st axis (the middle dimension) and print each slice
        for j in range(3):
            print(f"\nWhen the shared coulouring has colour {j + 1}:")
            # Print the solution for the slice corresponding to this j, truncated to 2 decimals
            for i in range(3):
                row = [float(round(v[k].value[i, j], 2)) for k in range(3)]
                print(row)
    else:
        print("No solution exists.")
