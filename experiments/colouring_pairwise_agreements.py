from itertools import permutations
import cvxpy as cp
import matplotlib.pyplot as plt


def check_solution(n, good_agreement_threshold, bad_agreement_threshold):
    """
    This function sets up the problem for given GOOD_AGREEMENT_THRESHOLD and BAD_AGREEMENT_THRESHOLD
    and checks if a solution exists.
    """
    v = [cp.Variable((3, 3), nonneg=True) for _ in range(3)]
    constraints = []

    # Constraints for balancedness
    for i in range(3):
        constraints.append(sum(cp.sum(v[k][i, :]) for k in range(3)) == n // 3)
    for j in range(3):
        constraints.append(sum(cp.sum(v[k][:, j]) for k in range(3)) == n // 3)
    for k in range(3):
        constraints.append(cp.sum(v[k]) == n // 3)

    # Constraints for good agreement with Colouring 1
    constraints.append(cp.sum(v[0][0, :] + v[1][1, :] + v[2][2, :]) >= good_agreement_threshold)
    constraints.append(cp.sum(v[0][:, 0] + v[1][:, 1] + v[2][:, 2]) >= good_agreement_threshold)

    # Constraints for bad agreement between Colourings 0 and 2
    for perm in permutations(list(range(3))):
        constraints.append(
            sum((v[k][0, perm[0]] + v[k][1, perm[1]] + v[k][2, perm[2]]) for k in range(3)) <= bad_agreement_threshold)

    # Solve the problem
    objective = cp.Minimize(0)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    # Return whether a solution exists
    return prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE


def find_min_bad_agreement(n, good_agreement_threshold):
    """
    This function finds the smallest BAD_AGREEMENT_THRESHOLD such that a solution exists for a given
    GOOD_AGREEMENT_THRESHOLD using binary search for optimization.
    """
    low = n // 3
    high = n
    best_bad_agreement_threshold = None

    # Perform binary search
    while low <= high:
        mid = (low + high) // 2
        if check_solution(n, good_agreement_threshold, mid):
            best_bad_agreement_threshold = mid  # Solution found, try for a smaller BAD_AGREEMENT_THRESHOLD
            high = mid - 1
        else:
            low = mid + 1

    return best_bad_agreement_threshold


def analyse_transitivity(n):
    good_agreement_thresholds = []
    bad_agreement_thresholds = []

    for GOOD_AGREEMENT_THRESHOLD in range(n // 3, n):
        min_bad_agreement = find_min_bad_agreement(n, GOOD_AGREEMENT_THRESHOLD)
        good_agreement_thresholds.append(GOOD_AGREEMENT_THRESHOLD)
        bad_agreement_thresholds.append(min_bad_agreement)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(good_agreement_thresholds, bad_agreement_thresholds, marker='o', linestyle='-', color='b')

    # Add title and labels
    plt.title(f"Minimum BAD_AGREEMENT_THRESHOLD vs GOOD_AGREEMENT_THRESHOLD for N = {n}")
    plt.xlabel("GOOD_AGREEMENT_THRESHOLD")
    plt.ylabel("Minimum BAD_AGREEMENT_THRESHOLD")
    plt.grid(True)
    plt.savefig("agreement_transitivity_analysis.png")
    plt.show()


if __name__ == "__main__":
    analyse_transitivity(99)
