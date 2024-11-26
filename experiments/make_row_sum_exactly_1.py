import numpy as np
np.set_printoptions(precision=8, suppress=True)


def adjust_row_sums_to_exactly_one(A):
    n = A.shape[0]
    row_sums = A.sum(axis=1)

    while not np.all(row_sums == 1):
        # Find the (i, j) with A[i, j] largest such that sum(A[i, :]) and sum(A[j, :]) are both not yet 1
        max_value = -np.inf
        i_max, j_max = -1, -1
        for i in range(n):
            if row_sums[i] != 1:
                for j in range(i + 1, n):
                    if (row_sums[j] - 1) * (row_sums[i] - 1) > 0 and A[i, j] > max_value: # i.e. neither of them are 1 and they are wrong from the same side
                        max_value = A[i, j]
                        i_max, j_max = i, j
        if i_max == -1 or j_max == -1:
            print("Cannot make more progress")
            break  # No valid (i, j) found, meaning all row sums are 1

        # Calculate how much to subtract from A[i_max, j_max] and A[j_max, i_max]
        subtract_amount = min(abs(row_sums[i_max] - 1), abs(row_sums[j_max] - 1), A[i_max, j_max])
        if row_sums[i_max] < 1:
            subtract_amount *= -1

        # Update A and row sums
        A[i_max, j_max] -= subtract_amount
        A[j_max, i_max] -= subtract_amount
        row_sums[i_max] -= subtract_amount
        row_sums[j_max] -= subtract_amount

    return A


if __name__ == "__main__":
    # Example usage
    A = np.array([
        [0, 0.02670, 0.19414, 0.06657, 0.27479, 0.34946, 0.05822, 0.03012],
        [0.02670, 0, 0.01004, 0.08708, 0.05447, 0.28525, 0.33164, 0.18201],
        [0.19414, 0.01004, 0, 0.30031, 0.07428, 0.06199, 0.29378, 0.01462],
        [0.06657, 0.08708, 0.30031, 0, 0.01096, 0.16554, 0.03216, 0.33735],
        [0.27479, 0.05447, 0.07428, 0.01096, 0, 0.01227, 0.20091, 0.29212],
        [0.34946, 0.28525, 0.06199, 0.16554, 0.01227, 0, 0.03250, 0.09299],
        [0.05822, 0.33164, 0.29378, 0.03216, 0.20091, 0.03250, 0, 0.05079],
        [0.03012, 0.18201, 0.01462, 0.33735, 0.29212, 0.09299, 0.05079, 0]
    ])

    result = adjust_row_sums_to_exactly_one(A)
    print(result.sum(1))
    print("Adjusted A:")
    print(result)
