import numpy as np


def check_validity(A, sizes, d):
    A = np.array(A)
    sizes = np.array(sizes)
    k = len(sizes)

    # 1. Check that each row sum equals d
    row_sums = A.sum(axis=1)
    print("Row sums:", row_sums)
    if np.allclose(row_sums, d):
        print("✅ All row sums equal d =", d)
    else:
        print("❌ Some row sums differ from d =", d)

    # 2. Check weighted symmetry: |V_i|*A[i,j] == |V_j|*A[j,i] for all i != j
    symmetry_ok = True
    for i in range(k):
        for j in range(k):
            if i != j:
                lhs = sizes[i] * A[i, j]
                rhs = sizes[j] * A[j, i]
                if not np.isclose(lhs, rhs):
                    print(f"❌ Symmetry fails for indices {i} and {j}: {lhs} != {rhs}")
                    symmetry_ok = False
    if symmetry_ok:
        print("✅ Weighted symmetry holds for all i ≠ j.")

    # 3. Form the centered matrix and check for non-real eigenvalues
    # For a k-coloring, d/(k-1) is the offset
    d_offset = d / (k - 1)
    J = np.ones((k, k))
    I = np.eye(k)
    A_centered = A - d_offset * (J - I)
    print("Centered matrix (A_centered):\n", A_centered)

    # Compute eigenvalues
    eigvals = np.linalg.eigvals(A_centered)
    print("Eigenvalues of the centered matrix:", eigvals)

    # Check if any eigenvalue is non-real (allowing for tiny numerical imaginary parts)
    nonreal = any(abs(ev.imag) > 1e-10 for ev in eigvals)
    if nonreal:
        print("✅ The centered matrix has at least one non-real eigenvalue.")
    else:
        print("❌ All eigenvalues are real.")


if __name__ == "__main__":
    # Candidate matrix:
    A = [
        [0, 0.5, 4, 1.5],
        [0.5, 0, 1.5, 4],
        [2, 0.75, 0, 3.25],
        [0.75, 2, 3.25, 0]
    ]
    sizes = [1, 1, 2, 2]
    d = 6

    check_validity(A, sizes, d)
