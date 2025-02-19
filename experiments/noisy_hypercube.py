import numpy as np
import matplotlib.pyplot as plt
from math import comb, floor
from tqdm import tqdm


# We'll define a function to compute all distinct eigenvalues and their multiplicities
# for the gamma-noisy hypercube with n and gamma.

def gamma_noisy_eigenvalues(n, gamma):
    """
    Returns a list of (lambda_r, multiplicity) for r = 0..n.
    lambda_r is the eigenvalue for all subsets S of size r.
    multiplicity = comb(n, r).
    """
    max_k = int(floor(gamma * n))
    values = []
    for r in tqdm(range(n + 1), desc="Computing eigenvalues"):
        lambda_r = 0
        for k in range(max_k + 1):
            ssum = 0
            for j in range(k + 1):
                if j <= r and (k - j) <= (n - r):
                    ssum += ((-1) ** j) * comb(r, j) * comb(n - r, k - j)
            lambda_r += ssum
        values.append((lambda_r, comb(n, r)))

    return values


def plot_eigenvalue_distribution(n, gamma):
    # get the (lambda, multiplicity) pairs
    vals = gamma_noisy_eigenvalues(n, gamma)

    unique_eigs, counts = zip(*vals)
    sorted_eigs = sorted(unique_eigs, reverse=True)
    second_max_eigenvalue = sorted_eigs[1]

    plt.figure(figsize=(8, 5))
    plt.hist(unique_eigs, bins=50, weights=counts, alpha=0.7, edgecolor='k')
    plt.title(f"Eigenvalue distribution for n={n}, gamma={gamma}")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")
    plt.tight_layout()
    d = sum(comb(n, k) for k in range(floor(gamma * n) + 1))
    plt.axvline(x=-d, color='red', linestyle='--', linewidth=2, label=f'-d = {d}')
    plt.axvline(x=second_max_eigenvalue, color='blue', linestyle='--', linewidth=2, label=f'\lambda_2 = {second_max_eigenvalue}')
    plt.legend()
    plt.show()

    # Count eigenvalues smaller than -d/3
    num_small_eigs = sum(count for eig, count in vals if eig < -d / 3)
    print(f"Number of eigenvalues smaller than -d/3: {num_small_eigs}")


if __name__ == "__main__":
    # Example usage
    n = 100
    gamma = 9 / 100
    plot_eigenvalue_distribution(n, gamma)