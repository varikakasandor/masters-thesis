import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

if __name__ == "__main__":

    num_matrices = 100
    std_values = np.linspace(0.0001, 0.1, 100)
    n_values = [3, 5, 10, 20]
    slopes = {}

    for n in n_values:
        target_vec = np.sqrt(1 / n) * np.ones(n)
        J = np.ones((n, n))
        I = np.eye(n)
        avg_deviations = []

        for std in std_values:
            deviations = []
            for _ in range(num_matrices):
                E = np.random.normal(0, std, (n, n))
                A = (J - I) + E
                eigvals, eigvecs = np.linalg.eig(A)
                principal_index = np.argmax(np.real(eigvals))
                principal_eigvec = np.real(eigvecs[:, principal_index])
                unit_vec = np.abs(principal_eigvec / np.linalg.norm(principal_eigvec))
                deviation = np.linalg.norm(unit_vec - target_vec)
                deviations.append(deviation)
            avg_deviations.append(np.mean(deviations))

        slope, _, _, _, _ = linregress(std_values, avg_deviations)
        slopes[n] = slope
        plt.plot(std_values, avg_deviations, label=f"n={n}")

    plt.xlabel("std (standard deviation of noise)")
    plt.ylabel("Average L2 deviation from target vector")
    plt.title("Stability of principal eigenvector under noise")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(slopes)
