import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Pool
from tools import analyse_spectrum, plot_degree_distribution_from_adjacency_matrix, analyse_2nd_largest_eigenvalue


def generate_d_regular_graph(n, d, alpha):
    # Sizes of the parts
    size_A = n // 3
    size_B = n // 3
    size_C = n // 6
    size_D = n // 6

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n))

    # Indices for parts
    A_indices = range(0, size_A)
    B_indices = range(size_A, size_A + size_B)
    C_indices = range(size_A + size_B, size_A + size_B + size_C)
    D_indices = range(size_A + size_B + size_C, n)

    # Edge probabilities
    prob_A_B = (d / 2) / size_A
    prob_A_C = (alpha * d) / size_A
    prob_A_D = ((1 - alpha) * d) / size_A
    prob_B_C = ((1 - alpha) * d) / size_B
    prob_B_D = (alpha * d) / size_B
    prob_C_D = 0

    # Add edges between parts with given probabilities
    for i in A_indices:
        for j in B_indices:
            if random.random() < prob_A_B:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in A_indices:
        for j in C_indices:
            if random.random() < prob_A_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in A_indices:
        for j in D_indices:
            if random.random() < prob_A_D:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in B_indices:
        for j in C_indices:
            if random.random() < prob_B_C:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    for i in B_indices:
        for j in D_indices:
            if random.random() < prob_B_D:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix


def process_alpha(args):
    n, d, alpha = args
    adjacency_matrix = generate_d_regular_graph(n, d, alpha)
    return analyse_2nd_largest_eigenvalue(adjacency_matrix)


def plot_alpha_vs_eigenvalue(n, d, num_grid_points):
    alphas = np.linspace(0.5, 0.75, num_grid_points)

    # Prepare arguments for parallel processing
    args = [(n, d, alpha) for alpha in alphas]

    with Pool() as pool:
        second_largest_eigenvalues = list(tqdm(pool.imap(process_alpha, args), total=len(alphas)))

    plt.plot(alphas, second_largest_eigenvalues, marker='o')
    plt.title(f"Relationship Between Alpha and Lambda_2 for n={n}, d={d}")
    plt.xlabel("Alpha")
    plt.ylabel("Normalised 2nd Largest Eigenvalue")
    plt.grid(True)

    # Save the plot
    plt.savefig(f"alpha_vs_eigenvalue_n{n}_d{d}.png")
    plt.show()


def process_single_d(args):
    d, n, num_grid_points = args
    alphas = np.linspace(1 / 2, 3 / 4, num_grid_points)
    second_largest_eigenvalues = []

    for alpha in alphas:
        adjacency_matrix = generate_d_regular_graph(n, d, alpha)
        second_largest = analyse_2nd_largest_eigenvalue(adjacency_matrix)
        second_largest_eigenvalues.append(second_largest)

    # Find the "elbow" where eigenvalue starts increasing significantly
    # diffs = np.diff(second_largest_eigenvalues)
    # elbow_index = np.argmax(diffs > 3e-3)  # Threshold to detect change
    slopes = np.diff(second_largest_eigenvalues) / np.diff(alphas)
    cutoff = slopes[-1] / 2
    elbow_index = np.argmax(slopes > cutoff)
    elbow_alpha = alphas[elbow_index] if elbow_index < len(alphas) else None
    return d, elbow_alpha


def find_elbow_points(n, num_grid_points, num_ds, d_min, d_max):
    ds = np.linspace(d_min, d_max, num_ds, dtype=int)
    args = [(d, n, num_grid_points) for d in ds]

    # Parallelize the computation over different d values
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_single_d, args), total=len(ds)))

    # Extract and plot elbow points
    ds, elbow_alphas = zip(*[(d, alpha) for d, alpha in results if alpha is not None])
    plt.plot(ds, elbow_alphas, marker='o')
    plt.title("Elbow Points for Different d Values")
    plt.xlabel("d")
    plt.ylabel("Alpha at Elbow Point")
    plt.grid(True)

    # Add best-fit line
    coefficients = np.polyfit(ds, elbow_alphas, 1)  # Linear fit
    best_fit_line = np.poly1d(coefficients)
    plt.plot(ds, best_fit_line(ds), label="Best Fit", linestyle="--")
    plt.legend()

    # Save the plot
    plt.savefig(f"./runs/elbow_points_n{n}.png")
    plt.show()


def analyse_spectrum_of_model_matrix():
    def A(alpha):
        return np.array([
            [0, 0, 1 / 4, 1 / 4, alpha / 2, (1 - alpha) / 2],
            [0, 0, 1 / 4, 1 / 4, alpha / 2, (1 - alpha) / 2],
            [1 / 4, 1 / 4, 0, 0, (1 - alpha) / 2, alpha / 2],
            [1 / 4, 1 / 4, 0, 0, (1 - alpha) / 2, alpha / 2],
            [alpha / 2, alpha / 2, (1 - alpha) / 2, (1 - alpha) / 2, 0, 0],
            [(1 - alpha) / 2, (1 - alpha) / 2, alpha / 2, alpha / 2, 0, 0]
        ])

    # Generate values of alpha
    alphas = np.linspace(1 / 2, 3 / 4, 100)  # alpha between 0 and 0.25
    lambda_2_values = [analyse_2nd_largest_eigenvalue(A(alpha)) for alpha in alphas]

    # Plot the relationship
    plt.plot(alphas, lambda_2_values, marker='o')
    plt.title(r"Relationship Between $\alpha$ and $\lambda_2(A(\alpha))$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\lambda_2(A(\alpha))$")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    n = 4000
    NUM_GRID_POINTS = 30
    NUM_DS = 50
    D_MIN = 200
    D_MAX = 500

    # find_elbow_points(n, NUM_GRID_POINTS, NUM_DS, D_MIN, D_MAX)

    d = 1750
    assert d <= 4 * n / 9

    plot_alpha_vs_eigenvalue(n, d, NUM_GRID_POINTS)
    # print(process_single_d((d, n, NUM_GRID_POINTS)))

    # alpha = 0.6  # Example parameter
    # adjacency_matrix = generate_d_regular_graph(n, d, alpha)
    # analyse_spectrum(adjacency_matrix)
    # plot_degree_distribution_from_adjacency_matrix(adjacency_matrix)

    analyse_spectrum_of_model_matrix()
