import numpy as np
from tools import analyse_spectrum, plot_degree_distribution
import random


def generate_graph(n, c, d, d_sb):
    d_ss = d - d_sb
    d_bb = d / 2 - d_sb * (c / (1 / 3 - 2 * c))
    assert d >= d_sb >= 0
    assert d >= d_ss >= 0
    assert d >= d_bb >= 0
    # Sizes of the parts
    size_Vii = int((1 / 3 - 2 * c) * n)
    size_Vij = int(c * n)
    # print(d_sb, d_ss, d_bb, size_Vii, size_Vij)
    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n))

    # Index ranges for parts V_ij and V_ii
    V_indices = {}
    index = 0
    for i in range(3):
        for j in range(3):
            if i == j:
                V_indices[f"V_{i}{i}"] = range(index, index + size_Vii)
                index += size_Vii
            else:
                V_indices[f"V_{i}{j}"] = range(index, index + size_Vij)
                index += size_Vij

    # Edges between V_ii and V_i'i' (distinct i, i')
    assert d_bb <= size_Vii
    for i in range(3):
        for i_prime in range(i + 1, 3):  # Only iterate over unique pairs (i, i_prime)
            Vii_indices = V_indices[f"V_{i}{i}"]
            Viprimeiprime_indices = V_indices[f"V_{i_prime}{i_prime}"]
            prob_bb = d_bb / size_Vii  # Adjusted probability to ensure expected degree d_bb
            for u in Vii_indices:
                for v in Viprimeiprime_indices:
                    if random.random() < prob_bb:
                        adjacency_matrix[u][v] = 1
                        adjacency_matrix[v][u] = 1  # Since the graph is undirected

    # Edges between V_ii and V_jj' (distinct i, j, j')
    assert d_sb <= size_Vii
    for i in range(3):
        for j in range(3):
            for j_prime in range(3):
                if i != j and j != j_prime and i != j_prime:
                    Vii_indices = V_indices[f"V_{i}{i}"]
                    Vjjprime_indices = V_indices[f"V_{j}{j_prime}"]
                    prob_sb = d_sb / size_Vii  # Adjusted probability for expected degree d_sb
                    for u in Vii_indices:
                        for v in Vjjprime_indices:
                            if random.random() < prob_sb:
                                adjacency_matrix[u][v] = 1
                                adjacency_matrix[v][u] = 1

    # Edges between V_ij and V_i'j (distinct i, i', j)
    assert d_ss <= size_Vij
    for i in range(3):
        for i_prime in range(i + 1, 3):  # Only iterate over unique pairs (i, i_prime)
            for j in range(3):
                if j != i and j != i_prime:
                    Vij_indices = V_indices[f"V_{i}{j}"]
                    Viprimej_indices = V_indices[f"V_{i_prime}{j}"]
                    prob_ss = d_ss / size_Vij  # Adjusted probability for expected degree d_ss
                    for u in Vij_indices:
                        for v in Viprimej_indices:
                            if random.random() < prob_ss:
                                adjacency_matrix[u][v] = 1
                                adjacency_matrix[v][u] = 1

    return adjacency_matrix


def find_optimal_d_sb(n, c, d):
    min_lambda_2 = float('inf')
    best_d_sb = None
    d_sb = 0
    while d_sb <= d:
        try:
            adjacency_matrix = generate_graph(n, c, d, d_sb)
            spectrum = analyse_spectrum(adjacency_matrix, print_info=False)
            lambda_2 = sorted(spectrum)[-2]  # Get the second largest eigenvalue
            if lambda_2 < min_lambda_2:
                min_lambda_2 = lambda_2
                best_d_sb = d_sb
            d_sb += 10  # Increment by a small value to densely search for optimal d_sb
        except AssertionError:
            d_sb += 10  # Increment by a small value to densely search for valid d_sb
    if best_d_sb is not None:
        print(f"Optimal d_sb found: {best_d_sb} with minimum lambda_2: {min_lambda_2}")
    else:
        print("No valid d_sb found within the range.")
    return best_d_sb


if __name__ == "__main__":
    n = 3000
    c = 1 / 20
    d = 300
    d_sb = find_optimal_d_sb(n, c, d)
    if d_sb is not None:
        adjacency_matrix = generate_graph(n, c, d, d_sb)
        print(analyse_spectrum(adjacency_matrix))
        plot_degree_distribution(adjacency_matrix)

    # adjacency_matrix = generate_graph(n=3000, c=1/20, d=1000, d_sb=900)
    # analyse_spectrum(adjacency_matrix)
    # plot_degree_distribution(adjacency_matrix)
