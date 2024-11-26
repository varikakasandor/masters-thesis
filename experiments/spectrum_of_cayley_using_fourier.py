import numpy as np
from itertools import product
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools


# Function to calculate the Fourier coefficient for a single element y
def fourier_coefficient(y, Zp_d, A, p):
    def f(x):
        return 1 if tuple(x) in A else 0

    # Calculate the Fourier coefficient for y
    eigenvalue = sum(
        f(x) * np.exp(-2j * np.pi * (np.dot(x, y) % p) / p) for x in Zp_d
    ).real  # Take the real part to get the Fourier coefficient
    return eigenvalue


# Function to calculate the Fourier coefficients and spectrum
# This version uses parallel processing to speed up the computation
def cayley_graph_spectrum_parallel(d, p, A):
    # Step 1: Generate all vectors in Z_p^d
    Zp_d = list(product(range(p), repeat=d))

    # Step 2: Use multiprocessing to calculate Fourier coefficients with tqdm progress bar
    with Pool(cpu_count()) as pool:
        spectrum = list(
            tqdm(pool.imap(functools.partial(fourier_coefficient, Zp_d=Zp_d, A=A, p=p), Zp_d), total=len(Zp_d),
                 desc="Calculating Fourier Coefficients"))

    spectrum.sort(reverse=True)
    return spectrum


# Wrapper function to find the best set A based on the criteria
def find_best_A_parallel(d, p, ITER):
    Zp_d = list(product(range(p), repeat=d))
    best_A = None
    lowest_lambda_2 = float('inf')

    for _ in tqdm(range(ITER), desc="Searching for best set A"):
        # Step 1: Choose a random set A of size 10d
        A = set(random.sample(Zp_d, 10 * d))

        # Step 2: Calculate the spectrum
        spectrum = cayley_graph_spectrum_parallel(d, p, A)
        spectrum.sort(reverse=True)  # Sort eigenvalues in descending order

        # Step 3: Check the conditions
        lambda_2 = spectrum[1]  # Second largest eigenvalue
        lambda_n_minus_2 = spectrum[-3]  # Second smallest eigenvalue

        if lambda_n_minus_2 <= -(4 / 10) * (10 * d) and lambda_2 < lowest_lambda_2:
            best_A = A
            lowest_lambda_2 = lambda_2

    return best_A, lowest_lambda_2


# Function to check for edges within sets
def check_edges_within_sets(d, p, A):
    # Define V_1, V_2, V_3 based on sum(x) % 3
    V1 = {v for v in product(range(p), repeat=d) if sum(v) % 3 == 0}
    V2 = {v for v in product(range(p), repeat=d) if sum(v) % 3 == 1}
    V3 = {v for v in product(range(p), repeat=d) if sum(v) % 3 == 2}

    # Check for edges within each V_i
    def has_internal_edges(V):
        for u in V:
            for v in V:
                if u != v and tuple((np.array(v) - np.array(u) + p) % p) in A:
                    return u, v
        return None, None

    u1, v1 = has_internal_edges(V1)
    u2, v2 = has_internal_edges(V2)
    u3, v3 = has_internal_edges(V3)

    return (u1, v1), (u2, v2), (u3, v3)


if __name__ == "__main__":
    # Example usage
    d = 7  # Dimension
    p = 3  # Prime modulus (can be any prime)
    A = {v for v in product(range(p), repeat=d) if 1 <= sum(v) <= 2}  # EXAMPLE

    # Calculate spectrum for set A
    spectrum = cayley_graph_spectrum_parallel(d, p, A)
    formatted_spectrum = [float(e) for e in spectrum]  # Convert np.float64 to standard float for nicer output
    selected_evals = [formatted_spectrum[0], formatted_spectrum[1], formatted_spectrum[-3], formatted_spectrum[-2],
                      formatted_spectrum[-1]]
    print("Spectrum summary: [{}]".format(
        ", ".join(
            [str(selected_evals[0]), str(selected_evals[1]), "...", str(selected_evals[2]), str(selected_evals[3]),
             str(selected_evals[4])])
    ))

    # Check for edges within V_1, V_2, V_3
    (u1, v1), (u2, v2), (u3, v3) = check_edges_within_sets(d, p, A)
    if u1 and v1:
        print("Example edge within V1:", u1, v1, tuple(int(x) for x in (np.array(u1) + np.array(v1)) % p))
    else:
        print("No edges within V1")

    if u2 and v2:
        print("Example edge within V2:", u2, v2, tuple(int(x) for x in (np.array(u2) + np.array(v2)) % p))
    else:
        print("No edges within V2")

    if u3 and v3:
        print("Example edge within V3:", u3, v3, tuple(int(x) for x in (np.array(u3) + np.array(v3)) % p))
    else:
        print("No edges within V3")

    """
    # Use the wrapper function to find the best set A
    ITER = 1000  # Number of iterations to try different sets A
    best_A, lowest_lambda_2 = find_best_A_parallel(d, p, ITER)
    if best_A is not None:
        print("Best set A found with lowest lambda_2:", best_A)
        print("Lowest lambda_2:", lowest_lambda_2)
    else:
        print("No set A found that meets the criteria.")
    """
