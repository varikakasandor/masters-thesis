import numpy as np
from itertools import product
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools


# Function to calculate the Fourier coefficient for a single element y
def fourier_coefficient(y, Zp_d, A, p):
    # Step 2: Define the indicator function f(x) for the set A
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


if __name__ == "__main__":
    # Example usage
    d = 8  # Dimension
    p = 3  # Prime modulus (can be any prime)
    A = {v for v in product(range(p), repeat=d) if 1 <= sum(v) <= 2}

    # Calculate spectrum for set A
    spectrum = cayley_graph_spectrum_parallel(d, p, A)
    formatted_spectrum = [float(e) for e in spectrum]  # Convert np.float64 to standard float for nicer output
    print("Spectrum:", formatted_spectrum)
