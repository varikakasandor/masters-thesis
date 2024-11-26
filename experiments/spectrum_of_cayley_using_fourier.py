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


if __name__ == "__main__":
    # Example usage
    d = 8  # Dimension
    p = 3  # Prime modulus (can be any prime)
    A = {v for v in product(range(p), repeat=d) if 1 <= sum(v) <= 2} # EXAMPLE

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
