import numpy as np
from itertools import product
import random
from tqdm import tqdm


# Function to calculate the Fourier coefficients and spectrum
def cayley_graph_spectrum(d, A):
    # Step 1: Generate all vectors in F_2^d
    F2_d = list(product([0, 1], repeat=d))
    # Step 2: Define the indicator function f(x) for the set A
    def f(x):
        return 1 if tuple(x) in A else 0

    # Step 3: Calculate Fourier coefficients (spectrum of the adjacency matrix)
    spectrum = []
    for y in F2_d:
        eigenvalue = sum(
            f(x) * (-1) ** np.dot(x, y) for x in F2_d
        )
        spectrum.append(eigenvalue)
    spectrum.sort(reverse=True)
    return spectrum


# Wrapper function to find the best set A based on the criteria
def find_best_A(d, ITER):
    F2_d = list(product([0, 1], repeat=d))
    best_A = None
    lowest_lambda_2 = float('inf')

    for _ in tqdm(range(ITER), desc="Searching for best set A"):
        # Step 1: Choose a random set A of size 10d
        A = set(random.sample(F2_d, 10 * d))

        # Step 2: Calculate the spectrum
        spectrum = cayley_graph_spectrum(d, A)
        spectrum.sort(reverse=True)  # Sort eigenvalues in descending order

        # Step 3: Check the conditions
        lambda_2 = spectrum[1]  # Second largest eigenvalue
        lambda_n_minus_2 = spectrum[-3]  # Second smallest eigenvalue

        if lambda_n_minus_2 <= -(4 / 10) * (10 * d) and lambda_2 < lowest_lambda_2:
            best_A = A
            lowest_lambda_2 = lambda_2

    return best_A, lowest_lambda_2


if __name__ == "__main__":
    d = 9
    # A = {(1, 0, 0, 1, 0, 0, 0, 0, 1), (1, 1, 0, 1, 0, 0, 0, 1, 0), (1, 0, 0, 0, 0, 1, 1, 1, 1), (0, 0, 0, 1, 1, 1, 1, 0, 1), (0, 0, 1, 1, 1, 1, 0, 1, 1), (0, 1, 1, 1, 0, 1, 0, 1, 0), (1, 1, 1, 1, 0, 1, 0, 0, 0), (0, 1, 1, 1, 1, 1, 0, 1, 1), (1, 1, 0, 0, 0, 0, 0, 0, 1), (0, 1, 1, 1, 1, 0, 0, 0, 1), (1, 1, 1, 1, 1, 0, 1, 1, 0), (1, 0, 1, 0, 0, 1, 1, 0, 1), (1, 0, 1, 0, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 0, 1, 0, 1), (0, 1, 0, 0, 1, 1, 1, 1, 1), (1, 0, 0, 1, 0, 1, 1, 1, 1), (0, 1, 0, 0, 0, 1, 0, 1, 0), (1, 1, 1, 1, 0, 0, 1, 1, 1), (0, 0, 1, 0, 0, 1, 0, 0, 1), (0, 0, 0, 1, 0, 0, 1, 0, 1), (0, 1, 1, 0, 0, 1, 0, 0, 1), (0, 0, 1, 0, 1, 1, 1, 0, 0), (0, 0, 0, 0, 1, 0, 0, 1, 1), (0, 0, 0, 0, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 0, 1, 0, 0), (0, 1, 1, 1, 0, 1, 1, 0, 1), (0, 1, 1, 0, 1, 0, 0, 1, 1), (1, 1, 0, 0, 1, 1, 0, 1, 1), (0, 0, 1, 0, 1, 0, 0, 1, 1), (0, 0, 1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 1, 0, 1, 0), (1, 1, 1, 0, 1, 0, 1, 0, 0), (1, 0, 1, 0, 0, 0, 0, 1, 0), (0, 1, 1, 1, 1, 1, 1, 0, 0), (1, 1, 1, 1, 0, 0, 0, 1, 1), (1, 0, 1, 0, 0, 0, 0, 0, 1), (1, 0, 1, 0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 1, 0, 1, 0, 0), (1, 0, 0, 1, 0, 0, 1, 0, 0), (0, 1, 1, 0, 1, 1, 0, 1, 0), (1, 0, 1, 1, 1, 0, 0, 0, 1), (1, 0, 0, 1, 0, 1, 1, 1, 0), (0, 0, 1, 1, 0, 1, 1, 0, 1), (0, 0, 1, 0, 0, 1, 0, 0, 0), (1, 0, 1, 1, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 1, 0, 1, 1, 1), (0, 1, 1, 0, 1, 1, 1, 0, 0), (1, 0, 0, 0, 1, 1, 1, 1, 0), (1, 1, 1, 1, 1, 0, 0, 1, 0), (0, 1, 0, 1, 0, 0, 0, 0, 0), (1, 0, 1, 0, 1, 1, 1, 1, 1), (1, 1, 0, 1, 1, 0, 1, 1, 1), (0, 1, 1, 1, 1, 0, 1, 1, 1), (1, 1, 0, 0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 0, 1, 0, 1, 1), (0, 0, 1, 1, 1, 0, 1, 1, 1), (0, 0, 1, 0, 1, 0, 0, 1, 0), (0, 0, 0, 1, 1, 1, 0, 1, 1), (1, 1, 1, 1, 0, 1, 0, 1, 0), (0, 0, 0, 1, 1, 1, 1, 1, 1), (1, 1, 1, 0, 1, 0, 0, 1, 1), (0, 1, 1, 0, 0, 1, 1, 0, 0), (0, 0, 0, 0, 1, 0, 1, 1, 0), (0, 1, 0, 0, 0, 0, 0, 1, 0), (1, 0, 0, 0, 0, 1, 1, 0, 0), (0, 1, 1, 0, 0, 0, 1, 0, 1), (0, 0, 1, 0, 0, 1, 1, 1, 1), (0, 1, 0, 0, 1, 1, 0, 1, 0), (0, 0, 1, 1, 0, 0, 0, 0, 1), (1, 0, 1, 1, 0, 0, 0, 1, 1), (1, 1, 1, 1, 1, 0, 1, 1, 1), (0, 0, 0, 1, 1, 0, 0, 1, 0), (0, 0, 1, 1, 0, 0, 1, 0, 1), (0, 1, 1, 0, 0, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0, 0, 1, 0), (0, 1, 1, 1, 1, 0, 0, 1, 0), (1, 0, 1, 1, 0, 1, 1, 0, 0), (0, 1, 1, 0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1, 1, 0, 1), (0, 1, 1, 0, 1, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0, 1, 1, 0), (1, 1, 0, 0, 1, 1, 0, 0, 1), (0, 0, 0, 0, 0, 0, 0, 1, 1), (0, 1, 0, 1, 0, 1, 0, 0, 0), (1, 1, 0, 0, 0, 1, 1, 1, 1), (0, 0, 0, 0, 1, 1, 0, 0, 1), (1, 1, 1, 1, 1, 1, 0, 0, 1), (1, 0, 1, 1, 1, 1, 0, 0, 1), (1, 0, 0, 0, 0, 0, 1, 0, 1)}  # Define A as a set of tuples representing vectors
    # spectrum = cayley_graph_spectrum(d, A)
    # formatted_spectrum = [float(e) for e in spectrum]  # Convert np.float64 to standard float for nicer output
    # print("Spectrum:", formatted_spectrum)

    #Use the wrapper function to find the best set A
    ITER = 1000  # Number of iterations to try different sets A
    best_A, lowest_lambda_2 = find_best_A(d, ITER)
    if best_A is not None:
        print("Best set A found with lowest lambda_2:", best_A)
        print("Lowest lambda_2:", lowest_lambda_2)
    else:
        print("No set A found that meets the criteria.")
