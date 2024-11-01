import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect


def target_function(alpha, beta):
    term1 = (1 - alpha - beta) * np.log(1 - alpha - beta)
    term2 = -2 * (1 - beta) * np.log(1 - beta)
    term3 = (2 / 3) * np.log(2 / 3)
    term4 = -(2 / 3 - alpha) * np.log(2 / 3 - alpha)
    term5 = -beta * np.log(beta)
    return term1 + term2 + term3 + term4 + term5


def find_smallest_alpha(beta):
    # Function that returns the value of target_function at specific beta and alpha
    def func(alpha):
        return target_function(alpha, beta)

    # Define range for alpha
    alpha_min = 0.00000001
    alpha_max = 0.66666666

    # Using bisection method to find root where the function is negative
    # We find the smallest alpha such that func(alpha) < 0
    try:
        result = bisect(lambda alpha: func(alpha), alpha_min, alpha_max, xtol=1e-5, rtol=1e-5)
        return result
    except ValueError:
        return None


if __name__ == "__main__":

    # Loop through values of beta in the interval (0, 1/3)
    step = 0.01
    beta_values = np.arange(0.01, 1 / 3, step)
    alpha_values = []

    for beta in beta_values:
        smallest_alpha = find_smallest_alpha(beta)
        if smallest_alpha is not None:
            alpha_values.append(smallest_alpha)
            print(f"For beta = {beta:.4f}, the smallest alpha is: {smallest_alpha:.4f}")
        else:
            alpha_values.append(None)
            print(f"No suitable alpha found for beta = {beta:.4f}")

    # Calculate alpha/beta for plotting
    expansion_values = [alpha / beta if alpha is not None else None for alpha, beta in zip(alpha_values, beta_values)]

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(beta_values, expansion_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Beta values')
    plt.ylabel('Beta-expansion')
    plt.title('Required Small-Set Expansion')
    plt.grid(True)
    plt.show()
