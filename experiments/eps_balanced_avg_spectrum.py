import numpy as np
import matplotlib.pyplot as plt


def objective_function(a, b, c):
    denominator = a * b * c
    numerator = 1 - 4 * (a * b + a * c + b * c)
    if denominator <= 0:  # Avoid division by zero
        return -np.inf
    value = np.sqrt(numerator / denominator + 9)
    return value if np.isreal(value) and value > 0 else -np.inf


def grid_search_max(epsilon, grid_size=100):
    best_value = -np.inf
    best_triplet = None

    lower_bound = 2 * epsilon
    upper_bound = 1 / 2 - epsilon

    a_values = np.linspace(lower_bound, upper_bound, grid_size)

    for a in a_values:
        for b in a_values:
            c = 1 - (a + b)
            if lower_bound < c < upper_bound:
                value = objective_function(a, b, c)
                if value > best_value:
                    best_value = value
                    best_triplet = (a, b, c)

    return best_value, best_triplet


def theoretical_function(epsilon):
    return np.sqrt(9 - (2 * (1 - 3 * epsilon)) / ((1 / 2 - epsilon) ** 2))


if __name__ == "__main__":
    # Example usage
    epsilon_values = np.linspace(0.01, 1 / 6, 50)
    max_values = [grid_search_max(eps)[0] for eps in epsilon_values]
    theoretical_values = [theoretical_function(eps) for eps in epsilon_values]

    plt.plot(epsilon_values, max_values, marker='o', label="Empirical Maximum")
    plt.plot(epsilon_values, theoretical_values, marker='x', linestyle='--', label="Theoretical Function")
    plt.xlabel("Epsilon")
    plt.ylabel("Maximum Value")
    plt.title("Maximum Value as a Function of Epsilon")
    plt.legend()
    plt.grid()
    plt.show()

    # Print result for a single epsilon
    epsilon = 0.01  # Define small epsilon
    best_value, best_triplet = grid_search_max(epsilon)
    print("Maximum value:", best_value)
    print("Best (a, b, c):", best_triplet)