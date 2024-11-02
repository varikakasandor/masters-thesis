import numpy as np
import matplotlib.pyplot as plt


# Define the functions involved in the conditions
def condition_1(x, f):
    term1 = ((3 * x - 1) * (3 * x + 2 - f)) ** 2
    term2 = (3 * x + 2) ** 2
    lhs_old = 2 * term1 / term2
    lhs_new = (3 * (9 * x ** 2 + 3 * (1 - f) * x + (f - 2)) ** 2) / (
                2 * (2 + 3 * x) * (9 * x ** 2 + (12 + 6 * f) * x + (4 - 2 * f)))
    lhs = np.maximum(lhs_old, lhs_new)
    rhs = (
            2 * np.log(2)
            - (1 - 3 * x) * np.log(1 - 3 * x)
            - (1 + 3 * x) * np.log(1 + 3 * x)
    )
    return lhs - rhs


def condition_2(x, f):
    return x <= (f - 2) / 3


def condition_3(x, f):
    return x >= 1 / 3 - 1 / (f + 1)


def find_smallest_f_for_x(x):
    for f in np.arange(0.1, 2000, 0.1):
        if condition_1(x, f) > 0 and condition_2(x, f) and condition_3(x, f):
            return f
    return None


if __name__ == "__main__":
    # Iterate over values of x in the interval (0, 1/3)
    x_values = np.linspace(1 / 6, 0.31, 100)
    f_values = []

    for x in x_values:
        smallest_f = find_smallest_f_for_x(x)
        if smallest_f is not None:
            f_values.append(smallest_f)
            print(f"x: {x:.5f}, smallest f: {smallest_f:.5f}")
        else:
            f_values.append(None)
            print(f"x: {x:.5f}, no valid f found")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, f_values, label='Smallest f for each x', marker='o', linestyle='-', color='b')
    plt.xlabel('alpha')
    plt.ylabel('required vertex expansion f')
    plt.title('Vertex expansion on sets of bounded size vs largest intersection')
    plt.grid(True)
    plt.legend()
    plt.savefig('vertex_expansion_vs_planted_intersection.png')
    plt.show()
