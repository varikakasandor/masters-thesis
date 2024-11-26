import networkx as nx
import numpy as np
import itertools


# Function to create an Odd graph of order k
def create_odd_graph(k):
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    G = nx.Graph()
    nodes = [tuple(sorted(combo)) for combo in itertools.combinations(range(2 * k - 1), k - 1)]
    G.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if len(set(u).intersection(v)) == 0:
                G.add_edge(u, v)
    return G


def main():
    # Create the Odd graph of order k
    k = 4  # You can set k to any value you want
    G = create_odd_graph(k)

    # Get the adjacency matrix of the Odd graph
    A = nx.adjacency_matrix(G).toarray()

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Print the eigenvalues
    print("Eigenvalues:")
    print(eigenvalues)

    # Find the smallest eigenvalue
    smallest_eigenvalue = np.min(eigenvalues)

    # Find the indices of the eigenvectors corresponding to the smallest eigenvalue
    indices = np.isclose(eigenvalues, smallest_eigenvalue)

    # Get the corresponding eigenvectors
    smallest_eigenvectors = eigenvectors[:, indices]

    # Process the eigenvectors: sort values in descending order, truncate to 1 decimal, and multiply by 10
    processed_eigenvectors = []
    for vec in smallest_eigenvectors.T:
        processed_vec = np.sort(vec)[::-1]  # Sort in descending order
        processed_vec = np.round(processed_vec * 10, 1)  # Multiply by 10 and truncate to 1 decimal
        processed_eigenvectors.append(processed_vec)

    # Print the processed eigenvectors corresponding to the smallest eigenvalue
    print("Processed eigenvectors corresponding to the smallest eigenvalue:")
    for vec in processed_eigenvectors:
        print(vec)


if __name__ == "__main__":
    main()
