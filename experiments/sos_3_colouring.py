import cvxpy as cp
import numpy as np
import networkx as nx

def solve_3_coloring_sos(graph):
    n = graph.number_of_nodes()  # Number of vertices
    adj = nx.adjacency_matrix(graph).todense()  # Adjacency matrix

    # Create binary variables for each node and each of the 3 colors
    X = cp.Variable((n, 3))  # X[i, k] is 1 if vertex i is colored with color k, else 0

    # Constraints
    constraints = []

    # Constraint 1: Each vertex must be colored with exactly one color
    for i in range(n):
        constraints.append(cp.sum(X[i, :]) == 1)

    # Constraint 2: Adjacent vertices must be colored with different colors
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:  # Check if i and j are adjacent
                for k in range(3):
                    constraints.append(X[i, k] + X[j, k] <= 1)

    # Objective function (can be zero, we're only solving feasibility)
    objective = cp.Minimize(0)

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(solver=cp.SCS, verbose=True)

    # Check if a solution was found
    if problem.status == cp.OPTIMAL:
        # Extract the coloring (rounding the solution)
        coloring = np.argmax(X.value, axis=1)
        return coloring
    else:
        print("No solution found (or problem is infeasible).")
        return None

if __name__ == "__main__":
    # Create a sample graph (you can replace this with any graph)
    G = nx.erdos_renyi_graph(10, 0.5)  # Random graph with 10 nodes
    coloring = solve_3_coloring_sos(G)

    if coloring is not None:
        print("3-coloring found:", coloring)
    else:
        print("No 3-coloring possible.")
