import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

class GraphEnv:
    def __init__(self, n):
        self.n = n
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n))
        self.adj_matrix = np.zeros((n, n), dtype=int)

    def reset(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n))
        self.adj_matrix = np.zeros((self.n, self.n), dtype=int)
        return self.adj_matrix

    def add_edge(self, u, v):
        if u != v and not self.graph.has_edge(u, v):
            self.graph.add_edge(u, v)
            self.adj_matrix[u, v] = 1
            self.adj_matrix[v, u] = 1

    def remove_edge(self, u, v):
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            self.adj_matrix[u, v] = 0
            self.adj_matrix[v, u] = 0

    def get_edge_expansion(self, S):
        boundary_edges = list(nx.edge_boundary(self.graph, S))
        return len(boundary_edges) / min(len(S), len(self.graph) - len(S))

    def is_arc_transitive(self):
        try:
            aut_group = nx.algorithms.isomorphism.GraphMatcher(self.graph, self.graph)
            return aut_group.is_isomorphic()
        except:
            return False

    def step(self, action, max_steps, current_step):
        u, v = random.sample(range(self.n), 2)
        if action == 0:
            self.add_edge(u, v)
        elif action == 1:
            self.remove_edge(u, v)

        reward = 0
        if self.is_arc_transitive():
            reward += 1
        else:
            reward -= 1

        small_set = random.sample(list(self.graph.nodes()), max(1, self.n // 4))
        expansion_factor = self.get_edge_expansion(small_set)
        if expansion_factor > 1.1:
            reward += 1
        else:
            reward -= 1

        done = False
        if current_step >= max_steps:
            done = True
        return self.adj_matrix, reward, done


class RLAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(2))
        self.epsilon = 0.1
        self.lr = 0.1
        self.gamma = 0.9
        self.final_graph = None

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])
        else:
            return np.argmax(self.q_table[str(state)])

    def learn(self, state, action, reward, next_state):
        q_predict = self.q_table[str(state)][action]
        q_target = reward + self.gamma * np.max(self.q_table[str(next_state)])
        self.q_table[str(state)][action] += self.lr * (q_target - q_predict)

    def train(self, episodes=100, max_steps=10):
        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            done = False
            step_count = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action, max_steps, step_count)
                self.learn(state, action, reward, next_state)
                state = next_state
                step_count += 1

        self.final_graph = self.env.graph

    def plot_final_graph(self):
        if self.final_graph is None:
            return

        plt.figure(figsize=(6, 6))
        nx.draw(self.final_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500,
                font_size=10)
        plt.title("Final Graph after Training")
        plt.show()

    def check_graph_properties(self):
        if self.final_graph is None:
            print("No graph available for analysis.")
            return

        print("Evaluating the final graph:")

        is_arc_transitive = self.env.is_arc_transitive()
        print(f"Arc-transitivity: {'Yes' if is_arc_transitive else 'No'}")

        low_expansion_sets = []
        for subset in nx.enumerate_all_cliques(self.final_graph):
            if len(subset) < len(self.final_graph) // 4:
                expansion_factor = self.env.get_edge_expansion(subset)
                if expansion_factor < 1.1:
                    low_expansion_sets.append((subset, expansion_factor))

        if low_expansion_sets:
            print(f"Found {len(low_expansion_sets)} small set(s) with low edge expansion.")
            for i, (subset, expansion_factor) in enumerate(low_expansion_sets):
                print(f"  Set {i + 1}: {subset}, Edge Expansion: {expansion_factor:.2f}")
        else:
            print("No small set with low edge expansion found.")


if __name__ == "__main__":
    # Example usage
    n = 50  # Number of vertices
    env = GraphEnv(n)
    agent = RLAgent(env)
    agent.train(episodes=1000, max_steps=10)
    agent.plot_final_graph()
    agent.check_graph_properties()


