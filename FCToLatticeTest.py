from itertools import combinations
import minorminer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define the function to find embeddings
def find_minimum_lattice_size(G):
    lattice_size = 2
    while lattice_size < 20:
        square_lattice = nx.grid_2d_graph(lattice_size, lattice_size)
        if len(G.edges()) == 0:
            embedding = {node: [(np.random.randint(0, lattice_size), np.random.randint(0, lattice_size))] for node in G.nodes()}
        else:
            embedding = minorminer.find_embedding(G.edges(), square_lattice.edges())
        if embedding:
            return lattice_size, embedding
        lattice_size += 1
    return None

# Step 2: Test different graph sizes and sparsities
node_range = range(3, 12)
sparsities = np.linspace(0.1, 1.0, 10)
num_tests = 20
success_rates = np.zeros((len(node_range), len(sparsities)))

for i, num_nodes in enumerate(node_range):
    for j, sparsity in enumerate(sparsities):
        success_count = 0
        for k in range(num_tests):
            G = nx.erdos_renyi_graph(num_nodes, sparsity)
            print(f"Testing: Nodes={num_nodes}, Sparsity={sparsity:.2f}, Example={k+1}")
            if find_minimum_lattice_size(G):
                success_count += 1
        success_rates[i, j] = success_count / num_tests

# Step 3: Plot the success rates
plt.figure(figsize=(8, 6))
plt.imshow(success_rates, cmap='coolwarm', aspect='auto', origin='lower', extent=[0.1, 1.0, 3, 11])
plt.colorbar(label='Success Rate')
plt.xlabel('Edge Probability (Sparsity)')
plt.ylabel('Number of Nodes')
plt.title('Embedding Success Rate')
plt.show()
