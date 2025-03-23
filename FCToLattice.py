from itertools import combinations
import minorminer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Define the graph to embed
G = nx.erdos_renyi_graph(3, 0.1)  # Random graph with 5 nodes and 50% edge probability

def find_unit_edges(points):
    edges = []
    for p1, p2 in combinations(points, 2):  # Check all pairs of points
        if np.linalg.norm(np.array(p1) - np.array(p2)) == 1:
            edges.append((p1, p2))
    return edges

# Step 2: Find the minimum lattice size 
def find_minimum_lattice_size(G):
    # Start with a 3x3 lattice
    lattice_size = 2
    while lattice_size < 20:
        print(f"Trying {lattice_size}x{lattice_size} lattice...")
        # Create the square lattice (no diagonal edges)
        square_lattice = nx.grid_2d_graph(lattice_size, lattice_size)
        
        if len(G.edges()) == 0:
            print("Graph has no edges! Assigning nodes randomly.")
            embedding = {node: [(np.random.randint(0, lattice_size), np.random.randint(0, lattice_size))] for node in G.nodes()}
        else:
            embedding = minorminer.find_embedding(G.edges(), square_lattice.edges())

        # Check if the embedding was successful
        if embedding:
            print(f"Embedding found with {lattice_size}x{lattice_size} lattice!")
            return lattice_size, square_lattice, embedding
        # If not, increase the lattice size and try again
        lattice_size += 1
    print("No embedding found up to 12x12 lattice.")
    return None  # Indicate failure

# Find the minimum lattice size and embedding
lattice_size, square_lattice, embedding = find_minimum_lattice_size(G)

# Step 3: Count the number of physical spins used
physical_spins = set()
for chain in embedding.values():
    physical_spins.update(chain)
num_spins = len(physical_spins)

print("Minimum lattice size:", lattice_size, "x", lattice_size)
print("Number of physical spins used:", num_spins)
print("Embedding:", embedding)
print("Initial Edges:", G.edges())

# Step 4: Visualize the embedding
# Plot the original graph
plt.figure(figsize=(10, 5))
plt.subplot(121)
pos_G = nx.spring_layout(G, seed=42)  # Use spring layout for better spacing
nx.draw(G, pos_G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=600, font_size=10, font_weight='bold')
plt.title("Original Graph")

# Plot the embedded graph on the square lattice
plt.subplot(122)
pos = {(x, y): (x, y) for x, y in square_lattice.nodes()}  # Position nodes in a grid
nx.draw(square_lattice, pos, node_size=100, node_color='lightgray', with_labels=True)

# Draw the chains (direct couplings) in blue
for node, chain in embedding.items():
    for qubit in chain:
        nx.draw_networkx_nodes(square_lattice, pos, nodelist=[qubit], node_color='red', node_size=200)
    for edge in find_unit_edges(chain):
        nx.draw_networkx_edges(square_lattice, pos, edgelist=[edge], edge_color='blue', width=2)

# Add labels to show which nodes in the lattice correspond to which nodes in the original graph
for node, chain in embedding.items():
    for qubit in chain:
        plt.text(
            pos[qubit][0], pos[qubit][1] + 0.1,  # Offset the label slightly above the node
            f"{node}",                          # Label with the original graph node
            horizontalalignment='center',
            color='black',
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
plt.title(f"Embedded Graph on {lattice_size}x{lattice_size} Lattice\n"
          f"Initial Edges in Red, Chains in Blue")

# Draw the initial edges of the original graph in red on the lattice plot
for u, v in G.edges():
    # Get the chains for nodes u and v from the embedding
    chain_u = embedding[u]
    chain_v = embedding[v]
    # Draw edges between the first qubit of chain_u and the first qubit of chain_v
    qubit_u = chain_u[-1]
    qubit_v = chain_v[-1]
    # Check if the edge is horizontal, vertical, or diagonal
    if (qubit_u[0] == qubit_v[0] and abs(qubit_u[1]-qubit_v[1]) == 1) or (qubit_u[1] == qubit_v[1] and abs(qubit_u[0]-qubit_v[0]) == 1):
        nx.draw_networkx_edges(
            square_lattice, pos,
            edgelist=[(qubit_u, qubit_v)],
            edge_color='red',
            width=2
        )
    else:
        # Check if there is another node in the chain that can form a straight connection
        for i_qubit in chain_u[::-1]:
            for j_qubit in chain_v[::-1]:
                if (i_qubit[0] == j_qubit[0] and abs(i_qubit[1]-j_qubit[1]) == 1) or (i_qubit[1] == j_qubit[1] and abs(i_qubit[0]-j_qubit[0]) == 1):
                    nx.draw_networkx_edges(
                        square_lattice, pos,
                        edgelist=[(i_qubit, j_qubit)],
                        edge_color='red',
                        width=2
                    )
                    break 
            else:
                continue  
            break  
plt.show()

