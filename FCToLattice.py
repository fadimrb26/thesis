from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import minorminer
import numpy as np

# Step 1: Define the fully connected graph
num_nodes = 5  # Number of nodes
sparsity = 0.5  # Sparsity of the graph
cap_values_femto = 20 # Edge capacitance value in femtofarads
G = nx.erdos_renyi_graph(num_nodes, sparsity)

# Step 2: Find minimum 2D lattice size for embedding
def find_minimum_lattice_size(G):
    lattice_size = 2
    while lattice_size < 20:
        print(f"Trying {lattice_size}x{lattice_size} lattice...")
        square_lattice = nx.grid_2d_graph(lattice_size, lattice_size)
        if len(G.edges()) == 0:
            print("Graph has no edges! Assigning nodes randomly.")
            embedding = {node: [(np.random.randint(0, lattice_size), np.random.randint(0, lattice_size))] for node in G.nodes()}
        else:
            embedding = minorminer.find_embedding(G.edges(), square_lattice.edges())
            
        if embedding:
            print(f"Embedding found with {lattice_size}x{lattice_size} lattice!")
            return lattice_size, square_lattice, embedding
        lattice_size += 1

    print("No embedding found up to 20x20 lattice.")
    return None, None, None

def find_unit_edges(points):
    edges = []
    for p1, p2 in combinations(points, 2):  # Check all pairs of points
        if np.linalg.norm(np.array(p1) - np.array(p2)) == 1:
            edges.append((p1, p2))
    return edges

# Step 4: Generate SPICE Netlist
def generate_spice_netlist(G):
    spice_code = [
        "* CMOS Capacitive-Coupled Ising Machine",
        ".model PMOS PMOS (VTO=-0.4 KP=100u)",
        ".model NMOS NMOS (VTO=0.4 KP=200u)",
        ".model SWMODEL SW(Ron=1 Roff=1G Von=1 Voff=0.5)\n",
        "* Define the latch subcircuit",
        ".SUBCKT LATCH Q Qb VDD GND PARAMS: VQ_init=0 VQb_init=0",
        "M1 Q  Qb VDD VDD PMOS W=360n L=45n",
        "M2 Q  Qb GND GND NMOS W=180n L=45n",
        "M3 Qb Q VDD VDD PMOS W=360n L=45n",
        "M4 Qb Q GND GND NMOS W=180n L=45n",
        "C1 Q  GND 100f",
        "C2 Qb GND 100f",
        ".IC V(Q)={VQ_init} V(Qb)={VQb_init}",
        ".ENDS LATCH\n",
        "* Power supply",
        "Vdd VDD 0 1.1V\n"
    ]

    # Map logical nodes to SPICE node names
    logical_to_spice = {node: f"Q{node}" for node in G.nodes()}

    # Add latch instances
    for node in G.nodes():
        spice_code.append(f"X{node} {logical_to_spice[node]} {logical_to_spice[node]}b VDD 0 LATCH PARAMS: VQ_init=0 VQb_init=0")

    # Add switch connections based on the graph edges
    switch_idx = 0
    for (u, v) in G.edges():
        spice_code.append(f"* Connection between latch {u} and latch {v}")
        spice_code.append(f"S{switch_idx}_{switch_idx * 4} {logical_to_spice[u]} N{switch_idx * 4} FLY_CTRL{switch_idx} 0 SWMODEL")
        spice_code.append(f"S{switch_idx}_{switch_idx * 4 + 1} {logical_to_spice[v]} N{switch_idx * 4 + 1} FLY_CTRL{switch_idx} 0 SWMODEL")
        spice_code.append(f"S{switch_idx}_{switch_idx * 4 + 2} {logical_to_spice[u]}b N{switch_idx * 4 + 2} FLY_CTRL{switch_idx} 0 SWMODEL")
        spice_code.append(f"S{switch_idx}_{switch_idx * 4 + 3} {logical_to_spice[v]}b N{switch_idx * 4 + 3} FLY_CTRL{switch_idx} 0 SWMODEL")

        spice_code.append(f"S{switch_idx}i_{switch_idx * 4} {logical_to_spice[u]}b N{switch_idx * 4} FLY_CTRL{switch_idx} 0 SWMODEL")
        spice_code.append(f"S{switch_idx}i_{switch_idx * 4 + 1} {logical_to_spice[v]} N{switch_idx * 4 + 1} FLY_CTRL{switch_idx} 0 SWMODEL")
        spice_code.append(f"S{switch_idx}i_{switch_idx * 4 + 2} {logical_to_spice[u]} N{switch_idx * 4 + 2} FLY_CTRL{switch_idx} 0 SWMODEL")
        spice_code.append(f"S{switch_idx}i_{switch_idx * 4 + 3} {logical_to_spice[v]}b N{switch_idx * 4 + 3} FLY_CTRL{switch_idx} 0 SWMODEL")

        switch_idx += 1

    # Add flying capacitors
    for i in range(switch_idx):
        spice_code.append(f"C_fly_{i} N{i * 4} N{i * 4 + 1} {cap_values_femto}f")
        spice_code.append(f"C_fly_{i+1} N{i * 4 + 2} N{i * 4 + 3} {cap_values_femto}f")

    # Add control signals for switches
    for i in range(switch_idx):
        spice_code.append(f"Vfly_ctrl{i} FLY_CTRL{i} 0 DC 0")
        spice_code.append(f"Vnfly_ctrl{i} NFLY_CTRL{i} 0 DC 1.1")

    # Add simulation control
    spice_code.append(".tran 0.1n 60n")
    spice_code.append(".end")

    return "\n".join(spice_code)

# Find the minimum lattice size and embedding
lattice_size, square_lattice, embedding = find_minimum_lattice_size(G)

# Generate and save the SPICE netlist
spice_netlist = generate_spice_netlist(G)
with open(r"C:\Users\User\Downloads\CAD_IC_PROJECT\r0969864_ElMerheby_G1\ngspice_interface\files\input_netlists\AutomatedFCIsing.cir", "w") as f:
    f.write(spice_netlist)

print("SPICE netlist generated: AutomatedFCIsing.cir")

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
