# %% ================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import time
from itertools import combinations

# %% [markdown]
# # Simulating Time With Square-Root Space [paper](https://arxiv.org/abs/2502.17779)
## Overview
# This notebook tries to follow the implementation of 'Simulating Time With Square-Root Space'. The significant result of this paper was making a little progress on the $P$ vs $PSPACE$ problem. The authors do this by reducing the Turing machines to 'Tree Evaluation' instances leverging a more space-efficient structure for 'Tree Evaluation' which is a separate algorithm found by Cook and Mertz [STOC 2024].

# %% =================================
# Define a simple 2D grid representing time t and block length b
t_values = np.linspace(100, 10000, 100)  # time from 10^2 to 10^4
b_values = np.linspace(10, 1000, 100)  # block size from 10 to 1000
T, B = np.meshgrid(t_values, b_values)

# Compute space using the formula: space = b + (t / b) * log(b)
Space = B + (T / B) * np.log(B)

# 3D Plotting the space usage surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(T, B, Space, cmap="viridis")
ax.set_xlabel("Time (t)")
ax.set_ylabel("Block Size (b)")
ax.set_zlabel("Space Complexity")
ax.set_title("Visualization of Space = b + (t/b) * log(b)")

plt.tight_layout()
plt.show()


# %% =================================
# Define time range
t_vals = np.linspace(100, 10000, 300)
# Optimal block size b = sqrt(t * log(t))
optimal_b = np.sqrt(t_vals * np.log(t_vals))
# Corresponding space complexity
optimal_space = optimal_b + (t_vals / optimal_b) * np.log(optimal_b)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    t_vals,
    optimal_space,
    label="Space at Optimal b",
    color="darkblue",
)
ax.set_xlabel("Time (t)")
ax.set_ylabel("Space Complexity")
ax.set_title(r"Optimal Space Complexity with $b = \sqrt{t \log t}$")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


# %% =================================
def simulate_binary_addition_tree(input1, input2, block_size):
    from math import ceil

    assert len(input1) == len(input2)
    n = len(input1)
    t = 2 * n
    B = ceil(t / block_size)

    def simulate_block(start, state):
        return state + 1

    state = 0
    for block in range(B):
        state = simulate_block(block * block_size, state)

    return state  # Should reflect final carry or accept state


def build_block_tree(num_blocks):
    """
    Build a binary tree representing block-level simulation dependency
    Each node represents a block that simulates b steps of the original Turing machine
    """
    G = nx.DiGraph()
    labels = {}

    def add_nodes(current, depth, max_depth):
        if depth > max_depth:
            return
        left = current * 2
        right = current * 2 + 1
        G.add_edge(current, left)
        G.add_edge(current, right)
        labels[current] = f"B{current}"
        labels[left] = f"B{left}"
        labels[right] = f"B{right}"
        add_nodes(left, depth + 1, max_depth)
        add_nodes(right, depth + 1, max_depth)

    depth = int(np.ceil(np.log2(num_blocks)))
    add_nodes(1, 1, depth)
    return G, labels


def draw_tree(G):
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1500,
        node_color="lightblue",
        font_size=10,
        arrows=True,
    )
    plt.title("Tree Structure for Simulating Binary Addition via Blocks")
    plt.tight_layout()
    plt.show()


# %%
input1 = [1, 0, 1, 0, 1, 0, 1, 0]
input2 = [0, 1, 0, 1, 0, 1, 0, 1]
block_size = 2
print(simulate_binary_addition_tree(input1, input2, block_size))

tree, labels = build_block_tree(4)
draw_tree(tree)


# %%
class TreeNode:
    def __init__(self, label, left=None, right=None, is_leaf=False, value=None):
        self.label = label
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = value  # Only used for leaves


def build_value_tree(depth, label_prefix="N"):
    """
    Build a full binary tree of depth `depth`, assigning values at leaves
    """
    counter = [1]  # mutable counter to assign unique values to leaves

    def build(d):
        if d == 0:
            val = counter[0]
            node = TreeNode(f"L{val}", is_leaf=True, value=val)
            counter[0] += 1
            return node
        left = build(d - 1)
        right = build(d - 1)
        return TreeNode(f"{label_prefix}{counter[0]}", left, right)

    return build(depth)


def evaluate_tree(node, depth=0):
    """
    DFS-style evaluation that reuses a small workspace
    """
    print("  " * depth + f"Evaluating {node.label}...")
    if node.is_leaf:
        print("  " * depth + f"Leaf value = {node.value}")
        return node.value

    left_val = evaluate_tree(node.left, depth + 1)
    right_val = evaluate_tree(node.right, depth + 1)

    node_value = left_val + right_val
    print("  " * depth + f"{node.label} = {left_val} + {right_val} = {node_value}")
    return node_value


# Create and evaluate a toy tree
root = build_value_tree(depth=3)  # 8 leaves
result = evaluate_tree(root)
print(f"\nFinal result at root: {result}")


# %%
class Event:
    """
    Event class with Minkowski spacetime embedding
    """

    def __init__(self, id, x, y, t, block_state):
        self.id = id
        self.x = x
        self.y = y
        self.t = t
        self.block_state = block_state
        self.coords = np.array([t, x, y])

    def interval_to(self, other):
        dt = self.t - other.t
        dx = self.x - other.x
        dy = self.y - other.y
        return -(dt**2) + dx**2 + dy**2

    def causally_connects(self, other, epsilon=1e-5):
        return self.interval_to(other) <= -epsilon and self.t < other.t


def generate_spacetime_events(num_blocks, block_states):
    """
    Generate spacetime events for a block-respecting Turing machine
    """
    np.random.seed(0)
    events = []
    for i in range(num_blocks):
        x, y = np.random.uniform(-1, 1, 2)
        t = i * 1.0
        events.append(Event(i, x, y, t, block_states[i]))
    return events


def build_causal_graph(events):
    """
    Build a causal graph from spacetime events
    """
    G = nx.DiGraph()
    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events):
            if i != j and e1.causally_connects(e2):
                G.add_edge(e1.id, e2.id)
    return G


def find_null_geodesics(events, tolerance=1e-3):
    """
    Find null geodesics in the causal graph
    """
    null_pairs = []
    for i, e1 in enumerate(events):
        for j, e2 in enumerate(events):
            if i != j and e1.t < e2.t:
                s2 = e1.interval_to(e2)
                if abs(s2) < tolerance:
                    null_pairs.append((e1.id, e2.id))
    return null_pairs


def plot_with_null_geodesics(events, causal_G, null_edges):
    """
    Plot the causal graph with null geodesics
    """
    pos = {e.id: (e.x, e.y) for e in events}
    labels = {e.id: f"B{e.id}" for e in events}
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_edges(causal_G, pos, edge_color="black", arrowsize=20)
    nx.draw_networkx_nodes(causal_G, pos, node_color="lightblue", node_size=900)
    nx.draw_networkx_labels(causal_G, pos, labels=labels, font_size=10)
    for i, (u, v) in enumerate(null_edges):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        plt.plot(
            [x0, x1],
            [y0, y1],
            color="red",
            linestyle="--",
            linewidth=2,
            label="null geodesic" if i == 0 else "",
        )
    plt.title("Causal Graph with Null Geodesics (Reversible Simulation Paths)")
    plt.legend()
    plt.axis("equal")
    plt.show()


# Simulate and visualize
num_blocks = 8
block_states = [f"State-{i}" for i in range(num_blocks)]
events = generate_spacetime_events(num_blocks, block_states)
G = build_causal_graph(events)
null_geodesic_edges = find_null_geodesics(events)
plot_with_null_geodesics(events, G, null_geodesic_edges)


# %%
class InteractionNet:
    """
    A reversible interaction net rule: swap (A,B) → (B,A) with logging for reversal
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.history = []  # to support reversibility

    def add_node(self, id, label):
        self.graph.add_node(id, label=label)

    def add_edge(self, src, dst):
        self.graph.add_edge(src, dst)

    def get_labels(self):
        return nx.get_node_attributes(self.graph, "label")

    def step(self):
        """
        Reversible interaction rule: swap labels of any connected pair (A,B) → (B,A)
        Store the operation to allow reversal.
        """
        for u, v in self.graph.edges:
            label_u = self.graph.nodes[u]["label"]
            label_v = self.graph.nodes[v]["label"]
            if (label_u, label_v) != (
                label_v,
                label_u,
            ):  # avoid swapping twice
                (
                    self.graph.nodes[u]["label"],
                    self.graph.nodes[v]["label"],
                ) = label_v, label_u
                self.history.append((u, v))
                return True
        return False

    def reverse(self):
        """Undo the last reversible interaction"""
        if self.history:
            u, v = self.history.pop()
            (
                self.graph.nodes[u]["label"],
                self.graph.nodes[v]["label"],
            ) = (
                self.graph.nodes[v]["label"],
                self.graph.nodes[u]["label"],
            )

    def draw(self, title=""):
        pos = nx.spring_layout(self.graph)
        labels = self.get_labels()
        plt.figure(figsize=(6, 4))
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color="lightgreen",
            node_size=800,
            font_size=10,
            arrows=True,
        )
        plt.title(title)
        plt.axis("off")
        plt.show()


# Create interaction net with reversible rule
net = InteractionNet()
net.add_node(1, "A")
net.add_node(2, "B")
net.add_node(3, "C")
net.add_edge(1, 2)
net.add_edge(2, 3)

# Initial state
net.draw("Initial Interaction Net")

# Apply one reversible step
net.step()
net.draw("After 1st Reversible Step")

# Apply another reversible step
net.step()
net.draw("After 2nd Reversible Step")

# Reverse last step
net.reverse()
net.draw("After Reversing Last Step")


# %% =================================
class ReversibleTuringNet(InteractionNet):
    def __init__(self):
        super().__init__()
        self.time = 0  # causal timestamp

    def add_tape_node(self, id, symbol):
        self.add_node(id, f"{symbol}@{self.time}")

    def reversible_head_step(self, head_id, direction=1):
        """
        Simulate a reversible head move:
        - Swap the head label with its neighbor in given direction
        - Log movement for reversal
        """
        neighbors = (
            list(self.graph.successors(head_id))
            if direction == 1
            else list(self.graph.predecessors(head_id))
        )
        if not neighbors:
            return False

        neighbor = neighbors[0]
        label_head = self.graph.nodes[head_id]["label"]
        label_neighbor = self.graph.nodes[neighbor]["label"]

        # Swap labels with time annotation
        self.time += 1
        new_label_head = label_neighbor.split("@")[0] + f"@{self.time}"
        new_label_neighbor = label_head.split("@")[0] + f"@{self.time}"

        self.graph.nodes[head_id]["label"] = new_label_head
        self.graph.nodes[neighbor]["label"] = new_label_neighbor
        self.history.append((head_id, neighbor, self.time))
        return True

    def reverse_head_step(self):
        """Undo the last reversible head step"""
        if not self.history:
            return
        head_id, neighbor, t = self.history.pop()
        label_head = self.graph.nodes[head_id]["label"]
        label_neighbor = self.graph.nodes[neighbor]["label"]

        # Restore prior state by undoing the swap and decrementing time
        old_label_head = label_neighbor.split("@")[0] + f"@{t - 1}"
        old_label_neighbor = label_head.split("@")[0] + f"@{t - 1}"
        self.graph.nodes[head_id]["label"] = old_label_head
        self.graph.nodes[neighbor]["label"] = old_label_neighbor
        self.time -= 1


# Create a reversible Turing head update net
rt_net = ReversibleTuringNet()
rt_net.add_tape_node(1, "H")  # Head
rt_net.add_tape_node(2, "1")
rt_net.add_tape_node(3, "0")
rt_net.add_edge(1, 2)
rt_net.add_edge(2, 3)

# Initial state
rt_net.draw("Turing Net: Initial Configuration")

start_time = time.time()
# Step 1: Move right (H <-> 1)
rt_net.reversible_head_step(1, direction=1)
rt_net.draw("After Step 1: Head moved right")
print(f"Time taken: {time.time() - start_time:.2f} seconds")
# Step 2: Move right again (H <-> 0)
rt_net.reversible_head_step(2, direction=1)
rt_net.draw("After Step 2: Head moved right again")

# Reverse step: Move back (H <-> 1)
rt_net.reverse_head_step()
rt_net.draw("After Reverse Step: Head moved back left")


# %% =================================
class StatefulTuringNet(ReversibleTuringNet):
    """
    A stateful Turing machine with tape writes and state transitions
    """

    def __init__(self):
        super().__init__()
        self.head_state = "q0"

    def add_tape_node(self, id, symbol):
        label = (
            f"{symbol}|{self.head_state}@{self.time}"
            if symbol == "H"
            else f"{symbol}@{self.time}"
        )
        self.add_node(id, label)

    def reversible_write_and_move(self, head_id, direction, new_symbol, next_state):
        neighbors = (
            list(self.graph.successors(head_id))
            if direction == 1
            else list(self.graph.predecessors(head_id))
        )
        if not neighbors:
            return False

        neighbor = neighbors[0]
        head_label = self.graph.nodes[head_id]["label"]
        neighbor_label = self.graph.nodes[neighbor]["label"]

        # Parse current head and state
        head_symbol, meta = (
            head_label.split("|") if "|" in head_label else (head_label, "")
        )
        current_state, _ = (
            meta.split("@") if "@" in meta else (self.head_state, self.time)
        )

        # Update time
        self.time += 1

        # Write new symbol to current cell and move head to neighbor
        new_head_label = f"{neighbor_label.split('@')[0]}|{next_state}@{self.time}"
        new_current_label = f"{new_symbol}@{self.time}"

        self.graph.nodes[head_id]["label"] = new_current_label
        self.graph.nodes[neighbor]["label"] = new_head_label
        self.head_state = next_state

        # Store all necessary info for reversal
        self.history.append(
            (
                head_id,
                neighbor,
                head_label,
                neighbor_label,
                current_state,
            )
        )
        return True

    def reverse_write_and_move(self):
        if not self.history:
            return

        # Unpack all 5 elements stored during the forward step
        (
            head_id,
            neighbor,
            old_head_label,
            old_neighbor_label,
            prev_state,
        ) = self.history.pop()

        # Undo state and labels
        self.graph.nodes[head_id]["label"] = old_head_label
        self.graph.nodes[neighbor]["label"] = old_neighbor_label
        self.time -= 1
        self.head_state = prev_state


# Instantiate and simulate write + transition steps
state_net = StatefulTuringNet()
state_net.add_tape_node(1, "H")  # Head starts here
state_net.add_tape_node(2, "1")
state_net.add_tape_node(3, "0")
state_net.add_edge(1, 2)
state_net.add_edge(2, 3)

# Step: write '1' and move right, change state
state_net.draw("Initial Config with State")
state_net.reversible_write_and_move(1, direction=1, new_symbol="1", next_state="q1")
state_net.draw("Step 1: Write '1', Move →, State q1")
state_net.reversible_write_and_move(2, direction=1, new_symbol="0", next_state="q2")
state_net.draw("Step 2: Write '0', Move →, State q2")

# Reverse steps
state_net.reverse_write_and_move()
state_net.draw("Reverse Step 1: Back to State q1")
state_net.reverse_write_and_move()
state_net.draw("Reverse Step 2: Back to Initial")


# %% =================================
class StatefulTuringAnimator:
    """
    Animate a stateful Turing machine with tape writes and state transitions
    """

    def __init__(self, net: StatefulTuringNet):
        self.net = net
        self.frames = []

    def record_frame(self, title):
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.spring_layout(self.net.graph, seed=42)
        labels = self.net.get_labels()
        nx.draw(
            self.net.graph,
            pos,
            with_labels=True,
            labels=labels,
            node_color="skyblue",
            node_size=900,
            font_size=10,
            ax=ax,
            arrows=True,
        )
        ax.set_title(title)
        ax.axis("off")
        self.frames.append(fig)

    def simulate_cycle(self):
        self.frames = []

        self.record_frame("Start: Head at H|q0@0")
        self.net.reversible_write_and_move(
            1, direction=1, new_symbol="1", next_state="q1"
        )
        self.record_frame("Step 1: Write 1, Move →, State q1")

        self.net.reversible_write_and_move(
            2, direction=1, new_symbol="0", next_state="q2"
        )
        self.record_frame("Step 2: Write 0, Move →, State q2")

        # Reverse with more frames
        self.net.reverse_write_and_move()
        self.record_frame("Reverse Step 1: ← Restore q1")
        self.record_frame("Reverse Step 1b: Stable state")

        self.net.reverse_write_and_move()
        self.record_frame("Reverse Step 2: ← Restore q0")
        self.record_frame("Final State: Original Recovered")

    def animate(self):
        self.simulate_cycle()
        fig_anim, ax = plt.subplots(figsize=(6, 4))
        ims = []

        for f in self.frames:
            canvas = f.canvas
            canvas.draw()
            image = np.asarray(canvas.renderer.buffer_rgba())
            im = plt.imshow(image, animated=True)
            ims.append([im])
            plt.close(f)

        ani = animation.ArtistAnimation(
            fig_anim, ims, interval=1000, repeat_delay=2000, blit=True
        )
        return ani


# Set up a fresh simulation and animate
state_net = StatefulTuringNet()
state_net.add_tape_node(1, "H")
state_net.add_tape_node(2, "1")
state_net.add_tape_node(3, "0")
state_net.add_edge(1, 2)
state_net.add_edge(2, 3)

state_animator = StatefulTuringAnimator(state_net)
ani = state_animator.animate()
plt.close()
# ani.save("./testing.gif", writer=PillowWriter(fps=1))


# %% =================================
class BinaryAdditionMinkowskiSim:
    """
    Simulate binary addition using tape and causal embedding
    """

    def __init__(self, a: str, b: str):
        self.a = a[::-1]  # reverse for LSB -> MSB
        self.b = b[::-1]
        self.result = ""
        self.events = []  # each step as (id, x, y, t, label)
        self.time = 0

    def simulate(self):
        """
        Simulate the binary addition
        """
        carry = 0
        for i in range(max(len(self.a), len(self.b))):
            digit_a = int(self.a[i]) if i < len(self.a) else 0
            digit_b = int(self.b[i]) if i < len(self.b) else 0
            total = digit_a + digit_b + carry
            out = total % 2
            carry = total // 2
            self.result += str(out)

            # Space (x), tape position (y), and time (t) embedding
            x, y, t = i, 0, self.time
            self.events.append(
                (
                    f"add_{i}",
                    x,
                    y,
                    t,
                    f"{digit_a}+{digit_b}+{carry}={out}",
                )
            )
            self.time += 1

        if carry:
            self.result += str(carry)
            self.events.append(
                (
                    "carry",
                    len(self.a),
                    0,
                    self.time,
                    "carry=1",
                )
            )
            self.time += 1

        return self.result[::-1]  # reverse back

    def visualize_minkowski(self):
        """
        Visualize the Minkowski spacetime embedding of the binary addition
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        for eid, x, y, t, label in self.events:
            ax.scatter(x, y, t, color="blue", s=50)
            ax.text(x, y, t + 0.2, label, fontsize=9)

            # Draw light cone
            r = 1
            cone_t = np.linspace(t, t + 1.5, 10)
            for dx in [-r, r]:
                for dy in [-r, r]:
                    ax.plot(
                        [x, x + dx],
                        [y, y + dy],
                        [t, t + 1.5],
                        color="red",
                        alpha=0.3,
                    )

        ax.set_xlabel("Tape Position (x)")
        ax.set_ylabel("Memory Dim (y)")
        ax.set_zlabel("Time (t)")
        ax.set_title("Minkowski Spacetime Embedding of Binary Addition")

        plt.tight_layout()
        plt.show()


# Simulate binary addition of two 4-bit numbers
simulator = BinaryAdditionMinkowskiSim("1011", "0110")  # 11 + 6 = 17 → 10001
sum_result = simulator.simulate()
simulator.visualize_minkowski()


# %% =================================
def minkowski_interval(p1, p2):
    """Compute Minkowski interval squared with signature (-++)."""
    t1, x1, y1 = p1[3], p1[1], p1[2]
    t2, x2, y2 = p2[3], p2[1], p2[2]
    return -((t2 - t1) ** 2) + (x2 - x1) ** 2 + (y2 - y1) ** 2


# Compute all pairwise intervals
pairs = []
for e1, e2 in combinations(simulator.events, 2):
    id1, id2 = e1[0], e2[0]
    interval_sqr = minkowski_interval(e1, e2)
    pairs.append(
        {
            "from": id1,
            "to": id2,
            "Δs² (Minkowski)": interval_sqr,
        }
    )

# Convert to DataFrame
df_intervals = pd.DataFrame(pairs)

# Display results
df_intervals


# %% =================================
def plot_causal_cones_with_arrows(events, null_threshold=1e-3):
    """
    Plot causal cones and reversible steps overlaid on the Minkowski space diagram
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for eid, x, y, t, label in events:
        ax.scatter(x, y, t, color="blue", s=60)
        ax.text(x, y, t + 0.2, label, fontsize=9)

    for e1, e2 in combinations(events, 2):
        s2 = minkowski_interval(e1, e2)
        if abs(s2) < null_threshold and e1[3] < e2[3]:
            x0, y0, t0 = e1[1], e1[2], e1[3]
            x1, y1, t1 = e2[1], e2[2], e2[3]
            ax.plot(
                [x0, x1],
                [y0, y1],
                [t0, t1],
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.8,
            )

    # Optional: draw faint light cones (future only)
    for eid, x, y, t, label in events:
        cone_t = np.linspace(t, t + 1.5, 10)
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                ax.plot(
                    [x, x + dx],
                    [y, y + dy],
                    [t, t + 1.5],
                    color="green",
                    alpha=0.2,
                )

    ax.set_xlabel("Tape Position (x)")
    ax.set_ylabel("Memory Dim (y)")
    ax.set_zlabel("Time (t)")
    ax.set_title("Causal Cones and Null Geodesics for Binary Addition Events")
    plt.tight_layout()
    plt.show()


plot_causal_cones_with_arrows(simulator.events)


# %% =================================
def plot_geodesics_with_memory(events, null_threshold=1e-3):
    """
    Plot null geodesics with memory state annotations
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each event
    for eid, x, y, t, label in events:
        ax.scatter(x, y, t, color="blue", s=70)
        ax.text(
            x,
            y,
            t + 0.2,  # z-coordinate
            label,
            fontsize=9,
            color="black",
        )

    # Draw null geodesic paths with memory state labels
    # Light cones for visual context
    for eid, x, y, t, label in events:
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                ax.plot(
                    [x, x + dx],
                    [y, y + dy],
                    [t, t + 1.5],
                    color="green",
                    alpha=0.15,
                )

    ax.set_xlabel("Tape Position (x)")
    ax.set_ylabel("Memory Dim (y)")
    ax.set_zlabel("Time (t)")
    ax.set_title("Null Geodesics with Symbolic Register/Carry State")
    plt.tight_layout()
    plt.show()


plot_geodesics_with_memory(simulator.events)

# %% [markdown]
"""
For any ordered pair $(p,q)$ with $t_q > t_p$:

| Relation | Condition on $\Delta s^2$ | Interpretation |
|----------|---------------------------|----------------|
| Reversible / null | $\Delta s^2 = 0$ | No information is lost |
| Irreversible / timelike | $\Delta s^2 < -\epsilon$ | Information-bearing; needs workspace to undo |
| Spacelike | $\Delta s^2 > 0$ | Independent / concurrent |

Here we took $\epsilon = 10^{-3}$.

2. Metric definition

Define
$$
d(p,q) = \begin{cases}
0 & \text{if } |\Delta s^2| \leq \epsilon \text{ (null)} \\
-\Delta s^2(p,q) & \text{if } \Delta s^2 < -\epsilon \text{ (timelike)} \\
\infty & \text{otherwise (not causally related)}
\end{cases}
$$

$d$ is pseudo-metric:

- Non-negative, $d(p,p)=0$.

- Symmetric only on the causal DAG's Hasse diagram (future ↔ past).

- Triangle inequality holds along any causal chain $p \prec r \prec q$:
$d(p,q) \leq d(p,r) + d(r,q)$ (additivity of proper time).

- Null links collapse to distance 0 $\Rightarrow$ hierarchical edges are "free".

- Timelike links accrue irreversibility-cost equal to proper time $-\Delta s^2$.

3. Cost table (generated above)

Open the interactive sheet "Computation Metric Space (costs)" to inspect every pair's
$\Delta s^2$, relation type, and irreversible cost.
"""


# %% =================================
# Compute classification of pairs
def classify_intervals(events, null_threshold=1e-3):
    rows = []
    for e1, e2 in combinations(events, 2):
        if e1[3] >= e2[3]:
            continue  # we only consider forward-in-time relations
        s2 = minkowski_interval(e1, e2)
        if abs(s2) < null_threshold:
            relation = "reversible-null"
            cost = 0.0
        elif s2 < -null_threshold:
            relation = "timelike-irreversible"
            cost = np.sqrt(-s2)  # proper-time cost
        else:
            relation = "spacelike"
            cost = np.nan
        rows.append(
            {
                "from": e1[0],
                "to": e2[0],
                "Δs²": s2,
                "relation": relation,
                "cost": cost,
            }
        )
    return pd.DataFrame(rows)


df_costs = classify_intervals(simulator.events)
print("Computation Metric Space (costs)")
print(df_costs)


# %% =================================
class BinaryIncrementSimulator:
    """
    Simulate binary increment (e.g., 0111 -> 1000) with some irreversible flushes (3)
    """

    def __init__(self, bits: str):
        self.bits = list(bits[::-1])  # LSB first
        self.events = []
        self.time = 0
        self.irrev_flag = False  # Track irreversible steps

    def step(self, idx, symbol, irreversible=False):
        x, y, t = idx, 0, self.time
        label = f"{'WRITE' if not irreversible else 'FLUSH'}:{symbol}"
        self.events.append((f"ev_{idx}_{t}", x, y, t, label))
        self.time += 1

    def simulate_increment(self):
        carry = 1
        for i in range(len(self.bits)):
            if self.bits[i] == "0":
                self.bits[i] = "1"
                self.step(i, "1")
                carry = 0
                break
            else:
                self.bits[i] = "0"
                self.step(
                    i, "0", irreversible=True
                )  # irreversible clear due to overflow
        if carry:
            self.bits.append("1")  # extend width
            self.step(len(self.bits) - 1, "1")

        return "".join(self.bits[::-1])


incrementer = BinaryIncrementSimulator("0111")  # 0111 -> 1000
new_binary = incrementer.simulate_increment()

df_increment_costs = classify_intervals(incrementer.events)
print("Increment Metric Space")
print(df_increment_costs)


# %%
def visualize_metric_space(events, df_costs):
    """
    Visualize the metric space with irreversible steps highlighted
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot events
    for eid, x, y, t, label in events:
        ax.scatter(x, y, t, color="blue", s=60)
        ax.text(x, y, t + 0.2, label, fontsize=9)

    # Plot edges with classification
    for _, row in df_costs.iterrows():
        e1 = next(e for e in events if e[0] == row["from"])
        e2 = next(e for e in events if e[0] == row["to"])
        x0, y0, t0 = e1[1], e1[2], e1[3]
        x1, y1, t1 = e2[1], e2[2], e2[3]
        color = {
            "reversible-null": "green",
            "timelike-irreversible": "#000",
            "spacelike": "gray",
        }.get(row["relation"], "black")

        ax.plot(
            [x0, x1],
            [y0, y1],
            [t0, t1],
            color=color,
            linestyle="--" if row["relation"] == "reversible-null" else "-",
            linewidth=2,
            alpha=0.8,
        )

    ax.set_xlabel("Tape Position (x)")
    ax.set_ylabel("Memory (y)")
    ax.set_zlabel("Time (t)")
    ax.set_title("Computation Metric Space (3D Visualization)")
    plt.tight_layout()
    plt.show()


visualize_metric_space(incrementer.events, df_increment_costs)
print("The dotted lines are reversible null links.")


# %% =================================
class BinaryIncrementWithEntropy:
    """
    Simulate increment with irreversible steps (e.g. scratch writes, unlinked resets)
    """

    def __init__(self, bits: str):
        self.bits = list(bits[::-1])  # LSB first
        self.events = []
        self.time = 0

    def step(self, idx, symbol, irreversible=False, y=0):
        x, t = idx, self.time
        label = f"{'FLUSH' if irreversible else 'WRITE'}:{symbol}"
        self.events.append((f"ev_{idx}_{t}", x, y, t, label))
        self.time += 1

    def simulate(self):
        carry = 1
        for i in range(len(self.bits)):
            if self.bits[i] == "0":
                self.bits[i] = "1"
                self.step(i, "1")  # reversible write
                carry = 0
                break
            else:
                self.bits[i] = "0"
                self.step(i, "0", irreversible=True)  # irreversible clear

        if carry:
            self.bits.append("1")
            self.step(len(self.bits) - 1, "1")

        self.step(idx=99, symbol="*", irreversible=True, y=1)

        return "".join(self.bits[::-1])


entropic_sim = BinaryIncrementWithEntropy("0111")
final_output = entropic_sim.simulate()

df_entropy = classify_intervals(entropic_sim.events)
print("Metric Space with Entropy Hotspots")
print(df_entropy)
