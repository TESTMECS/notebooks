# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# %%
# Create a simple heap-manipulating program: a linked list stack with push and pop
# Abstractly represent it as a list

# We'll show both:
# 1. Concrete Heap (memory addresses and pointers)
# 2. Abstract Stack (clean list)

# Define concrete heap (address -> value)
concrete_heap = {"ℓ": "0x1", "0x1": ("A", "0x2"), "0x2": ("B", "null")}

# Define abstract stack
abstract_stack = ["A", "B"]

# Build concrete heap graph
G_heap = nx.DiGraph()
G_heap.add_node("ℓ", label="ℓ", color="lightblue")
G_heap.add_node("0x1", label="0x1: (A, 0x2)", color="lightgreen")
G_heap.add_node("0x2", label="0x2: (B, null)", color="lightgreen")

G_heap.add_edge("ℓ", "0x1")
G_heap.add_edge("0x1", "0x2")

# Build abstract stack graph
G_stack = nx.DiGraph()
for i, val in enumerate(abstract_stack):
    G_stack.add_node(val, label=val, color="lightcoral")
    if i > 0:
        G_stack.add_edge(abstract_stack[i - 1], val)

# Plot both
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot concrete heap
pos_heap = nx.spring_layout(G_heap, seed=1)
colors = [G_heap.nodes[n]["color"] for n in G_heap.nodes()]
labels = {n: G_heap.nodes[n]["label"] for n in G_heap.nodes()}
nx.draw(
    G_heap,
    pos_heap,
    with_labels=True,
    labels=labels,
    node_color=colors,
    ax=axes[0],
    node_size=1500,
)
axes[0].set_title("Concrete Heap (Linked List)")

# Plot abstract stack
pos_stack = nx.spring_layout(G_stack, seed=2)
colors = [G_stack.nodes[n]["color"] for n in G_stack.nodes()]
labels = {n: G_stack.nodes[n]["label"] for n in G_stack.nodes()}
nx.draw(
    G_stack,
    pos_stack,
    with_labels=True,
    labels=labels,
    node_color=colors,
    ax=axes[1],
    node_size=1500,
)
axes[1].set_title("Abstract Stack (List)")

plt.tight_layout()
plt.savefig('heapAbstract/heap_vs_stack.png', dpi=150, bbox_inches='tight')
plt.close()  # Close instead of show
print("✅ Saved heap vs stack visualization")

# %%[markdown]
"""
The authors use a heap predicate stk(s) that says:
The heap must contain a pointer at location ℓ to the start of a valid linked list.
That linked list encodes the values in s.
If the stack is empty, then ℓ must point to null.
If the stack v :: s then p must point to pair (v, q)
where q is the next pointer in the list.
"""


# %%
# pi(n)
def project_heap_to_list(heap, l_ptr):
    """Project a heap-based linked list into a Python list, or return '✗' if invalid."""
    visited = set()
    result = []

    current = heap.get(l_ptr)
    if current is None:
        return "✗"  # ℓ doesn't point to anything

    while current != "null":
        if current in visited:
            return "✗"  # cycle detected
        visited.add(current)

        node = heap.get(current)
        if not isinstance(node, tuple) or len(node) != 2:
            return "✗"  # not a valid (value, next) pair

        value, next_ptr = node
        if value is None or next_ptr is None:
            return "✗"  # malformed node
        result.append(value)
        current = next_ptr

    return result


# Example heap
heap_example = {"ℓ": "0x1", "0x1": ("A", "0x2"), "0x2": ("B", "null")}

# Project this heap
projected_list = project_heap_to_list(heap_example, "ℓ")
print(f"Projected heap to list: {projected_list}")

# Simulated attention matrix for 6 tokens (e.g., "The boy who the girl saw ran")
tokens = ["The", "boy", "who", "the", "girl", "ran"]

# Simulated attention matrix (rows = queries, cols = keys)
# Let's simulate a pattern where:
# - "boy" attends to "The"
# - "who" attends to "boy"
# - "the" attends to "who"
# - "girl" attends to "the"
# - "saw" (not present) is elided; instead "ran" attends back to "boy"
attention_matrix = np.array(
    [
        [0.1, 0.2, 0.1, 0.05, 0.05, 0.0],  # The
        [0.8, 0.1, 0.05, 0.025, 0.025, 0.0],  # boy → The
        [0.05, 0.8, 0.1, 0.025, 0.025, 0.0],  # who → boy
        [0.025, 0.025, 0.6, 0.1, 0.05, 0.0],  # the → who
        [0.025, 0.025, 0.1, 0.7, 0.05, 0.0],  # girl → the
        [0.4, 0.5, 0.05, 0.025, 0.025, 0.0],  # ran → The/boy
    ]
)

# Build graph by thresholding attention
threshold = 0.5
G = nx.DiGraph()
G.add_nodes_from(tokens)

edges = []
for i, query in enumerate(tokens):
    for j, key in enumerate(tokens):
        if attention_matrix[i, j] > threshold:
            G.add_edge(tokens[i], tokens[j], weight=attention_matrix[i, j])
            edges.append((tokens[i], tokens[j]))

# Plot attention heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    attention_matrix,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="Blues",
    annot=True,
    fmt=".2f",
)
plt.title("Simulated Attention Matrix")
plt.xlabel("Key")
plt.ylabel("Query")
plt.tight_layout()
plt.savefig('heapAbstract/attention_matrix.png', dpi=150, bbox_inches='tight')
plt.close()  # Close instead of show
print("✅ Saved attention matrix visualization")

# Plot attention graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightyellow",
    edge_color="gray",
    node_size=2000,
    font_size=12,
)
nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle="-|>", arrowsize=20, width=2)
plt.title("Projected Attention Graph (Top-k Structure)")
plt.savefig('heapAbstract/attention_graph.png', dpi=150, bbox_inches='tight')
plt.close()  # Close instead of show
print("✅ Saved attention graph visualization")

# Analyze if this structure is a tree (i.e., no cycles, connected from a root)
is_dag = nx.is_directed_acyclic_graph(G)
is_tree = nx.is_arborescence(G)  # rooted tree
print(f"Initial attention graph - Is DAG? {is_dag}, Is tree? {is_tree}")


def project_until_convergence(
    pairs, spatial, time_vec, phi_model, eps1=1e-5, max_passes=10000
):
    """
    Adjusts time_vec so that all parent-child pairs satisfy ds^2 < -eps1.
    The phi_model introduces a gravity-aware time distortion.
    """
    num_passes = 0
    converged = False
    
    # Get device from phi_model
    device = next(phi_model.parameters()).device

    time_vec = time_vec.clone().detach().requires_grad_(False).to(device)
    spatial = spatial.clone().detach().requires_grad_(False).to(device)

    while not converged and num_passes < max_passes:
        num_passes += 1
        converged = True

        for i, j in pairs:
            t_i, t_j = time_vec[i].item(), time_vec[j].item()  # Convert to scalars
            x_i, x_j = spatial[i], spatial[j]
            delta_t = t_i - t_j
            delta_x = x_i - x_j
            dx2 = torch.sum(delta_x**2).item()

            coords_i = torch.cat([torch.tensor([t_i]).to(device), x_i])
            phi_i = phi_model(coords_i.unsqueeze(0))[0].item()

            ds2 = -phi_i * delta_t**2 + dx2

            if ds2 >= -eps1:
                # Adjust time_i to be slightly later than time_j
                new_delta_t = torch.sqrt(torch.tensor((dx2 + eps1) / phi_i + 1e-9)).item()
                time_vec[i] = t_j + new_delta_t
                converged = False

    return time_vec


def π_attn(attn_matrix, tokens, spatial, time_vec, phi_model, eps1=1e-5):
    """
    Project an attention matrix into a DAG structure using Minkowski space conditions.
    If the result is not a DAG, return ✗.
    """
    # Threshold the attention matrix to get strong edges
    threshold = 0.5
    edges = [
        (i, j)
        for i in range(len(tokens))
        for j in range(len(tokens))
        if attn_matrix[i, j] > threshold and i != j  # Avoid self-loops
    ]

    # Apply Minkowski projection to enforce ds^2 < -eps1 for each edge
    time_vec_projected = project_until_convergence(
        edges, spatial, time_vec, phi_model, eps1
    )

    # Build a directed graph with adjusted time order
    G = nx.DiGraph()
    G.add_nodes_from(tokens)

    for i, j in edges:
        if time_vec_projected[i] < time_vec_projected[j]:
            G.add_edge(tokens[i], tokens[j])
        else:
            G.add_edge(tokens[j], tokens[i])  # enforce causal order

    # Check DAG property
    if nx.is_directed_acyclic_graph(G):
        return G
    else:
        return "✗"


class DummyPhi(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # ensures phi > 0
        )

    def forward(self, coords):
        """
        coords: Tensor of shape (batch_size, 4) where columns are [t, x, y, z] or equivalent
        returns: Scalar phi > 0 per point
        """
        return self.net(coords).squeeze(-1)  # (batch_size,)


def train_phi_from_ds2_violations(
    phi_model, pairs, spatial, time_vec, eps1=1e-5, num_epochs=200, lr=1e-4
):
    """
    Trains phi_model to reduce ds^2 >= -eps1 violations over a hierarchy of (child, parent) pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi_model.to(device)
    optimizer = torch.optim.Adam(phi_model.parameters(), lr=lr)
    spatial = spatial.to(device)
    time_vec = time_vec.to(device)

    print(f"Training phi model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        phi_model.train()

        for i, j in pairs:
            # Convert tensor elements to scalars safely
            t_i, t_j = time_vec[i].item(), time_vec[j].item()
            x_i, x_j = spatial[i], spatial[j]

            delta_t = t_i - t_j
            delta_x = x_i - x_j
            dx2 = torch.sum(delta_x**2)

            coords_i = torch.cat([torch.tensor([t_i]).to(device), x_i])
            phi_i = phi_model(coords_i.unsqueeze(0))[0]
            ds2 = -phi_i * delta_t**2 + dx2

            loss = F.relu(ds2 + eps1)  # penalize causal violations only
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Avg phi loss: {total_loss / len(pairs):.6f}"
            )

    print("Training completed!")
    return phi_model


def draw_attention_graph(tokens, edges, title, is_valid=True):
    G = nx.DiGraph()
    G.add_nodes_from(tokens)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)
    node_color = "lightgreen" if is_valid else "salmon"
    edge_color = "green" if is_valid else "red"

    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_color,
        edge_color=edge_color,
        node_size=2000,
        font_size=12,
        arrows=True,
        arrowsize=20,
        width=2,
    )
    plt.title(title)
    plt.axis("off")
    # Save with filename based on title
    filename = title.replace("✅", "valid").replace("❌", "invalid").replace(" ", "_").lower()
    filename = f'heapAbstract/{filename}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of show
    print(f"✅ Saved {filename}")
    
    return G


# Placeholder inputs for demo (simulate BERT-like embedding setup)
num_tokens = 6
tokens = ["The", "boy", "who", "the", "girl", "ran"]
attn_matrix = attention_matrix  # from previous simulation

# Simulated 3D spatial embeddings (like PCA on BERT embeddings)
spatial = torch.randn(num_tokens, 3)
time_vec = torch.zeros(num_tokens)

# Create edges list with integer indices for training
edges_indices = [
    (i, j)
    for i in range(len(tokens))
    for j in range(len(tokens))
    if attn_matrix[i, j] > 0.5 and i != j
]

print(f"Found {len(edges_indices)} strong attention edges")

# Initialize and train phi model
phi_model = DummyPhi()
if len(edges_indices) > 0:
    train_phi_from_ds2_violations(phi_model, edges_indices, spatial, time_vec)

    # Run the π_attn projection
    projected_structure = π_attn(attn_matrix, tokens, spatial, time_vec, phi_model)
    print(f"Projected structure result: {type(projected_structure)}")
    
    if projected_structure != "✗":
        print("✅ Successfully projected attention to valid DAG!")
        # Convert networkx edges to list for visualization
        projected_edges = list(projected_structure.edges())
        draw_attention_graph(tokens, projected_edges, "✅ Valid Projected Attention DAG", is_valid=True)
    else:
        print("❌ Could not project to valid DAG")
        # Show original problematic structure
        original_edges = [(tokens[i], tokens[j]) for i, j in edges_indices]
        draw_attention_graph(tokens, original_edges, "❌ Original Attention (Invalid)", is_valid=False)
else:
    print("No strong attention edges found for training")

# Example visualizations with different structures
print("\n" + "="*50)
print("EXAMPLE STRUCTURES")
print("="*50)

# Example 1: Valid DAG (tree-like attention)
edges_valid = [
    ("boy", "The"),
    ("who", "boy"),
    ("the", "who"),
    ("girl", "the"),
    ("ran", "boy"),
]

# Example 2: Invalid attention (cycle present)
edges_invalid = [
    ("boy", "The"),
    ("who", "boy"),
    ("the", "who"),
    ("girl", "the"),
    ("The", "girl"),  # introduces a cycle
]

# Visualize both examples
G_valid = draw_attention_graph(tokens, edges_valid, "✅ Example Valid Attention DAG", is_valid=True)
G_invalid = draw_attention_graph(tokens, edges_invalid, "❌ Example Invalid Attention (Cycle)", is_valid=False)

print(f"Valid example - Is DAG? {nx.is_directed_acyclic_graph(G_valid)}")
print(f"Invalid example - Is DAG? {nx.is_directed_acyclic_graph(G_invalid)}")
# %%
def analyze_and_visualize_attention(attn_matrix, tokens, threshold=0.5):
    """
    Analyze and visualize whether an attention matrix projects to a DAG.
    Returns the result (True/False) and shows heatmap + attention graph.
    """
    num_tokens = len(tokens)
    edges = []

    # Extract edges above threshold
    for i in range(num_tokens):
        for j in range(num_tokens):
            if attn_matrix[i, j] > threshold:
                edges.append((tokens[i], tokens[j]))

    # Build graph
    G = nx.DiGraph()
    G.add_nodes_from(tokens)
    G.add_edges_from(edges)

    # Check for DAG
    is_dag = nx.is_directed_acyclic_graph(G)

    # Plot heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens,
                cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Attention Heatmap")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    
    # Save heatmap with descriptive filename
    heatmap_filename = f"heapAbstract/attention_heatmap_{'valid' if is_dag else 'invalid'}.png"
    plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {heatmap_filename}")

    # Plot graph
    draw_attention_graph(tokens, edges,
                         "✅ Valid Attention DAG" if is_dag else "❌ Invalid Attention (Cycle Detected)",
                         is_valid=is_dag)
    
    return is_dag

# Simulate two matrices
attn_valid = np.array([
    [0.1, 0.6, 0.1, 0.1, 0.05, 0.05],
    [0.7, 0.1, 0.1, 0.05, 0.025, 0.025],
    [0.05, 0.8, 0.1, 0.025, 0.025, 0.0],
    [0.025, 0.025, 0.8, 0.1, 0.05, 0.0],
    [0.025, 0.025, 0.1, 0.8, 0.05, 0.0],
    [0.7, 0.2, 0.05, 0.025, 0.025, 0.0],
])

attn_invalid = np.copy(attn_valid)
attn_invalid[0, 4] = 0.6  # introduces a backward attention that creates a cycle

# Token list
tokens = ["The", "boy", "who", "the", "girl", "ran"]

# Analyze both
print("Valid Case:")
valid_result = analyze_and_visualize_attention(attn_valid, tokens)

print("Invalid Case:")
invalid_result = analyze_and_visualize_attention(attn_invalid, tokens)

# Define utility functions (from previous cells)
# analyze_and_visualize_attention(...)
# draw_attention_graph(...)

# Load BERT and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

# Tokenize sentence
sentence = "The boy who the girl saw ran."
inputs = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Use last layer, first head
attn = outputs.attentions[-1][0, 0]  # [batch, head, Q, K]
attn_np = attn.numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Visualize & check DAG
is_valid = analyze_and_visualize_attention(attn_np, tokens, threshold=0.2)
print("Is attention a DAG?", is_valid)

class MinkowskiAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.n_heads = n_heads
        self.scale = (embed_dim // n_heads) ** -0.5

    def forward(self, x, time_coords, spatial_coords):
        B, T, D = x.shape
        H = self.n_heads

        Q = self.q_proj(x).view(B, T, H, -1)
        K = self.k_proj(x).view(B, T, H, -1)
        V = self.v_proj(x).view(B, T, H, -1)

        # Compute pairwise Minkowski distances
        t_i = time_coords.unsqueeze(1)  # (B, 1, T, 1)
        t_j = time_coords.unsqueeze(2)  # (B, T, 1, 1)
        delta_t = t_i - t_j

        x_i = spatial_coords.unsqueeze(1)  # (B, 1, T, d)
        x_j = spatial_coords.unsqueeze(2)  # (B, T, 1, d)
        delta_x = x_i - x_j
        dx2 = (delta_x ** 2).sum(-1, keepdim=True)  # (B, T, T, 1)

        ds2 = -delta_t ** 2 + dx2  # Minkowski interval

        causal_mask = (ds2 < 0).float()  # (B, T, T, 1)

        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, T, H, T)
        attn_scores = attn_scores.permute(0, 2, 1, 3)  # (B, H, T, T)

        # Apply soft causal penalty
        penalty = F.relu(ds2.squeeze(-1))  # (B, T, T)
        attn_scores = attn_scores - penalty.unsqueeze(1)  # Broadcast to heads

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = (attn_weights @ V.permute(0, 2, 1, 3))  # (B, H, T, d)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)

        return self.out_proj(attn_out), attn_weights


# --- Simple demo setup ---
B, T, D = 1, 6, 32
x = torch.randn(B, T, D)
time_coords = torch.linspace(0, 1, T).unsqueeze(0).repeat(B, 1)  # (B, T)
spatial_coords = torch.randn(B, T, 2)  # (B, T, 2)

attn_layer = MinkowskiAttention(embed_dim=D, n_heads=4)
out, weights = attn_layer(x, time_coords, spatial_coords)
print(out.shape)
print(weights.shape)

# --- Visualize attention weights ---
plt.imshow(weights[0, 0].detach(), cmap="plasma")
plt.colorbar()
plt.title("Minkowski-Causal Attention (Head 0)")
plt.xlabel("Key Index")
plt.ylabel("Query Index")
plt.show()
