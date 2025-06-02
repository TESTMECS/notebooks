# %%
# 🧪 AI Physics Sandbox: Causal Attention as a Physical System
# ------------------------------------------------------------
# This prototype treats attention events as particles in Minkowski space.
# Each attention flow (edge) is a causal interaction with geometric energy.
# We use interaction-net-style updates and simulate step-wise evolution.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import networkx as nx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

console = Console()

# %%
# === Step 0: Initialize Events as Nodes ===
N = 24  # number of events (can be attention heads or positions)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

# Trefoil-inspired embedding
x = np.sin(theta) + 2 * np.sin(2 * theta)
y = np.cos(theta) - 2 * np.cos(2 * theta)
z = -np.sin(3 * theta) + np.linspace(0, 2, N)  # time

positions = np.stack([x, y, z], axis=1)

# === Step 1: Build Initial Causal Graph ===
G = nx.DiGraph()
for i in range(N):
    G.add_node(i, pos=positions[i], label=f"e{i}")

rprint(
    Panel(
        f"[bold blue]🌌 Initializing {N} spacetime events[/bold blue]",
        style="blue",
    )
)

# %%
# Define causal edges based on Δt > 0 and Δs² < 0
edge_count = 0
for i in range(N):
    for j in range(i + 1, N):
        p1, p2 = positions[i], positions[j]
        dx, dy, dt = p2 - p1
        ds2 = -(dt**2) + dx**2 + dy**2
        if ds2 < -1e-2:
            G.add_edge(i, j, energy=np.sqrt(-ds2))
            edge_count += 1

rprint(f"[green]✨ Created {edge_count} causal connections[/green]")


# %%
# === Step 2: Define Energy and Interaction Rule ===
def total_energy(graph):
    return sum(
        data["energy"] for _, _, data in graph.edges(data=True)
    )


def interaction_step(graph):
    """Simulate a rule: merge shortest edge, redistribute its energy."""
    if graph.number_of_edges() == 0:
        return graph.copy()

    # Filter edges to only include those with nodes that actually exist
    valid_edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if graph.has_node(u) and graph.has_node(v)
    ]

    if not valid_edges:
        return graph.copy()

    min_edge = min(valid_edges, key=lambda e: e[2]["energy"])
    u, v, data = min_edge

    # Double-check nodes exist before proceeding
    if not (graph.has_node(u) and graph.has_node(v)):
        return graph.copy()

    # New node replaces u and v
    existing_nodes = list(graph.nodes())
    new_idx = max(existing_nodes) + 1 if existing_nodes else 0

    pos_u, pos_v = graph.nodes[u]["pos"], graph.nodes[v]["pos"]
    new_pos = 0.5 * (pos_u + pos_v)

    H = graph.copy()

    # Store neighbors before removing nodes
    u_neighbors = list(H.predecessors(u)) + list(H.successors(u))
    v_neighbors = list(H.predecessors(v)) + list(H.successors(v))

    # Remove nodes safely
    if H.has_node(u):
        H.remove_node(u)
    if H.has_node(v):
        H.remove_node(v)

    H.add_node(new_idx, pos=new_pos, label=f"merge({u},{v})")

    # Reconnect: create new edges from/to the merged node
    all_neighbors = set(u_neighbors + v_neighbors) - {u, v}

    for n in all_neighbors:
        if H.has_node(n):  # Make sure neighbor still exists
            # Calculate new edge energy
            dx, dy, dt = H.nodes[n]["pos"] - new_pos
            ds2 = -(dt**2) + dx**2 + dy**2
            if ds2 < -1e-2:
                energy = np.sqrt(-ds2)
                # Add edge in both directions if causal
                if dt > 0:  # n -> new_idx
                    H.add_edge(n, new_idx, energy=energy)
                elif dt < 0:  # new_idx -> n
                    H.add_edge(new_idx, n, energy=energy)

    return H


# %%
# === Step 3: Run Evolution ===
states = [G]
energy_history = [total_energy(G)]  # Track energy over time
edge_count_history = [G.number_of_edges()]  # Track edge count
node_count_history = [G.number_of_nodes()]  # Track node count
steps = 8

rprint(
    Panel(
        "[bold yellow]🔄 Running causal evolution simulation[/bold yellow]",
        style="yellow",
    )
)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task("Evolving spacetime...", total=steps)

    for step in range(steps):
        old_nodes = G.number_of_nodes()
        old_edges = G.number_of_edges()
        old_energy = total_energy(G)

        G = interaction_step(G)
        states.append(G)

        # Track evolution metrics
        new_energy = total_energy(G)
        energy_history.append(new_energy)
        edge_count_history.append(G.number_of_edges())
        node_count_history.append(G.number_of_nodes())

        new_nodes = G.number_of_nodes()
        new_edges = G.number_of_edges()

        rprint(
            f"[dim]Step {step + 1}: {old_nodes}→{new_nodes} nodes, "
            f"{old_edges}→{new_edges} edges, "
            f"energy: {old_energy:.3f}→{new_energy:.3f}[/dim]"
        )

        progress.advance(task)

        if G.number_of_edges() == 0:
            rprint("[red]⚠️  No more edges to evolve[/red]")
            break

# %%
# === Step 4: Create Summary Table ===
table = Table(title="🌌 Spacetime Evolution Summary")
table.add_column("Step", style="cyan", no_wrap=True)
table.add_column("Nodes", style="magenta")
table.add_column("Edges", style="yellow")
table.add_column("Total Energy", style="green")

for i, state in enumerate(states):
    energy = total_energy(state)
    table.add_row(
        str(i),
        str(state.number_of_nodes()),
        str(state.number_of_edges()),
        f"{energy:.4f}" if energy > 0 else "0.0000",
    )

console.print(table)


# %%
# === Step 5: Visualize Final State ===
def plot_graph_3d(graph, title="Causal Net"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    pos = nx.get_node_attributes(graph, "pos")

    # Plot edges with energy-based coloring
    edge_energies = [
        data["energy"] for _, _, data in graph.edges(data=True)
    ]
    if edge_energies:
        max_energy = max(edge_energies)
        for u, v, data in graph.edges(data=True):
            p1, p2 = pos[u], pos[v]
            intensity = data["energy"] / max_energy
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=(1, 1 - intensity, 1 - intensity),
                alpha=0.7,
                linewidth=2,
            )

    # Plot nodes
    for i, (x, y, z) in pos.items():
        ax.scatter(x, y, z, color="black", alpha=0.8)
        ax.text(
            x, y, z, graph.nodes[i]["label"], fontsize=10, alpha=0.9
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (space)")
    ax.set_ylabel("Y (space)")
    # Note: 3D axis doesn't have set_zlabel, using set_title instead
    plt.tight_layout()
    plt.show()


rprint(
    Panel(
        "[bold green]📊 Generating visualization plots[/bold green]",
        style="green",
    )
)

plot_graph_3d(states[0], title="🌟 Initial Causal Energy Flow")
if len(states) > 1:
    plot_graph_3d(
        states[-1], title="🔮 Evolved System with Merged Interactions"
    )

rprint(
    Panel(
        f"[bold blue]✅ Simulation complete! Final state: {states[-1].number_of_nodes()} nodes, {states[-1].number_of_edges()} edges[/bold blue]",
        style="blue",
    )
)


# %%
# === Step 6: Energy Evolution Animation ===
def animate_energy_evolution(
    energy_history,
    edge_count_history,
    node_count_history,
    save_path="energy_evolution.gif",
):
    """Create animated plot showing energy, edges, and nodes changing over time"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(12, 8)
    )

    steps = range(len(energy_history))

    def animate(frame):
        # Clear all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()

        current_steps = steps[: frame + 1]
        current_energy = energy_history[: frame + 1]
        current_edges = edge_count_history[: frame + 1]
        current_nodes = node_count_history[: frame + 1]

        # Energy evolution plot
        ax1.plot(
            current_steps,
            current_energy,
            "r-o",
            linewidth=2,
            markersize=6,
        )
        ax1.set_title(
            f"Total Energy Evolution (Step {frame})",
            fontweight="bold",
        )
        ax1.set_xlabel("Evolution Step")
        ax1.set_ylabel("Total Energy")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(energy_history) - 1)
        ax1.set_ylim(0, max(energy_history) * 1.1)

        # Edge count evolution
        ax2.plot(
            current_steps,
            current_edges,
            "b-s",
            linewidth=2,
            markersize=6,
        )
        ax2.set_title(
            f"Edge Count Evolution (Step {frame})", fontweight="bold"
        )
        ax2.set_xlabel("Evolution Step")
        ax2.set_ylabel("Number of Edges")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, len(edge_count_history) - 1)
        ax2.set_ylim(0, max(edge_count_history) * 1.1)

        # Node count evolution
        ax3.plot(
            current_steps,
            current_nodes,
            "g-^",
            linewidth=2,
            markersize=6,
        )
        ax3.set_title(
            f"Node Count Evolution (Step {frame})", fontweight="bold"
        )
        ax3.set_xlabel("Evolution Step")
        ax3.set_ylabel("Number of Nodes")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, len(node_count_history) - 1)
        ax3.set_ylim(0, max(node_count_history) * 1.1)

        # Energy efficiency (Energy per Edge)
        efficiency = [
            e / c if c > 0 else 0
            for e, c in zip(current_energy, current_edges)
        ]
        ax4.plot(
            current_steps,
            efficiency,
            "m-d",
            linewidth=2,
            markersize=6,
        )
        ax4.set_title(
            f"Energy Efficiency (Step {frame})", fontweight="bold"
        )
        ax4.set_xlabel("Evolution Step")
        ax4.set_ylabel("Energy per Edge")
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, len(energy_history) - 1)
        if efficiency:
            ax4.set_ylim(
                0, max(efficiency) * 1.1 if max(efficiency) > 0 else 1
            )

        plt.tight_layout()

    rprint(f"[cyan]🎬 Creating energy evolution animation...[/cyan]")
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(energy_history),
        interval=1000,
        repeat=True,
        blit=False,
    )

    # Save as GIF
    rprint(f"[yellow]💾 Saving animation to {save_path}...[/yellow]")
    ani.save(save_path, writer="pillow", fps=1)
    rprint(f"[green]✅ Animation saved successfully![/green]")

    plt.show()
    return ani


# %%
# === Step 7: 3D Network Evolution Animation ===
def animate_network_evolution(
    states, save_path="network_evolution.gif"
):
    """Animate the 3D network structure changing over time"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw_state(state, step):
        ax.clear()
        pos = nx.get_node_attributes(state, "pos")

        # Plot edges with energy-based coloring
        edge_energies = [
            data["energy"] for _, _, data in state.edges(data=True)
        ]
        if edge_energies and len(edge_energies) > 0:
            max_energy = max(edge_energies)
            for u, v, data in state.edges(data=True):
                p1, p2 = pos[u], pos[v]
                intensity = data["energy"] / max_energy
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=(1, 1 - intensity, 1 - intensity),
                    alpha=0.7,
                    linewidth=2,
                )

        # Plot nodes
        for i, (x, y, z) in pos.items():
            ax.scatter(x, y, z, color="black")
            ax.text(x, y, z, state.nodes[i]["label"], fontsize=8)

        ax.set_title(
            f"Causal Network Evolution - Step {step}\n"
            f"Nodes: {state.number_of_nodes()}, Edges: {state.number_of_edges()}, "
            f"Energy: {total_energy(state):.2f}"
        )
        ax.set_xlim((-4, 4))
        ax.set_ylim((-4, 4))
        ax.set_zlim((-2, 4))
        ax.set_xlabel("X (space)")
        ax.set_ylabel("Y (space)")
        ax.set_zlabel("Z (time)")

    def update(frame):
        draw_state(states[frame], frame)
        return []

    rprint(
        f"[cyan]🎬 Creating 3D network evolution animation...[/cyan]"
    )
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=1500,
        blit=False,
        repeat=True,
    )

    # Save as GIF
    rprint(
        f"[yellow]💾 Saving 3D animation to {save_path}...[/yellow]"
    )
    ani.save(save_path, writer="pillow", fps=1)
    rprint(f"[green]✅ 3D Animation saved successfully![/green]")

    plt.show()
    return ani


# Create both animations
rprint(
    Panel(
        "[bold magenta]🎭 Creating animated visualizations[/bold magenta]",
        style="magenta",
    )
)

# Energy evolution animation
energy_ani = animate_energy_evolution(
    energy_history, edge_count_history, node_count_history
)

# 3D network evolution animation
network_ani = animate_network_evolution(states)


# %%
# === Step 8: Physics Loss Function ===
def physics_loss(model_edges, true_edges, node_positions):
    """
    Penalize model for creating high-energy or unnecessary (non-causal) links.
    """
    loss = 0.0
    for u, v in model_edges:
        dx, dy, dt = node_positions[v] - node_positions[u]
        ds2 = -(dt**2) + dx**2 + dy**2
        if ds2 < -1e-2:
            energy = np.sqrt(-ds2)
            loss += (
                energy  # encourage low-energy/short timelike links
            )
        else:
            loss += 10.0  # heavy penalty for non-causal/spacelike
    return loss / len(model_edges) if len(model_edges) > 0 else 0.0


# %%
# === Step 9: Neural Network Training Framework ===


rprint(
    Panel(
        "[bold cyan]🧠 Setting up Neural Network Training Framework[/bold cyan]",
        style="cyan",
    )
)


class CausalNetworkDataset(Dataset):
    """Dataset for training causal connection prediction"""

    def __init__(
        self, node_positions, true_edges, all_possible_edges
    ):
        self.positions = torch.FloatTensor(node_positions)
        self.true_edges = set(true_edges)
        self.all_edges = all_possible_edges

        # Create labels: 1 for true causal edges, 0 for others
        self.labels = torch.FloatTensor([
            1.0 if edge in self.true_edges else 0.0
            for edge in all_possible_edges
        ])

        # Create edge features (position differences, spacetime intervals)
        self.edge_features = []
        for u, v in all_possible_edges:
            p1, p2 = self.positions[u], self.positions[v]
            dx, dy, dt = p2 - p1
            ds2 = -(dt**2) + dx**2 + dy**2

            # Features: [dx, dy, dt, ds2, spatial_distance, is_timelike]
            spatial_dist = torch.sqrt(dx**2 + dy**2)
            is_timelike = 1.0 if ds2 < 0 else 0.0

            features = torch.FloatTensor([
                dx,
                dy,
                dt,
                ds2,
                spatial_dist,
                is_timelike,
            ])
            self.edge_features.append(features)

        self.edge_features = torch.stack(self.edge_features)

    def __len__(self):
        return len(self.all_edges)

    def __getitem__(self, idx):
        return (
            self.edge_features[idx],
            self.labels[idx],
            self.all_edges[idx],
        )


class CausalConnectionPredictor(nn.Module):
    """Neural network to predict causal connections in spacetime"""

    def __init__(self, input_dim=6, hidden_dims=[64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        # Output layer (probability of causal connection)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def physics_aware_loss(
    predictions, labels, edge_features, edges, lambda_physics=1.0
):
    """Combined loss: BCE + Physics penalty"""
    # Standard binary cross-entropy
    bce_loss = F.binary_cross_entropy(predictions, labels)

    # Physics penalty for predicted edges
    physics_penalty = 0.0
    predicted_edges = []

    # Handle the batched tensor structure from DataLoader
    if isinstance(edges, list) and len(edges) == 2:
        # edges is [tensor([u1, u2, ...]), tensor([v1, v2, ...])]
        u_batch, v_batch = edges
        edge_tuples = [(u_batch[i].item(), v_batch[i].item()) for i in range(len(u_batch))]
    else:
        # Fallback for non-batched case
        edge_tuples = edges

    for pred, edge in zip(predictions, edge_tuples):
        if pred > 0.5:  # Predicted as causal connection
            u, v = edge  # Unpack the edge tuple
            predicted_edges.append((u, v))

    if predicted_edges:
        # Extract positions from the original data
        positions_dict = {
            i: positions[i] for i in range(len(positions))
        }
        physics_penalty = physics_loss(
            predicted_edges, [], positions_dict
        )

    total_loss = bce_loss + lambda_physics * physics_penalty
    return total_loss, bce_loss, physics_penalty


# %%
# === Step 10: Prepare Training Data ===
rprint("[yellow]📊 Preparing training data...[/yellow]")

# Get all possible edges (not just causal ones)
all_possible_edges = []
for i in range(N):
    for j in range(i + 1, N):
        all_possible_edges.append((i, j))

# Get true causal edges from the initial graph
true_causal_edges = list(states[0].edges())

# Create dataset
dataset = CausalNetworkDataset(
    positions, true_causal_edges, all_possible_edges
)

# Split into train/validation
train_idx, val_idx = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

rprint(
    f"[green]✅ Dataset created: {len(train_dataset)} training, {len(val_dataset)} validation samples[/green]"
)

# %%
# === Step 11: Train the Model ===
rprint("[magenta]🚀 Starting model training...[/magenta]")

# Initialize model
model = CausalConnectionPredictor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)

# Training metrics
train_losses = []
val_losses = []
physics_penalties = []
best_val_loss = float("inf")

epochs = 100

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
) as progress:
    task = progress.add_task(
        f"Training for {epochs} epochs...", total=epochs
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_physics_penalty = 0.0

        for batch_features, batch_labels, batch_edges in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_features)
            loss, bce_loss, phys_penalty = physics_aware_loss(
                predictions,
                batch_labels,
                batch_features,
                batch_edges,
            )

            loss.backward()
            optimizer.step()

            epoch_train_loss += bce_loss.item()
            epoch_physics_penalty += phys_penalty

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for (
                batch_features,
                batch_labels,
                batch_edges,
            ) in val_loader:
                predictions = model(batch_features)
                val_loss, _, _ = physics_aware_loss(
                    predictions,
                    batch_labels,
                    batch_features,
                    batch_edges,
                )
                epoch_val_loss += val_loss.item()

        # Average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_physics_penalty = epoch_physics_penalty / len(
            train_loader
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        physics_penalties.append(avg_physics_penalty)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_causal_model.pth")

        # Progress update
        if epoch % 10 == 0:
            rprint(
                f"[dim]Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Physics Penalty: {avg_physics_penalty:.4f}[/dim]"
            )

        progress.advance(task)

rprint("[green]✅ Training completed![/green]")

# %%
# === Step 12: Evaluate and Visualize Results ===
rprint("[blue]📈 Evaluating model performance...[/blue]")

# Load best model
model.load_state_dict(torch.load("best_causal_model.pth"))
model.eval()

# Get predictions on validation set
val_predictions = []
val_true_labels = []
val_edges_list = []

with torch.no_grad():
    for batch_features, batch_labels, batch_edges in val_loader:
        predictions = model(batch_features)
        val_predictions.extend(predictions.numpy())
        val_true_labels.extend(batch_labels.numpy())
        val_edges_list.extend(batch_edges)

val_predictions = np.array(val_predictions)
val_true_labels = np.array(val_true_labels)

# Calculate metrics


threshold = 0.5
predicted_labels = (val_predictions > threshold).astype(int)

accuracy = accuracy_score(val_true_labels, predicted_labels)
precision = precision_score(val_true_labels, predicted_labels)
recall = recall_score(val_true_labels, predicted_labels)
f1 = f1_score(val_true_labels, predicted_labels)
auc = roc_auc_score(val_true_labels, val_predictions)

# Create results table
results_table = Table(title="🎯 Model Performance Metrics")
results_table.add_column("Metric", style="cyan", no_wrap=True)
results_table.add_column("Value", style="green")

results_table.add_row("Accuracy", f"{accuracy:.4f}")
results_table.add_row("Precision", f"{precision:.4f}")
results_table.add_row("Recall", f"{recall:.4f}")
results_table.add_row("F1-Score", f"{f1:.4f}")
results_table.add_row("AUC-ROC", f"{auc:.4f}")

console.print(results_table)


# %%
# === Step 13: Training Visualization ===
def plot_training_results():
    """Plot training curves and model predictions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(15, 10)
    )

    epochs_range = range(1, len(train_losses) + 1)

    # Training curves
    ax1.plot(
        epochs_range,
        train_losses,
        "b-",
        label="Training Loss",
        linewidth=2,
    )
    ax1.plot(
        epochs_range,
        val_losses,
        "r-",
        label="Validation Loss",
        linewidth=2,
    )
    ax1.set_title("Training and Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Physics penalty
    ax2.plot(epochs_range, physics_penalties, "g-", linewidth=2)
    ax2.set_title("Physics Penalty Over Time", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Physics Penalty")
    ax2.grid(True, alpha=0.3)

    # Prediction distribution
    ax3.hist(
        val_predictions[val_true_labels == 1],
        bins=20,
        alpha=0.7,
        label="True Causal",
        color="green",
        density=True,
    )
    ax3.hist(
        val_predictions[val_true_labels == 0],
        bins=20,
        alpha=0.7,
        label="Non-Causal",
        color="red",
        density=True,
    )
    ax3.axvline(
        threshold,
        color="black",
        linestyle="--",
        label=f"Threshold ({threshold})",
    )
    ax3.set_title("Prediction Distribution", fontweight="bold")
    ax3.set_xlabel("Predicted Probability")
    ax3.set_ylabel("Density")
    ax3.legend()

    # ROC curve
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(val_true_labels, val_predictions)
    ax4.plot(
        fpr,
        tpr,
        "b-",
        linewidth=2,
        label=f"ROC Curve (AUC = {auc:.3f})",
    )
    ax4.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
    ax4.set_title("ROC Curve", fontweight="bold")
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_training_results()

# %%
# === Step 14: Physics-Based Model Comparison ===
rprint(
    "[purple]🔬 Comparing model predictions with physics ground truth...[/purple]"
)

# Generate predicted causal graph
print("DEBUG: val_edges_list", val_edges_list)
predicted_edges = [
    (edge[0], edge[1])
    for i, edge in enumerate(val_edges_list)
    if val_predictions[i] > threshold
]

# Calculate physics loss for predictions vs true edges
predicted_physics_loss = physics_loss(
    predicted_edges,
    [],
    {i: positions[i] for i in range(len(positions))},
)
true_physics_loss = physics_loss(
    true_causal_edges,
    [],
    {i: positions[i] for i in range(len(positions))},
)

rprint(
    f"[cyan]Physics Loss - True Edges: {true_physics_loss:.4f}[/cyan]"
)
rprint(
    f"[cyan]Physics Loss - Predicted Edges: {predicted_physics_loss:.4f}[/cyan]"
)
rprint(
    f"[cyan]Improvement: {((true_physics_loss - predicted_physics_loss) / true_physics_loss * 100):.2f}%[/cyan]"
)


rprint(
    Panel(
        f"[bold green]🎉 Neural Network Training Complete![/bold green]\n"
        f"Model Accuracy: {accuracy:.3f} | F1-Score: {f1:.3f} | AUC: {auc:.3f}\n"
        f"Physics-aware loss successfully integrated!\n"
        f"Saved: best_causal_model.pth & training_results.png",
        style="green",
    )
)
