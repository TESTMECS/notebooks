#!/usr/bin/env python3
"""
Neural Network Training with Physics Loss Function
Training a model to predict causal connections in spacetime using physics-aware loss
"""
# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
from pathlib import Path
from PIL import Image
import matplotlib.animation as animation
from torch_geometric.data import Data
from tqdm import tqdm
from itertools import combinations
from torch_geometric.nn import GCNConv, global_mean_pool

console = Console()

# %%
# === Step 1: Generate Spacetime Data ===
rprint(
    Panel(
        "[bold blue]🌌 Generating Spacetime Events[/bold blue]",
        style="blue",
    )
)

# Create trefoil-inspired spacetime events
N = 24
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

x = np.sin(theta) + 2 * np.sin(2 * theta)
y = np.cos(theta) - 2 * np.cos(2 * theta)
z = -np.sin(3 * theta) + np.linspace(0, 2, N)

positions = np.stack([x, y, z], axis=1)

# Find true causal edges (timelike separated events)
true_causal_edges = []
for i in range(N):
    for j in range(i + 1, N):
        p1, p2 = positions[i], positions[j]
        dx, dy, dt = p2 - p1
        ds2 = -(dt**2) + dx**2 + dy**2
        if ds2 < -1e-2:  # Timelike separation
            true_causal_edges.append((i, j))

rprint(
    f"[green]✨ Created {N} events with {len(true_causal_edges)} causal connections[/green]"
)


# %%
# === Step 2: Physics Loss Function ===
def physics_loss(model_edges, true_edges, node_positions):
    """Penalize model for creating high-energy or non-causal links"""
    if len(model_edges) == 0:
        return 0.0

    loss = 0.0
    for u, v in model_edges:
        dx, dy, dt = node_positions[v] - node_positions[u]
        ds2 = -(dt**2) + dx**2 + dy**2
        if ds2 < -1e-2:
            energy = np.sqrt(-ds2)
            loss += energy  # Encourage low-energy timelike links
        else:
            loss += 10.0  # Heavy penalty for non-causal/spacelike
    return loss / len(model_edges)


# %%
# === Step 3: Dataset Definition ===
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

        # Create edge features
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


# %%
# === Step 4: Neural Network Model ===
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

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


# %%
# === Step 5: Physics-Aware Loss Function ===
def physics_aware_loss(
    predictions,
    labels,
    edge_features,
    edges,
    node_positions,
    lambda_physics=1.0,
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
        edge_tuples = [
            (u_batch[i].item(), v_batch[i].item())
            for i in range(len(u_batch))
        ]
    else:
        # Fallback for non-batched case - ensure proper edge format
        edge_tuples = []
        if hasattr(edges, "__iter__"):
            for edge in edges:
                if isinstance(edge, tuple):
                    edge_tuples.append(edge)
                elif (
                    isinstance(edge, torch.Tensor)
                    and edge.dim() == 1
                    and len(edge) == 2
                ):
                    edge_tuples.append((
                        edge[0].item(),
                        edge[1].item(),
                    ))
                elif hasattr(edge, "__iter__") and len(edge) == 2:
                    u, v = edge
                    if hasattr(u, "item"):
                        u = u.item()
                    if hasattr(v, "item"):
                        v = v.item()
                    edge_tuples.append((int(u), int(v)))

    for pred, edge in zip(predictions, edge_tuples):
        if pred > 0.5:  # Predicted as causal connection
            u, v = edge
            predicted_edges.append((u, v))

    if predicted_edges:
        physics_penalty = physics_loss(
            predicted_edges, [], node_positions
        )

    total_loss = bce_loss + lambda_physics * physics_penalty
    return total_loss, bce_loss, physics_penalty


# %%
# === Step 6: Prepare Training Data ===
rprint("[yellow]📊 Preparing training data...[/yellow]")

# Get all possible edges
all_possible_edges = []
for i in range(N):
    for j in range(i + 1, N):
        all_possible_edges.append((i, j))

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
    f"[green]✅ Dataset: {len(train_dataset)} training, {len(val_dataset)} validation samples[/green]"
)

# %%
# === Step 7: Train the Model ===
rprint("[magenta]🚀 Starting model training...[/magenta]")

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
epochs = 50  # Reduced for faster training

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
                positions,
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
                    positions,
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
# === Step 8: Evaluate Model ===
rprint("[blue]📈 Evaluating model performance...[/blue]")

# Load best model
model.load_state_dict(torch.load("best_causal_model.pth"))
model.eval()

# Get predictions
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
threshold = 0.2
predicted_labels = (val_predictions > threshold).astype(int)

accuracy = accuracy_score(val_true_labels, predicted_labels)
precision = precision_score(val_true_labels, predicted_labels)
recall = recall_score(val_true_labels, predicted_labels)
f1 = f1_score(val_true_labels, predicted_labels)
auc = roc_auc_score(val_true_labels, val_predictions)

# Results table
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
# === Step 9: Visualize Results ===
def plot_training_results():
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
    plt.savefig(
        "causal_model_training_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


plot_training_results()

# %%
# === Step 10: Physics Comparison ===
rprint(
    "[purple]🔬 Comparing predictions with physics ground truth...[/purple]"
)

# Handle batched edges properly - flatten the batched structure
flattened_edges = []
for batch_edges in val_edges_list:
    if isinstance(batch_edges, list) and len(batch_edges) == 2:
        # batch_edges is [tensor([u1, u2, ...]), tensor([v1, v2, ...])]
        u_batch, v_batch = batch_edges
        batch_edge_tuples = [
            (u_batch[i].item(), v_batch[i].item())
            for i in range(len(u_batch))
        ]
        flattened_edges.extend(batch_edge_tuples)
    else:
        # Fallback for non-batched case - ensure we convert tensors to tuples
        if isinstance(batch_edges, torch.Tensor):
            # If it's a single tensor, skip it or handle appropriately
            continue
        elif hasattr(batch_edges, "__iter__"):
            for edge in batch_edges:
                if isinstance(edge, tuple):
                    flattened_edges.append(edge)
                elif (
                    isinstance(edge, torch.Tensor)
                    and edge.dim() == 1
                    and len(edge) == 2
                ):
                    # Convert tensor edge to tuple
                    flattened_edges.append((
                        edge[0].item(),
                        edge[1].item(),
                    ))
                elif hasattr(edge, "__iter__") and len(edge) == 2:
                    # Convert other iterable to tuple
                    u, v = edge
                    if hasattr(u, "item"):
                        u = u.item()
                    if hasattr(v, "item"):
                        v = v.item()
                    flattened_edges.append((int(u), int(v)))

predicted_edges = [
    (u, v)
    for i, (u, v) in enumerate(flattened_edges)
    if i < len(val_predictions) and val_predictions[i] > threshold
]

predicted_physics_loss = physics_loss(predicted_edges, [], positions)
true_physics_loss = physics_loss(true_causal_edges, [], positions)

rprint(
    f"[cyan]Physics Loss - True Edges: {true_physics_loss:.4f}[/cyan]"
)
rprint(
    f"[cyan]Physics Loss - Predicted Edges: {predicted_physics_loss:.4f}[/cyan]"
)

if true_physics_loss > 0:
    improvement = (
        (true_physics_loss - predicted_physics_loss)
        / true_physics_loss
        * 100
    )
    rprint(f"[cyan]Physics Improvement: {improvement:.2f}%[/cyan]")

rprint(
    Panel(
        f"[bold green]🎉 Neural Network Training Complete![/bold green]\n"
        f"Model Accuracy: {accuracy:.3f} | F1-Score: {f1:.3f} | AUC: {auc:.3f}\n"
        f"Physics-aware loss successfully integrated!\n"
        f"Saved: best_causal_model.pth & causal_model_training_results.png",
        style="green",
    )
)
# %%
# Assume you already have attention events or synthetic event edges (i.e. (i, j) pairs)
model = CausalConnectionPredictor()
model.load_state_dict(torch.load("best_causal_model.pth"))
model.eval()

# Create test set from your event graph (can also be real GPT2 trace)
test_dataset = CausalNetworkDataset(
    positions, true_causal_edges, all_possible_edges
)
test_loader = DataLoader(test_dataset, batch_size=32)

predictions, labels = [], []
with torch.no_grad():
    for features, label, _ in test_loader:
        out = model(features)
        predictions.extend(out.numpy())
        labels.extend(label.numpy())

# Visualize prediction distribution
plt.hist(
    [p for p, l in zip(predictions, labels) if l == 1],
    label="True Causal",
    alpha=0.7,
    color="green",
)
plt.hist(
    [p for p, l in zip(predictions, labels) if l == 0],
    label="Non-Causal",
    alpha=0.7,
    color="red",
)
plt.axvline(0.5, linestyle="--", color="black", label="Threshold")
plt.title("Model Predictions on Attention Events")
plt.legend()
plt.show()
# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
ax.scatter(x, y, z, c="cyan", s=60, label="Events")

for (i, j), p in zip(all_possible_edges, predictions):
    color = "red" if p > 0.5 else "gray"
    ax.plot(
        [x[i], x[j]],
        [y[i], y[j]],
        [z[i], z[j]],
        color=color,
        alpha=0.6,
    )

ax.set_title("Attention Events Colored by Causal Prediction")
plt.legend()
plt.tight_layout()
plt.show()

# %%
pred_edges = [
    (i, j)
    for (i, j), p in zip(all_possible_edges, predictions)
    if p > 0.5
]
true_loss = physics_loss(true_causal_edges, [], positions)
pred_loss = physics_loss(pred_edges, [], positions)

print(f"True Physics Loss: {true_loss:.4f}")
print(f"Predicted Physics Loss: {pred_loss:.4f}")

# %%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
loader = DataLoader(mnist, batch_size=1, shuffle=True)

# Get a single image
for image, label in loader:
    break

image = image.squeeze().numpy()  # Shape: (28, 28)
label = label.item()

# Sample positions of pixels > threshold
threshold = 0.2
events = np.argwhere(
    image > threshold
)  # (N, 2) pixel (y, x) positions
values = image[events[:, 0], events[:, 1]]

# Add a fake time coordinate (just index)
positions = np.column_stack([
    events[:, 1],
    events[:, 0],
    np.arange(len(events)),
])
print("Num events:", positions.shape[0])
from itertools import combinations

all_possible_edges = list(combinations(range(len(positions)), 2))
print("Num edges:", len(all_possible_edges))

# %%
test_dataset = CausalNetworkDataset(
    positions, true_causal_edges, all_possible_edges
)
test_loader = DataLoader(test_dataset, batch_size=32)

predictions, labels = [], []
with torch.no_grad():
    for features, label, _ in test_loader:
        out = model(features)
        predictions.extend(out.numpy())
        labels.extend(label.numpy())

# %%
from train_causal_model import (
    CausalConnectionPredictor,
    CausalNetworkDataset,
    physics_loss,
)

features = []
for i, j in all_possible_edges:
    pi, pj = positions[i], positions[j]
    features.append(np.concatenate([pi, pj]))
features = torch.tensor(features, dtype=torch.float32)

with torch.no_grad():
    predictions = model(features).squeeze().numpy()

# Apply threshold to get causal edges
causal_edges = [
    e for e, p in zip(all_possible_edges, predictions) if p > 1e-6
]
print(f"Number of causal edges: {len(causal_edges)}")


# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image, cmap="gray")
x, y = positions[:, 0], positions[:, 1]

# plot nodes
ax.scatter(x, y, c="cyan", s=8)

# plot edges
for i, j in causal_edges:
    ax.plot([x[i], x[j]], [y[i], y[j]], color="orange", linewidth=0.5)

ax.set_title(f"Causal Graph Over MNIST Digit: {label}")
plt.axis("off")
plt.show()


# %%

image = np.zeros((28, 28))  # Blank image
positions = np.array([
    [i, j, k]
    for k, (i, j) in enumerate(np.ndindex(28, 28))
    if (i + j) % 10 == 0
])
causal_edges = [
    (i, j)
    for i in range(len(positions))
    for j in range(i + 1, len(positions))
    if abs(i - j) == 1
]


# Extract causal edges in time order based on the time coordinate (z) from positions
def get_time_sorted_edges(positions, causal_edges):
    time_index = positions[:, 2]
    edge_times = [
        (i, j, max(time_index[i], time_index[j]))
        for (i, j) in causal_edges
    ]
    edge_times.sort(key=lambda x: x[2])
    return [(i, j) for (i, j, _) in edge_times]


# Animate edge appearance over time
def animate_causal_reconstruction(
    image,
    positions,
    causal_edges,
    save_path="causal_reconstruction.gif",
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    x, y = positions[:, 0], positions[:, 1]
    ax.scatter(x, y, c="cyan", s=10)

    sorted_edges = get_time_sorted_edges(positions, causal_edges)

    lines = []

    def init():
        return []

    def update(frame):
        i, j = sorted_edges[frame]
        x0, y0 = positions[i, 0], positions[i, 1]
        x1, y1 = positions[j, 0], positions[j, 1]
        (ln,) = ax.plot(
            [x0, x1], [y0, y1], color="orange", linewidth=1
        )
        lines.append(ln)
        return lines

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(sorted_edges),
        init_func=init,
        interval=50,
        blit=True,
        repeat=False,
    )
    ani.save(save_path, writer="pillow")
    plt.close()
    return save_path


# Run with dummy data
animate_causal_reconstruction(image, positions, causal_edges)

# %%
from torch_geometric.data import Data
from tqdm import tqdm

graph_data_list = []

print("Processing MNIST images...")
processed_count = 0
skipped_few_events = 0
skipped_no_causal = 0
prediction_stats = []

for img, label in tqdm(mnist):
    img = img.squeeze().numpy()
    events = np.argwhere(img > 0.2)
    values = img[events[:, 0], events[:, 1]]
    positions = np.column_stack([
        events[:, 1],
        events[:, 0],
        np.arange(len(events)),
    ])

    # Skip empty images
    if len(positions) < 3:
        skipped_few_events += 1
        continue

    edges = list(combinations(range(len(positions)), 2))

    # Create features in the same format as training: [dx, dy, dt, ds2, spatial_distance, is_timelike]
    features_list = []
    for i, j in edges:
        p1, p2 = positions[i], positions[j]
        dx, dy, dt = p2 - p1
        ds2 = -(dt**2) + dx**2 + dy**2
        spatial_dist = np.sqrt(dx**2 + dy**2)
        is_timelike = 1.0 if ds2 < 0 else 0.0

        features = [dx, dy, dt, ds2, spatial_dist, is_timelike]
        features_list.append(features)

    features = torch.tensor(features_list, dtype=torch.float32)

    with torch.no_grad():
        preds = model(features).squeeze().numpy()
        prediction_stats.extend(preds)

    # Lower threshold since model might be conservative
    threshold = 0.1  # Reduced from 0.5
    causal_edges = [
        (i, j) for (i, j), p in zip(edges, preds) if p > threshold
    ]

    if not causal_edges:
        skipped_no_causal += 1
        continue

    edge_index = torch.tensor(causal_edges, dtype=torch.long).T
    node_attr = torch.tensor(
        positions[:, :2], dtype=torch.float32
    )  # just (x, y) for now

    graph_data_list.append(
        Data(
            x=node_attr,
            edge_index=edge_index,
            y=torch.tensor([label]),
        )
    )
    processed_count += 1

    # Limit to avoid memory issues
    if processed_count >= 4000:
        break

print(f"Processed: {processed_count}")
print(f"Skipped (few events): {skipped_few_events}")
print(f"Skipped (no causal edges): {skipped_no_causal}")
print(f"Total graph data: {len(graph_data_list)}")

if prediction_stats:
    print(
        f"Prediction stats - Min: {np.min(prediction_stats):.4f}, Max: {np.max(prediction_stats):.4f}, Mean: {np.mean(prediction_stats):.4f}"
    )
    print(
        f"Predictions > 0.1: {np.sum(np.array(prediction_stats) > 0.1)}"
    )
    print(
        f"Predictions > 0.5: {np.sum(np.array(prediction_stats) > 0.5)}"
    )
else:
    print("No predictions made!")

# %%
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn


class CausalGNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 10)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)


# Only create dataloaders if we have data
if len(graph_data_list) > 0:
    # Split data safely
    train_size = min(
        len(graph_data_list) - 100, 3000
    )  # Ensure we have at least 100 for test
    test_size = len(graph_data_list) - train_size

    print(f"Creating train loader with {train_size} samples")
    print(f"Creating test loader with {test_size} samples")

    train_loader = DataLoader(
        graph_data_list[:train_size], batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        graph_data_list[train_size:], batch_size=64
    )

    model_gnn = CausalGNN()
    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
else:
    print(
        "No graph data available! Check threshold and feature extraction."
    )
    # Create some dummy data for testing
    print("Creating dummy data for testing...")
    dummy_graphs = []
    for i in range(100):
        x = torch.randn(5, 2)  # 5 nodes, 2 features each
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long
        )
        y = torch.tensor([i % 10])  # Random label
        dummy_graphs.append(Data(x=x, edge_index=edge_index, y=y))

    train_loader = DataLoader(
        dummy_graphs[:80], batch_size=32, shuffle=True
    )
    test_loader = DataLoader(dummy_graphs[80:], batch_size=64)

    model_gnn = CausalGNN()
    optimizer = torch.optim.Adam(model_gnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print("Using dummy data for demonstration.")

# %%
for epoch in range(100):
    model_gnn.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model_gnn(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# %%
model_gnn.eval()
correct = 0
total = 0
for batch in test_loader:
    out = model_gnn(batch)
    pred = out.argmax(dim=1)
    correct += (pred == batch.y).sum().item()
    total += len(batch.y)

print(f"Test Accuracy on Causal Graphs: {correct / total:.4f}")


# %%
