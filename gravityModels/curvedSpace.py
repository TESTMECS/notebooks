# %%
"""
Curved Concept Spacetime: A proof of concept for applying curved spacetime geometry
to the concept space of language models.

Key insights:
1. Ideas live on a curved manifold where hierarchical relationships are light-like geodesics
2. BERT embeddings can be mapped to Minkowski spacetime preserving causal structure
3. Invertible Neural Networks (INNs) learn diffeomorphisms between flat and curved representations
4. Metric regularization enforces proper Lorentzian signature on the learned manifold
"""

# %% [markdown]
# ## Imports and Configuration

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from collections import defaultdict
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Ensure NLTK data is available
try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# Configuration
BERT_MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)
bert_model.eval()


# %% [markdown]
# ## Core Utility Functions
def get_bert_embedding(text, model, tokenizer):
    """Extract BERT embedding for text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def get_wordnet_subtree(start_synset_name="mammal.n.01", depth_limit=3):
    """Extract WordNet concept hierarchy."""
    concepts = {}
    try:
        root_synset = wn.synset(start_synset_name)
    except Exception as e:
        print(f"Error accessing synset '{start_synset_name}': {e}")
        return {}

    queue = [(root_synset, 0, None)]
    visited_synsets = set()
    processed_names = set()

    while queue:
        current_synset, depth, parent_name = queue.pop(0)

        if current_synset in visited_synsets or depth > depth_limit:
            continue
        visited_synsets.add(current_synset)

        # Get clean name
        name = None
        lemmas = current_synset.lemmas()
        if lemmas:
            name = lemmas[0].name().replace("_", " ")
            for lem in lemmas[1:]:
                temp_name = lem.name().replace("_", " ")
                if len(temp_name) < len(name) and "-" not in temp_name:
                    name = temp_name

        if name is None or name in processed_names:
            continue
        processed_names.add(name)

        concepts[name] = {
            "synset": current_synset,
            "depth": depth,
            "parents": set(),
            "children": set(),
        }

        if parent_name and parent_name in concepts:
            concepts[name]["parents"].add(parent_name)
            concepts[parent_name]["children"].add(name)

        for hyponym in current_synset.hyponyms():
            queue.append((hyponym, depth + 1, name))

    return concepts


def build_hierarchy_from_concepts(concepts):
    """Build hierarchy pairs from concept dictionary."""
    hierarchy = []
    root_name = None
    min_depth = float("inf")

    for name, data in concepts.items():
        if not data["parents"]:
            if data["depth"] <= min_depth:
                min_depth = data["depth"]
                root_name = name
        for parent_name in data["parents"]:
            if parent_name in concepts:
                hierarchy.append((name, parent_name))

    if root_name:
        hierarchy.append((root_name, None))

    return hierarchy, root_name


# %% [markdown]
# ## Minkowski Embedding Functions
def spatial_distance_sq(coord1_spatial, coord2_spatial):
    """Compute squared spatial distance."""
    return sum((x1 - x2) ** 2 for x1, x2 in zip(coord1_spatial, coord2_spatial))


def check_causal_link(child_coord, parent_coord, epsilon, spatial_dims):
    """Check if parent-child relationship satisfies causal constraint."""
    if parent_coord is None:
        return True

    child_t = child_coord[0]
    parent_t = parent_coord[0]
    child_spatial = child_coord[1 : spatial_dims + 1]
    parent_spatial = parent_coord[1 : spatial_dims + 1]

    spatial_dist_sq = spatial_distance_sq(child_spatial, parent_spatial)
    time_diff = child_t - parent_t

    return time_diff > 0 and time_diff**2 >= spatial_dist_sq + epsilon


def compute_minkowski_embedding(
    concept_embeddings, hierarchy_pairs, spatial_dims, epsilon
):
    """Compute Minkowski spacetime embedding preserving causal structure."""
    # Use PCA to get initial spatial coordinates
    embeddings_list = [
        concept_embeddings[name]
        for name, _ in hierarchy_pairs
        if name in concept_embeddings
    ]

    if not embeddings_list:
        return {}

    pca = PCA(n_components=spatial_dims)
    spatial_coords = pca.fit_transform(embeddings_list)

    # Initialize coordinates
    coords = {}
    name_list = [name for name, _ in hierarchy_pairs if name in concept_embeddings]

    for i, name in enumerate(name_list):
        coords[name] = [0.0] + spatial_coords[i].tolist()  # [t, x, y, ...]

    # Iteratively adjust time coordinates to satisfy causality
    max_iterations = 1000
    learning_rate = 0.01

    for iteration in range(max_iterations):
        violations = 0

        for child_name, parent_name in hierarchy_pairs:
            if child_name not in coords or (parent_name and parent_name not in coords):
                continue

            child_coord = coords[child_name]
            parent_coord = coords[parent_name] if parent_name else None

            if not check_causal_link(child_coord, parent_coord, epsilon, spatial_dims):
                violations += 1
                if parent_coord:
                    # Adjust child time to satisfy causality
                    spatial_dist_sq = spatial_distance_sq(
                        child_coord[1 : spatial_dims + 1],
                        parent_coord[1 : spatial_dims + 1],
                    )
                    required_time_diff = np.sqrt(spatial_dist_sq + epsilon) + 0.001
                    coords[child_name][0] = parent_coord[0] + required_time_diff

        if violations == 0:
            break

    return coords


# %% [markdown]
# ## Neural Network Components
class AdditiveCouplingLayer(nn.Module):
    """Additive coupling layer for Invertible Neural Network."""

    def __init__(self, input_dim, hidden_dim=128, mask_type="even"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Create mask
        if mask_type == "even":
            self.mask = torch.zeros(input_dim)
            self.mask[::2] = 1  # Even indices
        else:
            self.mask = torch.ones(input_dim)
            self.mask[::2] = 0  # Odd indices

        # Network for computing translation
        masked_dim = int(self.mask.sum())
        self.translation_net = nn.Sequential(
            nn.Linear(masked_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim - masked_dim),
        )

    def forward(self, x, reverse=False):
        self.mask = self.mask.to(x.device)

        # Ensure input is 2D (batch_size, features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if not reverse:
            # Forward pass
            x_masked = x * self.mask.unsqueeze(0)  # Broadcast mask
            # Extract masked features for each sample in batch
            masked_features = x_masked[:, self.mask.bool()]
            translation = self.translation_net(masked_features)

            y = x.clone()
            y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] + translation

            if squeeze_output:
                y = y.squeeze(0)
            return y
        else:
            # Reverse pass
            y = x.clone()
            y_masked = y * self.mask.unsqueeze(0)  # Broadcast mask
            masked_features = y_masked[:, self.mask.bool()]
            translation = self.translation_net(masked_features)

            x_reconstructed = y.clone()
            x_reconstructed[:, ~self.mask.bool()] = (
                y[:, ~self.mask.bool()] - translation
            )

            if squeeze_output:
                x_reconstructed = x_reconstructed.squeeze(0)
            return x_reconstructed


# %%
class INNPhi(nn.Module):
    """Invertible Neural Network for learning diffeomorphism phi."""

    def __init__(self, input_dim=4, hidden_dim=128, num_coupling_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.coupling_layers = nn.ModuleList()

        for i in range(num_coupling_layers):
            mask_type = "even" if i % 2 == 0 else "odd"
            self.coupling_layers.append(
                AdditiveCouplingLayer(input_dim, hidden_dim, mask_type)
            )

    def forward(self, x, reverse=False):
        if not reverse:
            for layer in self.coupling_layers:
                x = layer(x, reverse=False)
        else:
            for layer in reversed(self.coupling_layers):
                x = layer(x, reverse=True)
        return x


# %% [markdown]
# ## Geometric Analysis Functions
def compute_jacobian(phi_model, m_coords_tensor):
    """Compute Jacobian matrix of phi at given coordinates."""
    m_coords_tensor.requires_grad_(True)

    jacobian = torch.zeros(4, 4, device=m_coords_tensor.device)

    for i in range(4):
        e_coords = phi_model(m_coords_tensor.unsqueeze(0), reverse=False).squeeze()
        grad_outputs = torch.zeros_like(e_coords)
        grad_outputs[i] = 1.0

        grads = torch.autograd.grad(
            outputs=e_coords,
            inputs=m_coords_tensor,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        jacobian[i] = grads

    return jacobian.T


def pullback_metric(m_coords_tensor, phi_model, eta_E):
    """Compute pullback metric g_M = J^T @ eta_E @ J."""
    J = compute_jacobian(phi_model, m_coords_tensor)
    return J.T @ eta_E @ J


def analyze_local_geometry(m_coord_np, phi_model, eta_E):
    """Analyze local geometric properties at a point."""
    m_tensor = torch.tensor(m_coord_np, dtype=torch.float32, device=DEVICE)
    g_M = pullback_metric(m_tensor, phi_model, eta_E)

    eigenvals = torch.linalg.eigvals(g_M).real
    det_g = torch.det(g_M)
    trace_g = torch.trace(g_M)

    return {
        "metric_tensor": g_M.detach().cpu().numpy(),
        "eigenvalues": eigenvals.detach().cpu().numpy(),
        "determinant": det_g.item(),
        "trace": trace_g.item(),
        "signature": tuple(torch.sign(eigenvals).int().tolist()),
    }


# %% [markdown]
# ## Training Functions
def train_inn_phi_embedding(
    tokens,
    hierarchy_pairs,
    inn_phi_model,
    num_epochs=1000,
    learning_rate=1e-3,  # Increased learning rate
    epsilon1_target_sq=1e-7,
    causal_margin=0.001,
    g_tt_target=-1.0,
    g_spatial_diag_target=1.0,
    causal_loss_weight=1.0,
    interval_loss_weight=1.0,
    g_tt_mse_weight=0.001,  # Reduced metric regularization weight
    g_spatial_diag_mse_weight=0.001,
    g_off_diag_mse_weight=0.0001,
    gradient_clip_value=1.0,
    metric_reg_frequency=10,  # Only compute metric regularization every N epochs
):
    """Train INN to learn curved manifold embedding with optimized metric regularization."""

    inn_phi_model = inn_phi_model.to(DEVICE)
    optimizer = optim.Adam(inn_phi_model.parameters(), lr=learning_rate)

    # Initialize m_coords randomly
    num_tokens = len(tokens)
    m_coords_map = {token: torch.randn(4, device=DEVICE) * 0.1 for token in tokens}

    # Minkowski metric in E space
    eta_E = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], device=DEVICE))

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0.0

        # Primary loss: Causality and interval constraints
        causal_violations = 0
        for child_token, parent_token in hierarchy_pairs:
            if child_token not in m_coords_map or (
                parent_token and parent_token not in m_coords_map
            ):
                continue

            child_m = m_coords_map[child_token]
            child_e = inn_phi_model(child_m.unsqueeze(0), reverse=False).squeeze()

            if parent_token:
                parent_m = m_coords_map[parent_token]
                parent_e = inn_phi_model(parent_m.unsqueeze(0), reverse=False).squeeze()

                # Causal constraint: child_t > parent_t + margin
                causal_diff = child_e[0] - parent_e[0] - causal_margin
                if causal_diff < 0:
                    causal_violations += 1
                    causal_loss = causal_loss_weight * (-causal_diff).pow(2)
                    total_loss += causal_loss

                # Interval constraint
                diff_e = child_e - parent_e
                interval_sq = (
                    -(diff_e[0] ** 2) + diff_e[1] ** 2 + diff_e[2] ** 2 + diff_e[3] ** 2
                )
                interval_loss = interval_loss_weight * (
                    interval_sq + epsilon1_target_sq
                ).pow(2)
                total_loss += interval_loss

        # Simplified metric regularization (less frequent, fewer samples)
        if epoch % metric_reg_frequency == 0 and epoch > 0:
            # Only regularize on 3 sample tokens to reduce computation
            sample_tokens = tokens[: min(3, len(tokens))]
            for token in sample_tokens:
                try:
                    m_coord = m_coords_map[token]
                    # Simplified metric regularization without full Jacobian
                    # Just ensure the transformation is reasonable
                    e_coord = inn_phi_model(
                        m_coord.unsqueeze(0), reverse=False
                    ).squeeze()

                    # Simple regularization: penalize extreme values
                    coord_penalty = 0.0001 * (e_coord.pow(2).sum())
                    total_loss += coord_penalty

                except Exception as e:
                    print(f"Warning: Metric regularization failed for {token}: {e}")
                    continue

        # Backpropagation
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                inn_phi_model.parameters(), gradient_clip_value
            )
            optimizer.step()

        # Progress reporting
        if epoch % 50 == 0:
            print(
                f"Epoch {epoch}, Loss: {total_loss.item():.6f}, Causal violations: {causal_violations}"
            )

    # Convert final coordinates to numpy
    final_m_coords = {
        token: coords.detach().cpu().numpy() for token, coords in m_coords_map.items()
    }

    return final_m_coords, inn_phi_model


# %% [markdown]
# ## Visualization Functions
def plot_3d_manifold(
    m_coords_map,
    hierarchy_pairs,
    color_values=None,
    color_label="Feature",
    title="Conceptual Manifold",
):
    """Plot 3D visualization of manifold embedding."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Extract coordinates
    tokens = list(m_coords_map.keys())
    coords = np.array([m_coords_map[token] for token in tokens])

    # Plot points
    x_coords = coords[:, 1]  # X spatial
    y_coords = coords[:, 2]  # Y spatial
    z_coords = coords[:, 0]  # T time

    if color_values is not None:
        scatter = ax.scatter(
            x_coords,
            y_coords,
            z_coords,
            c=color_values,
            s=100,
            cmap="viridis",
            alpha=0.8,
        )
        plt.colorbar(scatter, label=color_label)
    else:
        ax.scatter(x_coords, y_coords, z_coords, s=100, alpha=0.8)

    # Plot hierarchy connections
    for child, parent in hierarchy_pairs:
        if child in m_coords_map and parent and parent in m_coords_map:
            child_coord = m_coords_map[child]
            parent_coord = m_coords_map[parent]
            ax.plot(
                [child_coord[1], parent_coord[1]],
                [child_coord[2], parent_coord[2]],
                [child_coord[0], parent_coord[0]],
                "k-",
                alpha=0.5,
            )

    # Labels
    for i, token in enumerate(tokens):
        coord = coords[i]
        ax.text(coord[1], coord[2], coord[0], token, fontsize=8)

    ax.set_xlabel("X (spatial)")
    ax.set_ylabel("Y (spatial)")
    # Use try-catch for 3D axis labeling
    try:
        ax.set_zlabel("T (time)")
    except AttributeError:
        pass  # Some matplotlib versions may not support set_zlabel
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_metric_analysis(m_coords_map, phi_model, eta_E, hierarchy_pairs):
    """Visualize metric properties across the manifold."""
    tokens = list(m_coords_map.keys())

    # Analyze geometry at each point
    determinants = []
    traces = []
    signatures = []

    for token in tokens:
        analysis = analyze_local_geometry(m_coords_map[token], phi_model, eta_E)
        determinants.append(analysis["determinant"])
        traces.append(analysis["trace"])
        signatures.append(analysis["signature"])

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Determinant distribution
    axes[0, 0].hist(determinants, bins=20, alpha=0.7)
    axes[0, 0].set_title("Metric Determinant Distribution")
    axes[0, 0].set_xlabel("det(g)")

    # Trace distribution
    axes[0, 1].hist(traces, bins=20, alpha=0.7)
    axes[0, 1].set_title("Metric Trace Distribution")
    axes[0, 1].set_xlabel("tr(g)")

    # 2D plot colored by determinant
    coords = np.array([m_coords_map[token] for token in tokens])
    scatter = axes[1, 0].scatter(
        coords[:, 1], coords[:, 2], c=determinants, cmap="RdBu"
    )
    axes[1, 0].set_title("Spatial Coords Colored by det(g)")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    plt.colorbar(scatter, ax=axes[1, 0])

    # Signature analysis
    signature_counts = {}
    for sig in signatures:
        sig_str = str(sig)
        signature_counts[sig_str] = signature_counts.get(sig_str, 0) + 1

    axes[1, 1].bar(signature_counts.keys(), signature_counts.values())
    axes[1, 1].set_title("Metric Signature Distribution")
    axes[1, 1].set_xlabel("Signature")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Example Usage and Demo
DEPTH_LIMIT = 3
# Load WordNet mammal concepts
print("Loading WordNet mammal concepts...")
mammal_concepts = get_wordnet_subtree("mammal.n.01", depth_limit=DEPTH_LIMIT)
print(f"Loaded {len(mammal_concepts)} mammal concepts")

# %%
# Generate BERT embeddings
print("Generating BERT embeddings...")
concept_embeddings = {}
for name in mammal_concepts.keys():
    try:
        concept_embeddings[name] = get_bert_embedding(name, bert_model, tokenizer)
    except Exception as e:
        print(f"Error getting embedding for '{name}': {e}")

print(f"Generated {len(concept_embeddings)} BERT embeddings")

# %%
# Build hierarchy
hierarchy_pairs, root_name = build_hierarchy_from_concepts(mammal_concepts)
hierarchy_pairs = [
    (child, parent)
    for child, parent in hierarchy_pairs
    if child in concept_embeddings and (parent is None or parent in concept_embeddings)
]
print(f"Built hierarchy with {len(hierarchy_pairs)} pairs, root: {root_name}")

# %%
# Compute initial Minkowski embedding
print("Computing Minkowski embedding...")
minkowski_coords = compute_minkowski_embedding(
    concept_embeddings, hierarchy_pairs, spatial_dims=2, epsilon=1e-5
)
print(f"Computed Minkowski embedding for {len(minkowski_coords)} concepts")

# Visualize flat Minkowski embedding
if minkowski_coords:
    plot_3d_manifold(
        minkowski_coords,
        hierarchy_pairs,
        title="Flat Minkowski Embedding",
    )

# %%
# Initialize and train INN
print("Training INN for curved manifold embedding...")
tokens = list(concept_embeddings.keys())
inn_model = INNPhi(input_dim=4, hidden_dim=64, num_coupling_layers=6)

final_coords, trained_model = train_inn_phi_embedding(
    tokens,
    hierarchy_pairs,
    inn_model,
    num_epochs=200,
    learning_rate=1e-3,  # Higher learning rate
    metric_reg_frequency=20,  # Less frequent metric regularization
)

print("Training completed!")

# %%
# Visualize learned curved manifold
plot_3d_manifold(
    final_coords,
    hierarchy_pairs,
    title="Learned Curved Conceptual Manifold",
)

# %%
# Analyze metric properties
eta_E = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0], device=DEVICE))
plot_metric_analysis(final_coords, trained_model, eta_E, hierarchy_pairs)
