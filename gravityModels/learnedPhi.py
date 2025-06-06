# Wrapper: Gravity-Aware GNN + Phi Field + Encoder-Decoder Integration
# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# from enc2dec import Transformer


# %%
# === Gravity-aware GNN Module ===
class GravityAwareGNN(torch.nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, out_dim=64):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


# %%
class CausalConnectionPredictor(nn.Module):
    """Neural network to predict causal connections in spacetime"""

    def __init__(self, input_dim=6, hidden_dims=[64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = hidden_dim

        # Output layer (probability of causal connection)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def physics_loss(model_edges, node_positions):
    """
    Calculate energy-based physics loss for a set of edges.
    Penalizes high-energy or non-causal (spacelike) connections.
    """
    if not model_edges:
        return 0.0

    loss = 0.0
    for u, v in model_edges:
        if u in node_positions and v in node_positions:
            p1, p2 = node_positions[u], node_positions[v]
            dx = p2[0] - p1[0]  # x_emb difference
            dy = p2[1] - p1[1]  # y_emb difference
            dt = p2[2] - p1[2]  # time difference

            ds2 = -(dt**2) + dx**2 + dy**2  # Minkowski interval

            if ds2 < -1e-2:  # Timelike (causal)
                energy = np.sqrt(-ds2)
                loss += energy  # Encourage low-energy causal links
            else:  # Spacelike (non-causal)
                loss += 10.0  # Heavy penalty for non-causal connections

    return loss / len(model_edges)


def extract_edge_features(event1, event2):
    """Extract spacetime features for two events"""
    dx = event2.x_emb - event1.x_emb
    dy = event2.y_emb - event1.y_emb
    dt = event2.t - event1.t
    ds2 = -(dt**2) + dx**2 + dy**2

    spatial_dist = np.sqrt(dx**2 + dy**2)
    is_timelike = 1.0 if ds2 < 0 else 0.0

    return torch.FloatTensor([dx, dy, dt, ds2, spatial_dist, is_timelike])


# %%
class GravityCausalWrapper(nn.Module):
    def __init__(self, gnn_model, phi_model, edge_predictor):
        super().__init__()
        self.gnn = gnn_model  # Your GNN encoder (e.g., GCN, GAT)
        self.phi = phi_model  # Learned gravitational scalar field
        self.edge_predictor = (
            edge_predictor  # CausalConnectionPredictor from enc2dec.py
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Tensor of shape (2, num_edges)

        Returns:
            edge_probs: Predicted causal connection probability per edge (num_edges,)
        """
        x_emb = self.gnn(x, edge_index)  # Embedding should contain time as x_emb[:, 0]
        edge_features = []

        for idx in range(edge_index.shape[1]):
            i, j = edge_index[0, idx], edge_index[1, idx]
            ei, ej = x_emb[i], x_emb[j]

            delta = ej - ei
            dt, dx, dy = delta[0], delta[1], delta[2]
            dx2 = dx**2 + dy**2
            dt2 = dt**2

            phi_i = self.phi(ei.unsqueeze(0))  # Shape (1,) => scalar
            ds2 = -phi_i * dt2 + dx2

            spatial_dist = dx2.sqrt()
            is_timelike = (ds2 < 0).float()

            features = torch.stack([dx, dy, dt, ds2, spatial_dist, is_timelike])
            edge_features.append(features)

        edge_features = torch.stack(edge_features)  # (num_edges, 6)
        return self.edge_predictor(edge_features)


# %%
class GravityField(nn.Module):
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


def minkowski_ds2_with_phi(t1, x1, t2, x2, phi_model):
    delta_t = t1 - t2
    delta_x = x1 - x2
    dx2 = torch.sum(delta_x**2, dim=-1)
    coords = torch.cat([t1.unsqueeze(-1), x1], dim=-1)
    phi_vals = phi_model(coords)
    ds2 = -phi_vals * delta_t**2 + dx2
    return ds2


def project_until_convergence(
    pairs, spatial, time_vec, phi_model, eps1=1e-5, max_passes=10000
):
    num_passes = 0
    converged = False
    time_vec = time_vec.clone().detach().requires_grad_(False)
    spatial = spatial.clone().detach().requires_grad_(False)

    while not converged and num_passes < max_passes:
        num_passes += 1
        converged = True

        for i, j in pairs:
            t_i, t_j = time_vec[i], time_vec[j]
            x_i, x_j = spatial[i], spatial[j]
            delta_t = t_i - t_j
            delta_x = x_i - x_j
            dx2 = torch.sum(delta_x**2)

            coords_i = torch.cat([t_i.view(1), x_i])
            phi_i = phi_model(coords_i.unsqueeze(0))[0]

            ds2 = -phi_i * delta_t**2 + dx2

            if ds2 >= -eps1:
                new_delta_t = torch.sqrt((dx2 + eps1) / phi_i + 1e-9)
                time_vec[i] = t_j + new_delta_t.item()
                converged = False

    return time_vec


def train_phi_from_ds2_violations(
    phi_model, pairs, spatial, time_vec, eps1=1e-5, num_epochs=200, lr=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi_model.to(device)
    optimizer = torch.optim.Adam(phi_model.parameters(), lr=lr)
    spatial = spatial.to(device)
    time_vec = time_vec.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        phi_model.train()

        for i, j in pairs:
            t_i, t_j = time_vec[i], time_vec[j]
            x_i, x_j = spatial[i], spatial[j]
            delta_t = t_i - t_j
            delta_x = x_i - x_j
            dx2 = torch.sum(delta_x**2)

            coords_i = torch.cat([t_i.view(1), x_i])
            phi_i = phi_model(coords_i.unsqueeze(0))[0]
            ds2 = -phi_i * delta_t**2 + dx2

            loss = F.relu(ds2 + eps1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Avg phi loss: {total_loss / len(pairs):.6f}"
            )

    return phi_model


def causal_refine_embeddings_with_phi(
    BERT_embeddings, attention_matrix, phi_model, epsilon=1e-5
):
    """
    Args:
        BERT_embeddings: Tensor of shape (seq_len, hidden_dim)
        attention_matrix: Tensor of shape (seq_len, seq_len), softmaxed attentions
        phi_model: trained GravityField instance
        epsilon: threshold for ds² constraint

    Returns:
        refined_spacetime_coords: (seq_len, 4)
        refined_embeddings: updated BERT_embeddings with causal feedback
    """
    seq_len, hidden_dim = BERT_embeddings.shape
    device = BERT_embeddings.device

    # === Step 1: Project into spacetime ===
    spatial = BERT_embeddings[:, :3]  # take first 3 dims as proxy spatial
    time_vec = torch.zeros(seq_len, device=device)

    # Extract top-k attention edges (causal order)
    k = 2
    pairs = []
    for i in range(seq_len):
        topk_indices = torch.topk(attention_matrix[i], k).indices
        for j in topk_indices:
            if j.item() < i:
                pairs.append((i, j.item()))  # token i attends to token j (past)

    # === Step 2: Project until causal convergence ===
    time_vec = project_until_convergence(
        pairs, spatial, time_vec, phi_model, eps1=epsilon
    )

    # === Step 3: Combine t and x into spacetime coord ===
    spacetime_coords = torch.cat([time_vec.unsqueeze(-1), spatial], dim=-1)

    # === Step 4: Feed back into BERT embedding layer ===
    refined_embeddings = torch.cat(
        [spacetime_coords, BERT_embeddings[:, 3:]], dim=-1
    )  # prepend causal txy

    return spacetime_coords, refined_embeddings


# %%
model = causal_refine_embeddings_with_phi(
    BERT_embeddings=torch.randn(10, 10),
    attention_matrix=torch.randn(10, 10),
    phi_model=GravityField(input_dim=4),
    epsilon=1e-5,
)
print(model)
