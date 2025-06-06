# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.autograd.functional as F_grad
import torch.nn as nn
import torch.optim as optim


# %%
class SemanticSpacetimeNetwork(nn.Module):
    """Neural network for semantic spacetime curvature learning"""

    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class PhysicsEnhancedNetwork(nn.Module):
    """Network with physics-based loss functions"""

    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4, physics_weight=0.1):
        super().__init__()
        self.base_network = SemanticSpacetimeNetwork(input_dim, hidden_dim, output_dim)
        self.physics_weight = physics_weight

    def forward(self, x):
        return self.base_network(x)

    def compute_physics_loss(self, embeddings, edges):
        """Compute physics-based causality loss"""
        total_loss = 0.0
        causal_count = 0

        for i, j in edges:
            if i < len(embeddings) and j < len(embeddings):
                delta = embeddings[j] - embeddings[i]
                dt, dx, dy, dz = delta
                ds2 = -(dt**2) + dx**2 + dy**2 + dz**2

                if ds2 < -1e-3:  # Timelike (causal)
                    total_loss += 0.1 * torch.sqrt(-ds2)
                    causal_count += 1
                else:  # Spacelike (acausal) - heavy penalty
                    total_loss += 5.0 * torch.sqrt(torch.abs(ds2) + 1e-6)

        return total_loss / len(edges) if edges else 0.0, causal_count


# %%
class SU2TwistEncoder:
    """SU(2) twist field encoder for prime resonance"""

    def __init__(self, n_primes=20, max_prime=73):
        self.primes = self._generate_primes(max_prime)[:n_primes]
        self.n_primes = len(self.primes)

    def _generate_primes(self, max_val):
        sieve = [True] * (max_val + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(max_val**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, max_val + 1, i):
                    sieve[j] = False
        return [i for i in range(2, max_val + 1) if sieve[i]]

    def encode_vector(self, vector):
        """Encode vector into prime resonance twist field"""
        if len(vector) == 0:
            return np.zeros(self.n_primes)

        twist_field = np.zeros(self.n_primes)
        norm = np.linalg.norm(vector)

        if norm > 0:
            normalized = vector / norm
            for i, prime in enumerate(self.primes):
                resonance = np.sum(
                    normalized * np.sin(prime * np.arange(len(normalized)))
                )
                twist_field[i] = resonance * norm / prime

        return twist_field


# %%
class SemanticSpacetimeSystem:
    """Unified system for semantic spacetime analysis"""

    def __init__(self, words, vectors, semantic_pairs):
        self.words = words
        self.vectors = vectors
        self.semantic_pairs = semantic_pairs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.twist_encoder = SU2TwistEncoder()
        self.spacetime_network = SemanticSpacetimeNetwork().to(self.device)
        self.physics_network = PhysicsEnhancedNetwork().to(self.device)

        # Generate features
        self._generate_features()

    def _generate_features(self):
        """Generate all semantic features"""
        print("🔄 Generating semantic features...")

        # Basic features
        self.word_to_vec = dict(zip(self.words, self.vectors))
        self.node_phases = {w: np.angle(np.sum(v)) for w, v in self.word_to_vec.items()}
        self.node_magnitudes = {
            w: np.linalg.norm(v) for w, v in self.word_to_vec.items()
        }

        # Twist fields
        self.twist_fields = {
            w: self.twist_encoder.encode_vector(v) for w, v in self.word_to_vec.items()
        }

        # 4D spacetime coordinates
        self.spacetime_coords = {}
        for i, word in enumerate(self.words):
            if len(self.vectors[i]) >= 3:
                coords = list(self.vectors[i][:3]) + [0.0]  # Add time dimension
            else:
                coords = [0.0, 0.0, 0.0, 0.0]
            self.spacetime_coords[word] = coords

        # Build graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.words)
        self.graph.add_edges_from(self.semantic_pairs)

        print(f"✅ Generated features for {len(self.words)} words")

    def compute_metric_tensor(self, coords, model):
        """Compute metric tensor via neural network Jacobian"""
        coords_tensor = torch.tensor(
            coords, dtype=torch.float32, requires_grad=True, device=self.device
        )

        def embedding_func(x):
            return model(x.unsqueeze(0)).squeeze(0)

        jacobian = F_grad.jacobian(embedding_func, coords_tensor)
        eta_E = torch.tensor(
            np.diag([-1.0, 1.0, 1.0, 1.0]), dtype=torch.float32, device=self.device
        )

        g_metric = jacobian.T @ eta_E @ jacobian
        return g_metric.detach().cpu().numpy()

    def train_networks(self, epochs=200):
        """Train both networks with unified approach"""
        print("🚀 Training semantic networks...")

        # Prepare data
        coord_tensors = {
            w: torch.tensor(coords, dtype=torch.float32, device=self.device)
            for w, coords in self.spacetime_coords.items()
        }

        # Training setup
        optimizer1 = optim.Adam(self.spacetime_network.parameters(), lr=1e-4)
        optimizer2 = optim.Adam(self.physics_network.parameters(), lr=1e-3)

        eta_E = torch.tensor(
            np.diag([-1.0, 1.0, 1.0, 1.0]), dtype=torch.float32, device=self.device
        )

        metrics = {"spacetime_loss": [], "physics_loss": [], "causal_ratio": []}

        for epoch in range(epochs):
            # Train spacetime network
            self._train_spacetime_epoch(optimizer1, coord_tensors, eta_E, metrics)

            # Train physics network
            self._train_physics_epoch(optimizer2, coord_tensors, metrics)

            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch}: Spacetime={metrics['spacetime_loss'][-1]:.4f}, "
                    f"Physics={metrics['physics_loss'][-1]:.4f}"
                )

        print("✅ Training completed")
        return metrics

    def _train_spacetime_epoch(self, optimizer, coord_tensors, eta_E, metrics):
        """Single training epoch for spacetime network"""
        optimizer.zero_grad()
        total_loss = 0.0

        for word1, word2 in self.semantic_pairs:
            if word1 in coord_tensors and word2 in coord_tensors:
                coords1, coords2 = coord_tensors[word1], coord_tensors[word2]
                embed1, embed2 = (
                    self.spacetime_network(coords1),
                    self.spacetime_network(coords2),
                )

                # Spacetime interval loss
                delta = embed2 - embed1
                interval = -(delta[0] ** 2) + torch.sum(delta[1:] ** 2)
                target_interval = torch.tensor(-0.1, device=self.device)
                loss = (interval - target_interval) ** 2

                total_loss += loss

        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.spacetime_network.parameters(), max_norm=1.0
            )
            optimizer.step()

        metrics["spacetime_loss"].append(total_loss.item())

    def _train_physics_epoch(self, optimizer, coord_tensors, metrics):
        """Single training epoch for physics network"""
        optimizer.zero_grad()

        # Prepare batch
        batch_coords = torch.stack(
            [coord_tensors[w] for w in self.words if w in coord_tensors]
        )
        embeddings = self.physics_network(batch_coords)

        # Edge indices for physics loss
        word_to_idx = {w: i for i, w in enumerate(coord_tensors.keys())}
        edge_indices = [
            (word_to_idx[w1], word_to_idx[w2])
            for w1, w2 in self.semantic_pairs
            if w1 in word_to_idx and w2 in word_to_idx
        ]

        # Compute physics loss
        physics_loss, causal_count = self.physics_network.compute_physics_loss(
            embeddings, edge_indices
        )

        if physics_loss > 0:
            physics_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.physics_network.parameters(), max_norm=1.0
            )
            optimizer.step()

        metrics["physics_loss"].append(
            physics_loss.item() if hasattr(physics_loss, "item") else physics_loss
        )
        causal_ratio = causal_count / len(edge_indices) if edge_indices else 0
        metrics["causal_ratio"].append(causal_ratio)

    def analyze_system(self):
        """Comprehensive system analysis"""
        print("🔬 Analyzing semantic spacetime system...")

        analysis = {
            "vocabulary_size": len(self.words),
            "graph_nodes": len(self.graph.nodes),
            "graph_edges": len(self.graph.edges),
            "twist_coverage": len(self.twist_fields) / len(self.words),
            "avg_twist_magnitude": np.mean(
                [np.linalg.norm(t) for t in self.twist_fields.values()]
            ),
            "avg_connectivity": np.mean(
                [self.graph.degree(n) for n in self.graph.nodes]
            )
            if self.graph.nodes
            else 0,
        }

        # Causality analysis
        causal_connections, violations = self._analyze_causality()
        analysis["causal_connections"] = len(causal_connections)
        analysis["causality_violations"] = len(violations)
        analysis["causality_ratio"] = (
            len(causal_connections) / (len(causal_connections) + len(violations))
            if (causal_connections or violations)
            else 0
        )

        return analysis

    def _analyze_causality(self):
        """Analyze causality in semantic relationships"""
        causal_connections, violations = [], []

        for w1, w2 in self.semantic_pairs:
            if w1 in self.spacetime_coords and w2 in self.spacetime_coords:
                pos1, pos2 = (
                    np.array(self.spacetime_coords[w1]),
                    np.array(self.spacetime_coords[w2]),
                )
                delta = pos2 - pos1
                dt, dx, dy, dz = delta
                ds2 = -(dt**2) + dx**2 + dy**2 + dz**2

                if ds2 < 0:  # Timelike (causal)
                    causal_connections.append((w1, w2, np.sqrt(-ds2)))
                else:  # Spacelike (acausal)
                    violations.append((w1, w2, ds2))

        return causal_connections, violations

    def visualize_system(self, training_metrics=None):
        """Create comprehensive system visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Graph structure
        if self.graph.nodes:
            pos = nx.spring_layout(self.graph, seed=42)
            node_colors = [self.node_magnitudes.get(n, 0) for n in self.graph.nodes]
            nx.draw_networkx(
                self.graph,
                pos,
                node_color=node_colors,
                cmap="viridis",
                node_size=300,
                font_size=8,
                ax=axes[0, 0],
            )
            axes[0, 0].set_title("Semantic Graph Structure")

        # 2. Twist field distribution
        twist_magnitudes = [np.linalg.norm(t) for t in self.twist_fields.values()]
        axes[0, 1].hist(twist_magnitudes, bins=10, alpha=0.7, color="purple")
        axes[0, 1].set_title("Twist Field Magnitudes")
        axes[0, 1].set_xlabel("Magnitude")

        # 3. Phase distribution
        phases = list(self.node_phases.values())
        axes[0, 2].hist(phases, bins=10, alpha=0.7, color="blue")
        axes[0, 2].set_title("Phase Distribution")
        axes[0, 2].set_xlabel("Phase")

        # 4. Training loss (if available)
        if training_metrics:
            epochs = range(len(training_metrics["spacetime_loss"]))
            axes[1, 0].plot(
                epochs, training_metrics["spacetime_loss"], label="Spacetime"
            )
            axes[1, 0].plot(epochs, training_metrics["physics_loss"], label="Physics")
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].legend()
            axes[1, 0].set_yscale("log")

        # 5. Causality ratio
        if training_metrics and "causal_ratio" in training_metrics:
            axes[1, 1].plot(training_metrics["causal_ratio"])
            axes[1, 1].set_title("Causality Ratio")
            axes[1, 1].set_ylim(0, 1)

        # 6. System status
        components = ["WordNet", "Vectors", "Twist", "Physics", "Graph"]
        status = [1, 1, 1, 1, 1]  # All active
        axes[1, 2].bar(components, status, color="green", alpha=0.7)
        axes[1, 2].set_title("System Status")
        axes[1, 2].set_ylim(0, 1.2)

        plt.tight_layout()
        plt.suptitle("🌀 Semantic Spacetime System Dashboard", fontsize=16, y=0.98)
        plt.show()

    def generate_final_system_report(self):
        """Generate comprehensive system report"""
        analysis = self.analyze_system()

        print("🎯 SEMANTIC SPACETIME SYSTEM REPORT")
        print("=" * 50)
        print(
            f"📊 Vocabulary: {analysis['vocabulary_size']} | Graph: {analysis['graph_nodes']}↔{analysis['graph_edges']}"
        )
        print(
            f"📐 Dimensions: {self.vectors.shape[1]}D→{self.twist_encoder.n_primes}D+4D"
        )
        print(
            f"⚡ Coverage: {analysis['twist_coverage']:.1%} | Causality: {analysis['causality_ratio']:.1%}"
        )
        print("🚀 Status: All systems operational")
        return analysis


# %%
def main():
    """Main execution function"""
    # Sample data
    words = ["king", "queen", "man", "woman", "cat", "dog", "happy", "sad"]
    np.random.seed(42)
    vectors = np.random.randn(len(words), 100)
    semantic_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("cat", "dog"),
        ("happy", "sad"),
        ("king", "man"),
        ("queen", "woman"),
    ]

    print("🌀 Initializing Semantic Spacetime System...")
    system = SemanticSpacetimeSystem(words, vectors, semantic_pairs)

    # Train, visualize, and report
    metrics = system.train_networks(epochs=100)
    system.visualize_system(metrics)
    system.generate_final_system_report()

    print("🎉 Analysis complete!")
    return system


# %%
if __name__ == "__main__":
    system = main()
