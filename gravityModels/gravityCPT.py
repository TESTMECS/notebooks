import matplotlib.pyplot as plt
import numpy as np
import torch
from gravity import SemanticWavefunction, get_embedding
from scipy.ndimage import gaussian_filter
from transformers import BertModel, BertTokenizer

# Load model globally for reuse
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
bert.eval()


def attention_vector_from_embedding(embedding: torch.Tensor):
    """
    Use BERT attention to extract a directional vector from a [CLS]-style input.
    Input embedding must be (768,) — returns a (2,) vector [dx, dy]
    """
    # Create fake input using embedding as [CLS] token
    # Append dummy tokens (e.g. [MASK], [SEP]) for context
    dummy_tokens = torch.randn(2, 768) * 0.01
    fake_sequence = torch.cat([embedding.unsqueeze(0), dummy_tokens], dim=0).unsqueeze(
        0
    )

    # Inject as input embeddings
    with torch.no_grad():
        output = bert(inputs_embeds=fake_sequence)
        attentions = output.attentions  # List of (batch, heads, seq, seq)

    # Example: Use last layer attention from [CLS] (token 0) to others
    attn = attentions[-1][0]  # (heads, seq, seq)
    avg_attn = attn.mean(dim=0)  # (seq, seq)
    vector = avg_attn[0, 1:] @ torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]]
    )  # Dummy projection

    return vector.numpy()


def bert_attention_map(z: complex, wave_fn):
    """Return directional vector from BERT attention given a point in semantic space."""
    psi_z = wave_fn.psi(z).real  # Real-valued interpolation
    vec = attention_vector_from_embedding(psi_z)
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-6)


class SemanticGravityField:
    def __init__(self, wave_fn, attention_map, grid_size=50):
        """
        :param wave_fn: SemanticWavefunction object
        :param attention_map: Callable returning attention vectors for points in latent space
        :param grid_size: Number of points in each dimension of 2D grid
        """
        self.wave_fn = wave_fn
        self.attn_fn = attention_map
        self.grid_size = grid_size
        self.X, self.Y = np.meshgrid(
            np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size)
        )
        self.Z = self._compute_potential()

    def _compute_potential(self):
        """Evaluate CPT energy at each grid point (real interpolation only)."""
        V = np.zeros_like(self.X)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                z = complex(self.X[i, j], self.Y[i, j])  # Use real+imag interpolation
                V[i, j] = self.wave_fn.cpt_asymmetry(z)
        return gaussian_filter(V, sigma=1.2)

    def _compute_flow(self):
        """Gradient (force field) from potential."""
        dV_dx, dV_dy = np.gradient(-self.Z)
        return dV_dx, dV_dy

    def simulate_particles(self, n_particles=10, steps=50, dt=0.01):
        """Simulate particles flowing under CPT potential gradient."""
        dV_dx, dV_dy = self._compute_flow()
        particles = np.random.rand(n_particles, 2)
        trails = [particles.copy()]

        for _ in range(steps):
            idx = (particles[:, 1] * (self.grid_size - 1)).astype(int)
            idy = (particles[:, 0] * (self.grid_size - 1)).astype(int)
            vx = dV_dx[idx, idy]
            vy = dV_dy[idx, idy]
            particles[:, 0] += vx * dt
            particles[:, 1] += vy * dt
            particles = np.clip(particles, 0, 1)  # stay in bounds
            trails.append(particles.copy())

        return np.array(trails)

    def visualize(self, show_particles=True):
        """Plot the potential and optional flow lines and particles."""
        plt.figure(figsize=(8, 6))
        plt.contourf(self.X, self.Y, self.Z, levels=50, cmap="plasma")
        plt.colorbar(label="Semantic Potential (CPT Energy)")

        dV_dx, dV_dy = self._compute_flow()
        plt.streamplot(
            self.X, self.Y, dV_dx, dV_dy, color="white", linewidth=0.6, density=1.0
        )

        if show_particles:
            trails = self.simulate_particles()
            for i in range(trails.shape[1]):
                plt.plot(
                    trails[:, i, 0], trails[:, i, 1], alpha=0.7, lw=1.5, color="cyan"
                )

        plt.title("Semantic Gravity Field: CPT Potential and Attention Flow")
        plt.xlabel("Re(z) (Interpolation)")
        plt.ylabel("Im(z) (Metaphoric Drift)")
        plt.grid(False)
        plt.show()


a = get_embedding("The sun is bright").detach()
b = get_embedding("The moon is dark").detach()
wave = SemanticWavefunction(a, b)


def attn_map(z):
    return bert_attention_map(z, wave)


field = SemanticGravityField(wave, attention_map=attn_map)
field.visualize()
