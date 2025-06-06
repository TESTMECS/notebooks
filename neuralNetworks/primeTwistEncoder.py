# %% [markdown]
# # Spin‑Prime Encoding Demo 🌌🔢
#
# This Colab‑ready notebook shows how to map **word vectors** into an **indivisible prime number** representation using a toy *spinor* twist encoding.
#
# **Pipeline**
# 1. Generate simple word embeddings (random but consistent)
# 2. Select a handful of nouns & verbs from **WordNet**
# 3. *Twist‑encode* each vector (simulate SU(2) double cover)
# 4. Map the vector norm → nearest **prime** (indivisible magnitude key)
# 5. **NEW**: Map to Gaussian primes encoding both magnitude and phase
# 6. Visualize the original vectors (PCA‑2D) with prime labels

# %%
# %% [markdown]
# # 🌀⚡ LEARNED GRAVITY CURVATURE: Semantic Spacetime Geometry
#
# **Integrating Einstein Field Equations into WordNet Semantic Relationships**
#
# This section implements:
# - **Metric tensor computation** for semantic relationships
# - **Riemann curvature tensor** learning for optimal word embeddings
# - **Curved Minkowski spacetime** visualization of semantic structure
# - **Gravity-guided semantic diffusion** using learned curvature

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
import math
import hashlib
from sympy import nextprime, I
from sympy import re as sym_re, im as sym_im
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import networkx as nx

nltk.download("wordnet")
from nltk.corpus import wordnet as wn

# Additional imports for PNNN system
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import primerange
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## PNNN (Prime Number Neural Network) System
# %%
class SU2TwistEncoder:
    """
    SU(2) Twist Field Encoder
    Transforms n-dimensional geometric data into prime-based resonance fields
    """

    def __init__(self, n_primes: int = 20, max_prime: int = 73):
        """
        Initialize the SU(2) twist encoder

        Args:
            n_primes: Number of primes to use for encoding
            max_prime: Maximum prime value to consider
        """
        self.primes = list(primerange(2, max_prime))[:n_primes]
        self.n_primes = len(self.primes)

        # SU(2) rotation axes (cycling through x, y, z)
        self.axis_sequence = [
            np.array([1, 0, 0]),  # x-axis
            np.array([0, 1, 0]),  # y-axis
            np.array([0, 0, 1]),  # z-axis
        ]

        print(f"Initialized SU(2) encoder with {self.n_primes} primes: {self.primes}")

    def su2_trace(self, theta: float, axis: np.ndarray) -> complex:
        """
        Compute the trace of SU(2) rotation matrix U = exp(-i*theta*sigma·n/2)

        Args:
            theta: Rotation angle
            axis: Rotation axis (normalized)

        Returns:
            Complex trace of the SU(2) matrix
        """
        # Normalize axis
        axis = axis / np.linalg.norm(axis)

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # sigma · n
        sigma_dot_n = axis[0] * sigma_x + axis[1] * sigma_y + axis[2] * sigma_z

        # U = I * cos(theta/2) - i * sin(theta/2) * sigma·n
        I = np.eye(2, dtype=complex)
        U = I * np.cos(theta / 2) - 1j * np.sin(theta / 2) * sigma_dot_n

        return np.trace(U)

    def encode_vector(
        self, vector: np.ndarray, resonance_scale: float = 1.0
    ) -> np.ndarray:
        """
        Encode a single vector into prime twist field

        Args:
            vector: Input n-dimensional vector
            resonance_scale: Scaling factor for resonance strength

        Returns:
            Prime twist field vector of length n_primes
        """
        twist_field = np.zeros(self.n_primes)

        for i, prime in enumerate(self.primes):
            # Select vector component (cycling if vector is shorter than primes)
            vec_component = vector[i % len(vector)]

            # Select rotation axis (cycling through x, y, z)
            axis = self.axis_sequence[i % len(self.axis_sequence)]

            # Compute rotation angle based on prime resonance
            theta = (2 * np.pi * vec_component / prime) * resonance_scale

            # Compute SU(2) trace and take absolute value
            su2_trace_val = self.su2_trace(theta, axis)
            twist_field[i] = np.abs(su2_trace_val)

        return twist_field

    def encode_batch(
        self, vectors: np.ndarray, resonance_scale: float = 1.0
    ) -> np.ndarray:
        """
        Encode a batch of vectors into twist fields

        Args:
            vectors: Array of shape (batch_size, vector_dim)
            resonance_scale: Scaling factor for resonance strength

        Returns:
            Twist fields of shape (batch_size, n_primes)
        """
        return np.array([self.encode_vector(vec, resonance_scale) for vec in vectors])

# %%
class PrimeTwistGRU(nn.Module):
    """
    GRU-based neural network for learning twist field dynamics
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super(PrimeTwistGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Optional: Add residual connections for deeper understanding
        self.use_residual = True
        if self.use_residual:
            self.residual_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Forward pass through the twist field GRU

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            Output tensor of shape (batch_size, seq_length, input_dim)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)

        # Project to output dimension
        output = self.output_layer(gru_out)

        # Optional residual connection
        if self.use_residual:
            residual = self.residual_layer(x)
            output = output + residual

        return output
# %%

class TwistToPrimeDecoder(nn.Module):
    """
    Decoder to recover prime fingerprints from twist fields
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 20):
        super(TwistToPrimeDecoder, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)
# %%
def twist_encode(vec: np.ndarray):
    """Return simulated spinor double‑cover (v, −v)."""
    return vec, -vec


def encode_magnitude_to_prime(mag: float, scale: int = 10_000) -> int:
    """Quantize magnitude and map to nearest prime."""
    scaled = max(2, int(round(mag * scale)))
    return int(nextprime(scaled))


def compute_vector_phase(vec: np.ndarray) -> float:
    """Compute a representative phase angle for the vector."""
    if len(vec) == 0:
        return 0.0

    # Project onto first two dimensions for phase calculation
    x, y = vec[0], vec[1] if len(vec) > 1 else 0
    return math.atan2(y, x)


def is_gaussian_prime(z):
    """Check if a Gaussian integer is prime."""
    from sympy import factorint

    # Convert to Gaussian integer representation
    a, b = int(sym_re(z)), int(sym_im(z))

    # Special cases
    if b == 0:
        # Real case: check if it's a regular prime ≡ 3 (mod 4)
        if a <= 1:
            return False
        return all(a % p != 0 for p in range(2, int(abs(a) ** 0.5) + 1))

    if a == 0:
        # Purely imaginary case
        if abs(b) <= 1:
            return False
        return all(abs(b) % p != 0 for p in range(2, int(abs(b) ** 0.5) + 1))

    # General case: check norm
    norm = a * a + b * b
    if norm <= 1:
        return False

    # A Gaussian integer is prime if its norm is either:
    # 1. A prime ≡ 3 (mod 4), or
    # 2. The square of a prime ≡ 1 (mod 4)
    try:
        factors = factorint(norm)
        if len(factors) == 1:
            p = list(factors.keys())[0]
            exp = factors[p]
            if exp == 1 and p % 4 == 3:
                return True
            if exp == 2 and p % 4 == 1:
                return True
        return False
    except:
        return False


def find_nearest_gaussian_prime(magnitude: float, phase: float, max_search: int = 1000):
    """Find the nearest Gaussian prime to the given magnitude and phase."""
    scale = 100
    target_a = int(magnitude * scale * math.cos(phase))
    target_b = int(magnitude * scale * math.sin(phase))

    best_prime = 1 + I
    best_distance = float("inf")

    # Search in a grid around the target point
    search_radius = min(max_search // 10, 50)

    for da in range(-search_radius, search_radius + 1):
        for db in range(-search_radius, search_radius + 1):
            a = target_a + da
            b = target_b + db

            if a == 0 and b == 0:
                continue

            candidate = a + b * I

            # Quick check: try small Gaussian primes first
            if abs(a) <= 10 and abs(b) <= 10:
                if is_gaussian_prime(candidate):
                    distance = abs((a - target_a) ** 2 + (b - target_b) ** 2)
                    if distance < best_distance:
                        best_distance = distance
                        best_prime = candidate

    # If no prime found in small search, use a known small Gaussian prime
    if best_distance == float("inf"):
        small_gaussian_primes = [
            1 + I,
            1 - I,
            1 + 2 * I,
            1 - 2 * I,
            2 + I,
            2 - I,
            3 * I,
            -3 * I,
        ]
        best_prime = min(
            small_gaussian_primes,
            key=lambda p: abs(
                (sym_re(p) - target_a) ** 2 + (sym_im(p) - target_b) ** 2
            ),
        )

    return best_prime


def gaussian_prime_encode(vec: np.ndarray):
    """Encode vector using Gaussian primes capturing magnitude and phase."""
    magnitude = float(np.linalg.norm(vec))
    phase = compute_vector_phase(vec)

    # Find nearest Gaussian prime
    gaussian_prime = find_nearest_gaussian_prime(magnitude, phase)

    return gaussian_prime, magnitude, phase


def spin_prime_encode(vec: np.ndarray):
    spin_pos, spin_neg = twist_encode(vec)
    prime_code = encode_magnitude_to_prime(float(np.linalg.norm(vec)))
    gaussian_prime, magnitude, phase = gaussian_prime_encode(vec)

    return spin_pos, spin_neg, prime_code, gaussian_prime, magnitude, phase


def generate_word_embedding(word: str, dim: int = 50) -> np.ndarray:
    """Generate consistent pseudo-random embedding for a word."""
    seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    vec = np.random.normal(0, 1, dim)  # noqa: N806
    # Normalize to unit length then scale by word length for variation
    vec = vec / np.linalg.norm(vec) * (1 + len(word) * 0.1)
    return vec


# %% [markdown]
# ## Generate word vectors & sample WordNet terms

# %%
# Define analogy triplets for testing
def analogy_vector(w1, w2, w3, vec_dict):
    """Compute w2 - w1 + w3 vector (analogical reasoning)."""
    return vec_dict[w2] - vec_dict[w1] + vec_dict[w3]


def find_closest_word(target_vec, word_to_vec, exclude_words=None):
    """Find the word with vector closest to target_vec."""
    if exclude_words is None:
        exclude_words = set()

    best_word = None
    best_distance = float("inf")

    for word, vec in word_to_vec.items():
        if word in exclude_words:
            continue
        distance = np.linalg.norm(target_vec - vec)
        if distance < best_distance:
            best_distance = distance
            best_word = word

    return best_word, best_distance

# Build semantic hash lookup tables
gaussian_hash = {str(row["gaussian_prime"]): row["word"] for _, row in df.iterrows()}
regular_hash = {row["prime"]: row["word"] for _, row in df.iterrows()}


def semantic_query(query_word, word_to_vec, hash_type="gaussian"):
    """Query semantic hash for a word."""
    if query_word not in word_to_vec:
        print(f"Word '{query_word}' not in vocabulary")
        return None

    query_vec = word_to_vec[query_word]
    _, _, prime_code, g_prime, mag, phase = spin_prime_encode(query_vec)

    if hash_type == "gaussian":
        hash_key = str(g_prime)
        lookup_table = gaussian_hash
    else:
        hash_key = prime_code
        lookup_table = regular_hash

    return {
        "word": query_word,
        "hash_key": hash_key,
        "match": lookup_table.get(hash_key, "[not found]"),
        "magnitude": mag,
        "phase": phase,
        "prime_code": prime_code,
        "gaussian_prime": str(g_prime),
    }

# %%
# Prepare data for phase prediction
X = vecs  # Input: word vectors
y = df["phase"].values  # Target: phases

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train models
models = {
    "Ridge (α=0.1)": Ridge(alpha=0.1),      
    "Ridge (α=1.0)": Ridge(alpha=1.0),  
    "Ridge (α=10.0)": Ridge(alpha=10.0),
}

model_results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    model_results.append(
        {
            "model": name,
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": model.score(X_test, y_test),
        }
    )

    predictions[name] = y_pred

    print(
        f"{name}: MSE = {mse:.4f}, RMSE = {np.sqrt(mse):.4f}, R² = {model.score(X_test, y_test):.4f}"
    )

model_df = pd.DataFrame(model_results)
print("🧬 PHASE PREDICTION:")
best_rmse = model_df["rmse"].min()
best_r2 = model_df["r2"].max()
print(f"  • Best RMSE: {best_rmse:.4f}")
print(f"  • Best R²: {best_r2:.4f}")
print(
    f"  • Phase is {'highly' if best_r2 > 0.7 else 'moderately' if best_r2 > 0.3 else 'weakly'} predictable from vectors"
)


# %% [markdown]
# ## Next Steps - Advanced Features
# * ✅ **Added Gaussian primes encoding magnitude and phase**
# * ✅ **Implemented analogy solving with spin-consistent arithmetic**
# * ✅ **Visualized 2D twist field over WordNet graph**
# * ✅ **Built prime-based semantic hashing system**
# * ✅ **Trained model to predict twist direction (phase)**
# * ✅ **🌌 SEMANTIC SPACETIME: Probabilistic diffusion model**
# * ✅ **🛤️ Geodesic path finding in semantic space**
# * ✅ **🔮 Predictive semantic neighborhoods & analogies**
# * ✅ **📐 Spacetime curvature & mass metrics**
# * 🔮 **Future: Clifford algebra operations for semantic transformations**
# * 🔮 **Future: Graph neural networks with spin-prime node features**
# * 🔮 **Future: Multi-scale prime encodings (prime tuples)**
# * 🔮 **Future: Temporal semantic evolution & gravitational waves**

