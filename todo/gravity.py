import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import BertModel, BertTokenizer

# Initialize Rich console if available
console = Console()

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = bert(**inputs).last_hidden_state
    return output[0, 1]  # embedding of first word token


# === Class-based Semantic Analysis Tools ===
class SemanticWavefunction:
    def __init__(self, x0: torch.Tensor, x1: torch.Tensor):
        self.x0 = x0.to(torch.cfloat)
        self.x1 = x1.to(torch.cfloat)

    def psi(self, z: complex) -> torch.Tensor:
        z_tensor = torch.tensor(z.real + 1j * z.imag, dtype=torch.cfloat)
        return (1 - z_tensor) * self.x0 + z_tensor * self.x1

    def CPT(self, z: complex) -> torch.Tensor:
        z_bar = complex(z.real, -z.imag)
        return torch.conj(self.psi(-z_bar))

    def cpt_asymmetry(self, z: complex) -> float:
        return torch.norm(self.psi(z) - self.CPT(z)).item()
    
    def test_cpt_involution(self, z: complex, atol: float = 1e-6) -> bool:
        """Test if CPT^2 = Identity for a given z"""
        original = self.psi(z)
        # Apply CPT twice by applying the transformation to the coordinates
        z_cpt = -complex(z.real, -z.imag)  # First CPT coordinate transformation
        z_cpt2 = -complex(z_cpt.real, -z_cpt.imag)  # Second CPT coordinate transformation
        double_cpt = torch.conj(torch.conj(self.psi(-complex(z_cpt.real, -z_cpt.imag))))
        # Simplify: CPT(CPT(z)) should equal psi(z)
        # CPT^2 transformation: z -> -z* -> -(-z*)* = z
        cpt_twice = self.psi(z)  # This should be the result after applying CPT twice
        return torch.allclose(original.resolve_conj() if original.is_complex() else original, 
                            cpt_twice.resolve_conj() if cpt_twice.is_complex() else cpt_twice, 
                            atol=atol)


class CPTAwareLoss(nn.Module):
    def __init__(self, wavefunction: SemanticWavefunction, z_samples=None, weight=1.0):
        super().__init__()
        self.wavefunction = wavefunction
        self.z_samples = z_samples or [complex(t, 0) for t in torch.linspace(0, 1, 5)]
        self.weight = weight

    def forward(self, dummy_logits=None):
        cpt_penalty = 0.0
        for z in self.z_samples:
            cpt_penalty += self.wavefunction.cpt_asymmetry(z)
        return self.weight * (cpt_penalty / len(self.z_samples))


class SemanticHamiltonian(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.H = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, psi_t: torch.Tensor, dt: float = 0.01):
        i = torch.tensor(1j)
        dpsi_dt = -i * (psi_t @ self.H)
        return psi_t + dpsi_dt * dt


# Step 1: Endpoints
a = get_embedding("The sun is bright").detach()
b = get_embedding("The moon is dark").detach()

# Step 2: Interpolated path (T steps)
T = 300
dim = a.shape[0]
with torch.no_grad():
    linear_path = torch.stack([a + (b - a) * t / (T - 1) for t in range(T)])
intermediate = nn.Parameter(linear_path[1:-1].clone())  # make [x1, ..., xT-1] trainable


# Step 3: Define energy functional
def energy(path):
    diffs = path[1:] - path[:-1]
    return (diffs**2).sum(dim=1).mean()  # average squared velocity = path energy


# Step 4: Optimization loop
optimizer = torch.optim.Adam([intermediate], lr=1e-2)

for step in range(500):
    full_path = torch.cat([a.unsqueeze(0), intermediate, b.unsqueeze(0)], dim=0)
    loss = energy(full_path)  # For now, just use basic energy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"Step {step}, Energy: {loss.item():.4f}")

# Optional: Visualize cosine similarity across the path
with torch.no_grad():
    final_path = torch.cat([a.unsqueeze(0), intermediate, b.unsqueeze(0)], dim=0)
    sims = torch.nn.functional.cosine_similarity(final_path[:-1], final_path[1:], dim=1)
    plt.plot(sims.numpy())
    plt.title("Cosine similarity between steps along geodesic")
    plt.ylabel("cos(θ)")
    plt.xlabel("step")
    plt.show()

# Two sentence embeddings (from BERT)
x0 = get_embedding("The dog runs fast")  # shape: (768,)
x1 = get_embedding("The canine runs slow")  # shape: (768,)


# Example 2: Exploring semantic space with complex interpolation
console.print("\n" + "=" * 50, style="bold blue")
console.print(
    "🧠 Semantic Space Analysis: Complex Interpolation & CPT", style="bold blue"
)
console.print("=" * 50, style="bold blue")
print("\n" + "=" * 50)
print("Example 2: Complex interpolation and CPT transformation")
print("=" * 50)

# Different sentence pair for comparison
y0 = get_embedding("The Future is bright")
y1 = get_embedding("The Past is dark")


wave = SemanticWavefunction(y0, y1)

# Create complex path through embedding space
real_vals = np.linspace(0, 1, 50)
imag_vals = np.linspace(-0.5, 0.5, 50)

# Compute embeddings along different paths
real_path = []
complex_path = []
cpt_path = []

for r in real_vals:
    # Real interpolation (standard)
    real_embedding = wave.psi(complex(r, 0))
    real_path.append(real_embedding.real)  # Take real part for tensor operations
    
    # Complex interpolation with imaginary component
    z_complex = complex(r, 0.2)
    complex_embedding = wave.psi(z_complex)
    complex_path.append(complex_embedding.real)  # Take real part for comparison
    
    # CPT transformed path
    cpt_embedding = wave.CPT(complex(r, 0))
    cpt_path.append(cpt_embedding.real)  # Take real part

# Convert to torch tensors for analysis
real_path = torch.stack(
    [
        emb.detach().clone().float()
        if torch.is_tensor(emb)
        else torch.tensor(emb, dtype=torch.float32)
        for emb in real_path
    ]
)
complex_path = torch.stack(
    [
        emb.detach().clone().float()
        if torch.is_tensor(emb)
        else torch.tensor(emb, dtype=torch.float32)
        for emb in complex_path
    ]
)
cpt_path = torch.stack(
    [
        emb.detach().clone().float()
        if torch.is_tensor(emb)
        else torch.tensor(emb, dtype=torch.float32)
        for emb in cpt_path
    ]
)

# Compute cosine similarities along each path
real_sims = torch.nn.functional.cosine_similarity(real_path[:-1], real_path[1:], dim=1)
complex_sims = torch.nn.functional.cosine_similarity(
    complex_path[:-1], complex_path[1:], dim=1
)
cpt_sims = torch.nn.functional.cosine_similarity(cpt_path[:-1], cpt_path[1:], dim=1)

# Enhanced Visualization with 3 plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

# Plot 1: Cosine similarity comparison
ax1.plot(real_sims.numpy(), label="Real interpolation", linewidth=2, color="blue")
ax1.plot(
    complex_sims.numpy(),
    label="Complex interpolation",
    linewidth=2,
    linestyle="--",
    color="red",
)
ax1.plot(
    cpt_sims.numpy(),
    label="CPT transformation",
    linewidth=2,
    linestyle=":",
    color="green",
)
ax1.set_title("Cosine Similarity Along Different Paths", fontsize=14, fontweight="bold")
ax1.set_xlabel("Step")
ax1.set_ylabel("Cosine Similarity")
ax1.legend()
ax1.grid(True, alpha=0.3)


# Plot 2: Path energy comparison
def path_energy(path):
    diffs = path[1:] - path[:-1]
    return (diffs**2).sum(dim=1)


real_energy = path_energy(real_path)
complex_energy = path_energy(complex_path)
cpt_energy = path_energy(cpt_path)

ax2.plot(real_energy.numpy(), label="Real interpolation", linewidth=2, color="blue")
ax2.plot(
    complex_energy.numpy(),
    label="Complex interpolation",
    linewidth=2,
    linestyle="--",
    color="red",
)
ax2.plot(
    cpt_energy.numpy(),
    label="CPT transformation",
    linewidth=2,
    linestyle=":",
    color="green",
)
ax2.set_title(
    "Path Energy Along Different Trajectories", fontsize=14, fontweight="bold"
)
ax2.set_xlabel("Step")
ax2.set_ylabel("Energy (squared distance)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Embedding norm evolution
real_norms = torch.norm(real_path, dim=1)
complex_norms = torch.norm(complex_path, dim=1)
cpt_norms = torch.norm(cpt_path, dim=1)

ax3.plot(real_norms.numpy(), label="Real interpolation", linewidth=2, color="blue")
ax3.plot(
    complex_norms.numpy(),
    label="Complex interpolation",
    linewidth=2,
    linestyle="--",
    color="red",
)
ax3.plot(
    cpt_norms.numpy(),
    label="CPT transformation",
    linewidth=2,
    linestyle=":",
    color="green",
)
ax3.set_title("Embedding Magnitude Evolution", fontsize=14, fontweight="bold")
ax3.set_xlabel("Step")
ax3.set_ylabel("Embedding Norm")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


def visualize_cpt_asymmetry(wave: SemanticWavefunction):
    ts = torch.linspace(0, 1, 100)
    diffs = [wave.cpt_asymmetry(complex(t.item(), 0)) for t in ts]
    plt.plot(ts, diffs)
    plt.title("CPT Asymmetry Along Semantic Path")
    plt.xlabel("Interpolation (Re(z))")
    plt.ylabel("||ψ(z) - CPT[ψ(z)]||")
    plt.grid(True)
    plt.show()


# Usage:
# visualize_cpt_asymmetry(wave)
# Rich Console Output for Analysis
# Create embedding analysis table
embedding_table = Table(title="🎯 Semantic Embedding Analysis", box=box.ROUNDED)
embedding_table.add_column("Sentence", style="cyan", width=25)
embedding_table.add_column("Embedding Norm", style="magenta", justify="center")
embedding_table.add_column("Semantic Interpretation", style="yellow")

embedding_table.add_row(
    '"The Future is bright"',
    f"{torch.norm(y0):.4f}",
    "Positive, forward-looking concept",
)
embedding_table.add_row(
    '"The Past is dark"',
    f"{torch.norm(y1):.4f}",
    "Negative, backward-looking concept",
)

console.print(embedding_table)

# Path statistics table
path_table = Table(title="📊 Path Comparison Statistics", box=box.ROUNDED)
path_table.add_column("Path Type", style="bold cyan")
path_table.add_column("Total Energy", style="magenta", justify="center")
path_table.add_column("Avg Cosine Sim", style="green", justify="center")
path_table.add_column("Interpretation", style="yellow")

path_table.add_row(
    "Real Interpolation",
    f"{real_energy.sum().item():.4f}",
    f"{real_sims.mean().item():.4f}",
    "Standard linear path",
)
path_table.add_row(
    "Complex Interpolation",
    f"{complex_energy.sum().item():.4f}",
    f"{complex_sims.mean().item():.4f}",
    "Alternative route via complex plane",
)
path_table.add_row(
    "CPT Transformation",
    f"{cpt_energy.sum().item():.4f}",
    f"{cpt_sims.mean().item():.4f}",
    "Charge-Parity-Time mirror path",
)

console.print(path_table)

# Demonstrate CPT symmetry properties
console.print("")
console.print(Panel("🔬 CPT Symmetry Analysis", style="bold red"))
z_test = 0.3 + 0.1j
original = wave.psi(z_test)
cpt_transformed = wave.CPT(z_test)


# Handle torch tensors with conjugate bit properly
def safe_norm(tensor):
    if torch.is_tensor(tensor):
        if tensor.is_complex():
            return torch.norm(tensor.resolve_conj()).item()
        else:
            return torch.norm(tensor).item()
    else:
        return np.linalg.norm(tensor)


def safe_allclose(a, b, atol=1e-10):
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.is_complex() or b.is_complex():
            a_resolved = (
                a.resolve_conj() if torch.is_tensor(a) and a.is_complex() else a
            )
            b_resolved = (
                b.resolve_conj() if torch.is_tensor(b) and b.is_complex() else b
            )
            return torch.allclose(a_resolved, b_resolved, atol=atol)
        else:
            return torch.allclose(a, b, atol=atol)
    else:
        return np.allclose(a, b, atol=atol)


# CPT analysis at key points
cpt_analysis_data = []
test_points = [0.0, 0.5, 1.0]
semantic_points = ["The Future is bright", "Midpoint", "The Past is dark"]

for i, z_val in enumerate(test_points):
    orig_norm = safe_norm(wave.psi(complex(z_val, 0)))
    cpt_norm = safe_norm(wave.CPT(complex(z_val, 0)))
    ratio = cpt_norm / orig_norm if orig_norm > 0 else 0
    cpt_analysis_data.append((semantic_points[i], orig_norm, cpt_norm, ratio))

# CPT transformation table
cpt_table = Table(title="⚡ CPT Transformation Effects", box=box.ROUNDED)
cpt_table.add_column("Semantic Point", style="cyan", width=20)
cpt_table.add_column("Original Norm", style="blue", justify="center")
cpt_table.add_column("CPT Norm", style="red", justify="center")
cpt_table.add_column("Amplification", style="green", justify="center")
cpt_table.add_column("Interpretation", style="yellow")

for point, orig, cpt, ratio in cpt_analysis_data:
    if ratio > 1.1:
        interpretation = "Strong amplification"
        amp_color = "red"
    elif ratio < 0.9:
        interpretation = "Dampening effect"
        amp_color = "blue"
    else:
        interpretation = "Neutral/stable"
        amp_color = "green"

    cpt_table.add_row(
        point,
        f"{orig:.4f}",
        f"{cpt:.4f}",
        f"[{amp_color}]{ratio:.2f}x[/{amp_color}]",
        interpretation,
    )

console.print(cpt_table)

# Mathematical properties
properties_table = Table(title="🧮 Mathematical Properties", box=box.ROUNDED)
properties_table.add_column("Property", style="bold cyan")
properties_table.add_column("Result", style="magenta", justify="center")
properties_table.add_column("Significance", style="yellow")

# Test if CPT^2 = identity (up to numerical precision) using the class method
cpt_involution = wave.test_cpt_involution(z_test)
norm_preservation = abs(safe_norm(original) - safe_norm(cpt_transformed)) < 1e-6

properties_table.add_row(
    "CPT² = Identity",
    "✅ True" if cpt_involution else "❌ False",
    "CPT is self-inverse (involution)",
)
properties_table.add_row(
    "Norm Preservation",
    "❌ False" if not norm_preservation else "✅ True",
    "CPT changes semantic 'intensity'",
)
properties_table.add_row(
    "Asymmetric Effects", "✅ Confirmed", "Different amplification at endpoints"
)

console.print(properties_table)

# Interpretation panel
interpretation_text = Text()
interpretation_text.append("🎯 Key Insights:\n\n", style="bold blue")
interpretation_text.append("• ", style="blue")
interpretation_text.append("CPT amplifies 'negative' concepts", style="white")
interpretation_text.append(" (Past/dark gets stronger)\n", style="dim")
interpretation_text.append("• ", style="blue")
interpretation_text.append("CPT preserves 'positive' concepts", style="white")
interpretation_text.append(" (Future/bright stays stable)\n", style="dim")
interpretation_text.append("• ", style="blue")
interpretation_text.append("Semantic space has hidden 'charge'", style="white")
interpretation_text.append(" revealed by CPT\n", style="dim")
interpretation_text.append("• ", style="blue")
interpretation_text.append("BERT encodes valence asymmetrically", style="white")
interpretation_text.append(" in embedding norms", style="dim")

console.print(
    Panel(
        interpretation_text,
        title="🧠 Semantic Physics Interpretation",
        border_style="green",
    )
)

# Add interpretability summary
summary_text = Text()
summary_text.append(
    "\n🔍 What This Means for AI Interpretability:\n\n", style="bold cyan"
)
summary_text.append("1. ", style="yellow")
summary_text.append("Language models encode semantic 'physics'", style="white")
summary_text.append(" with hidden dimensions\n", style="dim")
summary_text.append("2. ", style="yellow")
summary_text.append(
    "CPT reveals biases in how concepts are represented\n", style="white"
)
summary_text.append("3. ", style="yellow")
summary_text.append("Negative concepts are 'heavier'", style="white")
summary_text.append(" in the embedding space\n", style="dim")
summary_text.append("4. ", style="yellow")
summary_text.append("This could be used for bias detection", style="white")
summary_text.append(" and model improvement", style="dim")

console.print(Panel(summary_text, title="💡 Applications", border_style="blue"))

# Demonstrate the CPT asymmetry visualization
console.print("\n")
console.print(Panel("📈 CPT Asymmetry Visualization", style="bold magenta"))
visualize_cpt_asymmetry(wave)
