# !pip install transformers torch matplotlib

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Load BERT base
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()


# Embed a sentence using BERT [CLS] token
def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token embedding


# Sentences
s1 = "The cat sat on the mat."
s2 = "A feline rested upon the rug."

x0 = get_bert_embedding(s1)  # shape: (1, 768)
x1 = get_bert_embedding(s2)


# Analytic continuation geodesic
def psi(z: complex):
    z_torch = torch.tensor([z.real, z.imag], dtype=torch.float32)
    z_complex = z_torch[0] + 1j * z_torch[1]
    return (1 - z_complex) * x0 + z_complex * x1  # shape: (1, 768), complex


# CPT operation
def CPT(psi_fn, z: complex):
    z_bar = complex(z.real, -z.imag)
    return torch.conj(psi_fn(-z_bar))


# Visualize interpolation in PCA space
def visualize_psi_path():
    from sklearn.decomposition import PCA

    points = []
    for t in torch.linspace(0, 1, 100):
        z = complex(t.item(), 0)
        point = psi(z).real.squeeze().numpy()
        points.append(point)
    pca = PCA(n_components=2)
    path_2d = pca.fit_transform(points)
    plt.plot(path_2d[:, 0], path_2d[:, 1])
    plt.title("Semantic Wavefunction Path (Real Geodesic Projection)")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.grid(True)
    plt.show()


visualize_psi_path()


# Schrödinger-like evolution with learnable Hamiltonian
class SemanticHamiltonian(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.H = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, psi_t, dt=0.01):
        i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
        dpsi_dt = -i * psi_t @ self.H
        return psi_t + dpsi_dt * dt  # Euler step


# Example evolution
H = SemanticHamiltonian()
psi_t = psi(0.0).to(torch.cfloat)  # initial state

trajectory = []
for _ in range(100):
    trajectory.append(psi_t.real.detach().numpy())
    psi_t = H(psi_t)

# Project and plot evolved trajectory
from sklearn.decomposition import PCA

trajectory = torch.stack([torch.tensor(x) for x in trajectory])
proj = PCA(n_components=2).fit_transform(trajectory.numpy())
plt.plot(proj[:, 0], proj[:, 1])
plt.title("Schrödinger Evolution of Semantic State")
plt.grid(True)
plt.show()


class SemanticWavefunction:
    def __init__(self, x0: torch.Tensor, x1: torch.Tensor):
        self.x0 = x0.to(torch.cfloat)  # Ensure complex representation
        self.x1 = x1.to(torch.cfloat)

    def psi(self, z: complex) -> torch.Tensor:
        """Interpolated semantic wavefunction between x0 and x1 over complex z."""
        z_tensor = torch.tensor(z.real + 1j * z.imag, dtype=torch.cfloat)
        return (1 - z_tensor) * self.x0 + z_tensor * self.x1

    def CPT(self, z: complex) -> torch.Tensor:
        """Applies CPT symmetry: complex conjugation + parity (negate path) + time reversal."""
        z_bar = complex(z.real, -z.imag)
        return torch.conj(self.psi(-z_bar))

    def cpt_asymmetry(self, z: complex) -> float:
        """Returns the norm difference between the wavefunction and its CPT conjugate."""
        psi_val = self.psi(z)
        psi_cpt = self.CPT(z)
        return torch.norm(psi_val - psi_cpt).item()


class CPTAwareLoss(nn.Module):
    def __init__(
        self,
        wavefunction: SemanticWavefunction,
        z_samples: list = None,
        weight: float = 1.0,
    ):
        """
        :param wavefunction: An instance of SemanticWavefunction.
        :param z_samples: A list of complex numbers to sample along the geodesic.
        :param weight: How strongly to weight the CPT penalty.
        """
        super().__init__()
        self.wavefunction = wavefunction
        self.z_samples = z_samples or [complex(t, 0) for t in torch.linspace(0, 1, 5)]
        self.weight = weight

    def forward(self, dummy_logits=None):
        cpt_penalty = 0.0
        for z in self.z_samples:
            cpt_penalty += self.wavefunction.cpt_asymmetry(z)
        cpt_penalty /= len(self.z_samples)
        return self.weight * cpt_penalty


# loss = decoder_loss + CPTAwareLoss(wave_fn)(None)


class SemanticHamiltonian(nn.Module):
    """A simple Hamiltonian for Schrödinger-like time evolution."""

    def __init__(self, dim=768):
        super().__init__()
        self.H = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def forward(self, psi_t: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        i = torch.tensor(1j)
        dpsi_dt = -i * (psi_t @ self.H)
        return psi_t + dpsi_dt * dt  # Euler integration


def visualize_cpt_asymmetry(wave: SemanticWavefunction):
    import matplotlib.pyplot as plt

    ts = torch.linspace(0, 1, 100)
    diffs = [wave.cpt_asymmetry(complex(t.item(), 0)) for t in ts]
    plt.plot(ts, diffs)
    plt.title("CPT Asymmetry Along Semantic Path")
    plt.xlabel("Interpolation (Re(z))")
    plt.ylabel("||ψ(z) - CPT[ψ(z)]||")
    plt.grid(True)
    plt.show()


x0 = get_bert_embedding("The future is bright.")
x1 = get_bert_embedding("The past is dark.")
wave_fn = SemanticWavefunction(x0, x1)

loss_cpt = CPTAwareLoss(wave_fn, weight=0.5)
print(loss_cpt())  # Use inside training loop

visualize_cpt_asymmetry(wave_fn)  # See asymmetry curve
