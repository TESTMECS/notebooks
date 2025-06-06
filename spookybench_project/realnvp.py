import torch
from torch import nn


class CouplingLayer(nn.Module):
    """Affine coupling layer used in RealNVP."""

    def __init__(self, dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:  # noqa: E501
        x_masked = x * self.mask
        s = self.scale_net(x_masked) * (1 - self.mask)
        t = self.translate_net(x_masked) * (1 - self.mask)
        if not reverse:
            y = x_masked + (x * torch.exp(s) + t) * (1 - self.mask)
        else:
            y = x_masked + ((x - t) * torch.exp(-s)) * (1 - self.mask)
        return y


class RealNVP(nn.Module):
    """Simple RealNVP flow with alternating coupling layers."""

    def __init__(self, dim: int, hidden_dim: int, num_layers: int = 4):  # noqa: E501
        super().__init__()
        masks = []
        for i in range(num_layers):
            mask = self._create_mask(dim, parity=i % 2)
            masks.append(mask)
        self.layers = nn.ModuleList([CouplingLayer(dim, hidden_dim, m) for m in masks])  # noqa: E501

    @staticmethod
    def _create_mask(dim: int, parity: int) -> torch.Tensor:
        mask = torch.zeros(dim)
        mask[parity::2] = 1
        return mask

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:  # noqa: E501
        if not reverse:
            for layer in self.layers:
                x = layer(x, reverse=False)
        else:
            for layer in reversed(self.layers):
                x = layer(x, reverse=True)
        return x


class RealNVPDecoder(nn.Module):
    """Decoder that applies a RealNVP flow before a linear readout."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4):  # noqa: E501
        super().__init__()
        self.flow = RealNVP(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.flow(x)
        return self.fc(z)


def test_realnvp_roundtrip():
    flow = RealNVP(dim=4, hidden_dim=8, num_layers=2)
    x = torch.randn(5, 4)
    y = flow(x)
    x_rec = flow(y, reverse=True)
    print(x_rec)
    assert torch.allclose(x, x_rec, atol=1e-6)


def test_realnvp_decoder_shape():
    decoder = RealNVPDecoder(input_dim=4, hidden_dim=8, num_layers=2)
    inp = torch.randn(3, 4)
    out = decoder(inp)
    print(out.shape)
    assert out.shape == (3, 1)


if __name__ == "__main__":
    test_realnvp_roundtrip()
    test_realnvp_decoder_shape()
