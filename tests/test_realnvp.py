import pytest

pytest.importorskip("torch")
import torch

from spookybench_project.realnvp import RealNVP, RealNVPDecoder


def test_realnvp_roundtrip():
    flow = RealNVP(dim=4, hidden_dim=8, num_layers=2)
    x = torch.randn(5, 4)
    y = flow(x)
    x_rec = flow(y, reverse=True)
    assert torch.allclose(x, x_rec, atol=1e-6)


def test_realnvp_decoder_shape():
    decoder = RealNVPDecoder(input_dim=4, hidden_dim=8, num_layers=2)
    inp = torch.randn(3, 4)
    out = decoder(inp)
    assert out.shape == (3, 1)
