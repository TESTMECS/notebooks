import pytest

pytest.importorskip("torch")
import torch

from TODO.combined import (
    GravityField,
    minkowski_ds2_with_phi,
    project_until_convergence,
)


def test_gravityfield_positive():
    model = GravityField(input_dim=4)
    coords = torch.zeros(2, 4)
    out = model(coords)
    assert out.shape == (2,)
    assert torch.all(out >= 0)


def test_minkowski_ds2_with_phi():
    model = GravityField(input_dim=4)
    t1 = torch.tensor([1.0])
    x1 = torch.tensor([[1.0, 0.0, 0.0]])
    t2 = torch.tensor([0.0])
    x2 = torch.tensor([[0.0, 0.0, 0.0]])
    ds2 = minkowski_ds2_with_phi(t1, x1, t2, x2, model)
    assert ds2.shape == (1,)


def test_project_until_convergence_simple():
    model = GravityField(input_dim=4)
    spatial = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    time_vec = torch.zeros(2)
    pairs = [(1, 0)]
    new_time = project_until_convergence(pairs, spatial, time_vec, model)
    t1, t0 = new_time[1], new_time[0]
    x1, x0 = spatial[1], spatial[0]
    ds2 = minkowski_ds2_with_phi(
        t1.unsqueeze(0), x1.unsqueeze(0), t0.unsqueeze(0), x0.unsqueeze(0), model
    )
    assert ds2.item() < -1e-5
