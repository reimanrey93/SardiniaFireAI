import torch

from sardinia_fire.models.hybrid_fire import (
    HybridFireCNN,
    TabularFireMLP,
)


def test_tabular_mlp_forward_backward():
    model = TabularFireMLP(n_features=20)
    x = torch.randn(4, 20)

    result = model(x)
    assert result.shape == (4,)

    result.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_hybrid_cnn_forward_backward():
    model = HybridFireCNN(n_features=20)

    raster = torch.randn(4, 3, 32, 32)
    tabular = torch.randn(4, 20)

    result = model(raster, tabular)
    assert result.shape == (4,)

    result.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_hybrid_uses_both_inputs():
    torch.manual_seed(42)
    model = HybridFireCNN(n_features=20).eval()

    raster = torch.randn(2, 3, 32, 32)
    tabular = torch.randn(2, 20)

    with torch.no_grad():
        reference = model(raster, tabular)
        changed_spatial = model(raster + 5.0, tabular)
        changed_tabular = model(raster, tabular + 5.0)

    assert not torch.allclose(reference, changed_spatial)
    assert not torch.allclose(reference, changed_tabular)
