"""CPU smoke checks; no CAMS data or optional compressor packages required."""
import importlib.util
from pathlib import Path

import torch


def main():
    spec = importlib.util.spec_from_file_location(
        'models_under_test', Path(__file__).resolve().parents[1] / 'src/models.py'
    )
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    gate = models.MultiscaleSpatialAttention3D()
    x = torch.randn(2, 8, 3, 11, 9)
    assert torch.equal(gate(x), x), 'Initial attention must be identity'
    gate(x).square().mean().backward()
    assert gate.mask.weight.grad.abs().sum() > 0, 'Mask must receive gradients'
    for shape in [(1, 3, 17, 19), (1, 6, 96, 84)]:
        model = models.get_model('ResidualSAM3DAutoencoder', input_shape=shape,
                                 latent_dim=64, dropout_rate=0)
        sample = torch.randn(1, *shape)
        output, latent = model(sample)
        assert output.shape == sample.shape
        assert latent.shape == (1, 64)
        torch.testing.assert_close(output, model.decode(latent))
        output.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        print('Forward/backward and code-only reconstruction OK:', shape)
    model = models.get_model('ResidualSAM3DAutoencoder', input_shape=(1, 3, 16, 16),
                             latent_dim=8, dropout_rate=0)
    sample = torch.rand(2, 1, 3, 16, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    with torch.no_grad():
        initial = (model(sample)[0] - sample).square().mean().item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = (model(sample)[0] - sample).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = (model(sample)[0] - sample).square().mean().item()
    assert final < initial, (initial, final)
    print(f'Small-batch learning OK: MSE {initial:.6f} -> {final:.6f}')


if __name__ == '__main__':
    main()
