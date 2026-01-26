import torch

from src.losses.spectral import SpectralEMA, batch_transition_matrix


def test_batch_transition_matrix_shape_diag():
    probs = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    labels = torch.tensor([1, 0])
    margins = torch.tensor([0.1, 0.2])
    mat = batch_transition_matrix(probs, labels, margins, gamma=0.5, tau=0.1)
    assert mat.shape == (2, 2)
    assert torch.allclose(torch.diag(mat), torch.zeros(2))


def test_ema_update():
    ema = SpectralEMA(num_classes=2, beta_ema=0.5, device=torch.device("cpu"), n_refresh=1)
    mat = torch.tensor([[0.0, 0.5], [0.2, 0.0]])
    ema.update(mat)
    assert torch.allclose(ema.mat, 0.5 * mat)


def test_power_iteration_sigma_max():
    ema = SpectralEMA(num_classes=2, beta_ema=0.0, device=torch.device("cpu"), n_refresh=1)
    mat = torch.tensor([[0.0, 0.5], [0.2, 0.0]])
    ema.mat = mat
    sigma = ema.sigma_max(t_pi=20)
    sv = torch.linalg.svdvals(mat).max()
    assert torch.allclose(sigma, sv, atol=1e-2)
