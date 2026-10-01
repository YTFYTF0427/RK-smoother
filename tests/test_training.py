"""Smoke tests for the stencil->parameters training on the two-grid loss."""

import torch

from neuralfa.training import (
    StencilNet,
    batched_two_grid_norm,
    batched_two_grid_spectral,
    stencil_from_cfl,
    train,
)


def test_batched_spectral_matches_unbatched():
    from neuralfa.lfa import two_grid_factor_1d

    net = StencilNet(3)
    nu = torch.tensor([0.5, 3.0, 24.0], dtype=torch.float64)
    with torch.no_grad():
        alphas, beta = net(stencil_from_cfl(nu))
        rho = batched_two_grid_spectral(nu, alphas, beta, n_theta=129)
        for i, v in enumerate(nu):
            ref = two_grid_factor_1d(float(v), alphas[i], beta[i],
                                     n_theta=129, measure="spectral")
            assert torch.allclose(rho[i], ref, atol=1e-12)


def test_spectral_training_decreases_loss():
    torch.manual_seed(0)
    nu = torch.exp(torch.empty(64, dtype=torch.float64)
                   .uniform_(torch.log(torch.tensor(1 / 24)),
                             torch.log(torch.tensor(24.0))))
    net = train(2, n_steps=150, batch=32, lr=3e-3, seed=0, verbose=False,
                measure="spectral")
    with torch.no_grad():
        alphas, beta = net(stencil_from_cfl(nu))
        rho = float(batched_two_grid_spectral(nu, alphas, beta).mean())
    assert rho < 1.0


def test_batched_loss_matches_unbatched():
    from neuralfa.lfa import two_grid_factor_1d

    net = StencilNet(2)
    nu = torch.tensor([0.5, 3.0, 12.0], dtype=torch.float64)
    with torch.no_grad():
        alphas, beta = net(stencil_from_cfl(nu))
        eta = batched_two_grid_norm(nu, alphas, beta, n_theta=64)
        for i, v in enumerate(nu):
            ref = two_grid_factor_1d(float(v), alphas[i], beta[i], n_theta=64)
            assert torch.allclose(eta[i], ref, atol=1e-12)


def test_short_training_decreases_loss():
    torch.manual_seed(1)
    nu = torch.exp(torch.empty(128, dtype=torch.float64)
                   .uniform_(torch.log(torch.tensor(1 / 24)),
                             torch.log(torch.tensor(24.0))))

    def mean_eta(net):
        with torch.no_grad():
            alphas, beta = net(stencil_from_cfl(nu))
            return float(batched_two_grid_norm(nu, alphas, beta).mean())

    torch.manual_seed(0)
    before = mean_eta(StencilNet(2))
    net = train(2, n_steps=200, batch=32, lr=3e-3, seed=0, verbose=False)
    after = mean_eta(net)
    assert after < before
    assert after < 1.0  # contracts on average after even a short training
