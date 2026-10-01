"""Tests for the advection-diffusion extension."""

import numpy as np
import pytest
import torch

from neuralfa.advdiff import (
    StencilNetAD,
    batched_norm_ad,
    net_params_ad,
    stencil_from_params,
    train_ad,
)
from neuralfa.lfa import (
    two_grid_block_symbol_ad_1d,
    two_grid_factor_1d,
    two_grid_factor_ad_1d,
)
from neuralfa.solver_validation import (
    build_hierarchy,
    cycle_error_matrix,
    measured_factors,
    two_grid_hierarchy,
)


def test_mu_zero_reduces_to_pure_advection():
    alphas = torch.tensor([0.135, 0.375], dtype=torch.float64)
    c = torch.tensor(1.37, dtype=torch.float64)
    for nu in (0.5, 3.0, 24.0):
        adv = two_grid_factor_1d(nu, alphas, c, n_theta=129)
        ad = two_grid_factor_ad_1d(nu, 0.0, alphas, c, n_theta=129)
        assert torch.allclose(adv, ad, atol=1e-14)


@pytest.mark.parametrize("nu,mu,alphas,c", [
    (3.0, 0.3, [0.3], 0.93),
    (24.0, 2.4, [0.135, 0.375], 1.37),
    (1.0, 0.5, [0.3], 0.9),
])
def test_advdiff_two_grid_exactness(nu, mu, alphas, c):
    """Measured spectral radius of the dense two-level advection-diffusion
    cycle equals the LFA block prediction at discrete frequencies."""
    m = 64
    levels = two_grid_hierarchy(build_hierarchy(m, nu, mu0=mu, periodic=True))
    params = [(np.asarray(alphas), c)] * len(levels)
    rho, _ = measured_factors(cycle_error_matrix(levels, params))
    thetas = 2.0 * np.pi * torch.arange(m // 2, dtype=torch.float64) / m
    blocks = two_grid_block_symbol_ad_1d(
        nu, mu, torch.tensor(alphas, dtype=torch.float64),
        torch.tensor(c, dtype=torch.float64), thetas)
    rho_lfa = float(torch.linalg.eigvals(blocks).abs().amax(dim=-1).max())
    assert rho == pytest.approx(rho_lfa, abs=1e-10)


def test_batched_norm_matches_unbatched():
    net = StencilNetAD(3)
    nu = torch.tensor([0.5, 3.0, 24.0], dtype=torch.float64)
    mu = torch.tensor([0.05, 1.0, 0.01], dtype=torch.float64)
    with torch.no_grad():
        alphas, beta = net(stencil_from_params(nu, mu))
        eta = batched_norm_ad(nu, mu, alphas, beta, n_theta=65)
        for i in range(3):
            ref = two_grid_factor_ad_1d(float(nu[i]), float(mu[i]),
                                        alphas[i], beta[i], n_theta=65)
            assert torch.allclose(eta[i], ref, atol=1e-12)


def test_short_ad_training_contracts():
    net = train_ad(2, n_steps=200, batch=32, seed=0, verbose=False)
    torch.manual_seed(3)
    nu = torch.exp(torch.empty(64, dtype=torch.float64).uniform_(
        np.log(1 / 24), np.log(24.0)))
    mu = nu / torch.exp(torch.empty(64, dtype=torch.float64).uniform_(
        0.0, np.log(1e4)))
    with torch.no_grad():
        alphas, beta = net(stencil_from_params(nu, mu))
        assert float(batched_norm_ad(nu, mu, alphas, beta).mean()) < 1.0


def test_net_params_ad_shapes():
    net = StencilNetAD(3)
    params = net_params_ad(net, [3.0, 1.5], [0.3, 0.075])
    assert len(params) == 2
    assert params[0][0].shape == (2,) and isinstance(params[0][1], float)
