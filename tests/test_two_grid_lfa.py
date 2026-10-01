"""Validate the two-grid harmonic-pair block symbol against direct matrices.

For the Birken (2012) model problem on a periodic grid with m cells, the
two-grid error propagation operator

    E_TG = S^nu2 (I - P A_c^{-1} R A) S^nu1

is (exactly) unitarily block-diagonalised by the DFT into the 2x2 blocks
E_hat(theta_k), theta_k = 2*pi*k/m, k = 0, ..., m/2 - 1, computed by
`two_grid_block_symbol_1d`.  These tests assemble E_TG directly as a dense
matrix and check that spectral radius and spectral norm agree with the
maxima over the LFA blocks to machine precision, which is the content of
the exact block-diagonalisation proposition in the paper.
"""

import numpy as np
import pytest
import torch

from neuralfa.lfa import two_grid_block_symbol_1d, two_grid_factor_1d


def upwind_matrix(m):
    """Periodic upwind difference matrix B: (Bu)_i = u_i - u_{i-1}."""
    B = torch.eye(m, dtype=torch.float64)
    B -= torch.diag(torch.ones(m - 1, dtype=torch.float64), -1)
    B[0, m - 1] -= 1.0
    return B


def smoother_matrix(A, alphas, dt_star):
    """Error propagation matrix of one s-stage RK smoothing step."""
    m = A.shape[0]
    I = torch.eye(m, dtype=A.dtype)
    G = I.clone()
    for a in alphas:
        G = I - float(a) * dt_star * (A @ G)
    return I - dt_star * (A @ G)


def direct_two_grid_matrix(m, cfl, alphas, c, nu1=1, nu2=0):
    """Assemble E_TG = S^nu2 (I - P A_c^{-1} R A) S^nu1 as a dense matrix.

    Aggregation transfers: coarse cell i averages fine cells {2i, 2i+1};
    P = 2 R^T.  Coarse operator by re-discretisation, A_c = I + (cfl/2) B_c
    (equal to the Galerkin operator R A P for this transfer pair).
    """
    A = torch.eye(m, dtype=torch.float64) + cfl * upwind_matrix(m)
    Ac = torch.eye(m // 2, dtype=torch.float64) + (cfl / 2.0) * upwind_matrix(m // 2)
    R = torch.zeros(m // 2, m, dtype=torch.float64)
    for i in range(m // 2):
        R[i, 2 * i] = 0.5
        R[i, 2 * i + 1] = 0.5
    P = 2.0 * R.T
    S = smoother_matrix(A, alphas, c / cfl)
    cgc = torch.eye(m, dtype=torch.float64) - P @ torch.linalg.solve(Ac, R @ A)
    return torch.linalg.matrix_power(S, nu2) @ cgc @ torch.linalg.matrix_power(S, nu1)


def lfa_blocks(m, cfl, alphas, c, nu1=1, nu2=0):
    thetas = 2.0 * np.pi * torch.arange(m // 2, dtype=torch.float64) / m
    alphas_t = torch.as_tensor(alphas, dtype=torch.float64)
    c_t = torch.as_tensor(c, dtype=torch.float64)
    return two_grid_block_symbol_1d(cfl, alphas_t, c_t, thetas, nu1=nu1, nu2=nu2)


# (cfl, alphas, c) from Birken's Tables 4.1/4.2 plus a deliberately
# non-optimal point; (nu1, nu2) exercises smoother powers on both sides.
CASES = [
    (3.0, [0.3], 0.93, 1, 0),
    (24.0, [0.33], 0.98, 1, 0),
    (3.0, [0.135, 0.375], 1.37, 1, 0),
    (3.0, [0.3], 0.93, 2, 1),
    (6.0, [0.5], 0.5, 1, 1),
]


@pytest.mark.parametrize("cfl,alphas,c,nu1,nu2", CASES)
def test_block_diagonalisation_spectral_radius(cfl, alphas, c, nu1, nu2):
    m = 64
    E = direct_two_grid_matrix(m, cfl, alphas, c, nu1, nu2)
    rho_direct = torch.linalg.eigvals(E).abs().max()
    blocks = lfa_blocks(m, cfl, alphas, c, nu1, nu2)
    rho_lfa = torch.linalg.eigvals(blocks).abs().amax(dim=-1).max()
    assert float(rho_direct) == pytest.approx(float(rho_lfa), abs=1e-10)


@pytest.mark.parametrize("cfl,alphas,c,nu1,nu2", CASES)
def test_block_diagonalisation_spectral_norm(cfl, alphas, c, nu1, nu2):
    m = 64
    E = direct_two_grid_matrix(m, cfl, alphas, c, nu1, nu2)
    norm_direct = torch.linalg.matrix_norm(E, ord=2)
    blocks = lfa_blocks(m, cfl, alphas, c, nu1, nu2)
    norm_lfa = torch.linalg.matrix_norm(blocks, ord=2).max()
    assert float(norm_direct) == pytest.approx(float(norm_lfa), abs=1e-10)


def test_mesh_independence():
    """The continuum LFA factor upper-bounds every discrete grid and the
    discrete values converge to it as m grows."""
    cfl, alphas, c = 3.0, [0.3], 0.93
    cont = float(two_grid_factor_1d(cfl, torch.tensor(alphas, dtype=torch.float64),
                                    torch.tensor(c, dtype=torch.float64),
                                    n_theta=4001, measure="norm"))
    vals = []
    for m in (16, 64, 256):
        blocks = lfa_blocks(m, cfl, alphas, c)
        vals.append(float(torch.linalg.matrix_norm(blocks, ord=2).max()))
    assert all(v <= cont + 1e-12 for v in vals)
    assert abs(vals[-1] - cont) < 1e-3


def test_norm_dominates_spectral_radius():
    """eta_bar >= rho_LFA always; strictly for the non-normal blocks here."""
    alphas = torch.tensor([0.3], dtype=torch.float64)
    c = torch.tensor(0.93, dtype=torch.float64)
    norm = float(two_grid_factor_1d(3.0, alphas, c, measure="norm"))
    spec = float(two_grid_factor_1d(3.0, alphas, c, measure="spectral"))
    assert norm >= spec
    # Advection-dominated blocks are non-normal: the gap is genuine.
    assert norm > spec * 1.01


def test_gradients_flow_through_norm_loss():
    alphas = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.93, dtype=torch.float64, requires_grad=True)
    eta = two_grid_factor_1d(3.0, alphas, c, measure="norm")
    eta.backward()
    assert alphas.grad is not None and torch.isfinite(alphas.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert alphas.grad.abs().sum() > 0 and c.grad.abs() > 0


def test_two_grid_loss_is_optimisable():
    """A few Adam steps on eta_bar from a poor start must decrease it."""
    alphas = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    start = float(two_grid_factor_1d(3.0, alphas, c, measure="norm").detach())
    opt = torch.optim.Adam([alphas, c], lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = two_grid_factor_1d(3.0, alphas, c, measure="norm")
        loss.backward()
        opt.step()
    end = float(two_grid_factor_1d(3.0, alphas.detach(), c.detach(),
                                   measure="norm"))
    assert end < start
