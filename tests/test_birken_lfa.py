"""Validate the LFA smoothing-factor machinery against Birken (2012).

Reference: P. Birken, "Optimizing Runge-Kutta smoothers for unsteady flow
problems", ETNA 39, pp. 298-312, 2012. The reference values used below are
embedded in the tests; a local copy of the paper is not required.

Model problem (Birken Sec. 2): u_t + b u_x = 0 with b > 0, periodic boundary
conditions, first-order upwind in space, implicit Euler in time.  This gives
the linear system with matrix A = I + cfl*B, eq. (2.4).  The s-stage RK
smoother with pseudo-timestep dt* = c*dx/(b*dt) = c/cfl has amplification
polynomial P_s evaluated at z(theta, c; cfl) = -c/cfl - c + c*exp(-i*theta),
eq. (4.1); z depends on (b, dt, dx) only through the CFL number.  Birken's
optimisation problem, eq. (4.2), is

    min_{c, alpha}  max_{|theta| in [pi/2, pi]}  |P_s(z(theta, c; cfl))|^2,

and Tables 4.1 (2-stage) and 4.2 (3-stage) report the optima found by grid
search with resolution 0.005 in alpha, 0.005 in c (0.01 for the 3-stage c),
and 200 frequency points.

These tests check three things:

1. Evaluating our smoothing factor at Birken's published (alpha, c)
   reproduces his published objective values ("Opt-value" = mu^2).
2. Optimising our differentiable smoothing factor recovers his optima --
   matching or beating his grid-search objective, with parameters within
   his grid resolution.
3. Gradients flow through the loss (the point of the differentiable LFA
   formulation).

If these pass, the core recurrence in `rk_smoother_symbol` (used by both the
1D and 2D smoothing-factor losses) matches Birken's stability polynomials
P_2, P_3 (eqs. (3.2), (3.3)) and the CFL parameterisation is implemented
correctly.
"""

import numpy as np
import pytest
import torch

from neuralfa.lfa import (
    smoothing_factor_rk_1d,
    stencil_symbol,
    upwind_advection_symbol_1d,
)

# Birken (2012), Table 4.1: optimal 2-stage smoothers, columns (CFL, alpha,
# c, Opt-value) where Opt-value = max |P_2|^2 over the high frequencies.
BIRKEN_TABLE_4_1 = [
    (1.0, 0.275, 0.745, 0.01923),
    (3.0, 0.300, 0.930, 0.05888),
    (6.0, 0.315, 0.960, 0.08011),
    (9.0, 0.320, 0.975, 0.08954),
    (12.0, 0.325, 0.970, 0.09453),
    (24.0, 0.330, 0.980, 0.10257),
]

# Birken (2012), Table 4.2: optimal 3-stage smoothers, columns (CFL, alpha1,
# alpha2, c, Opt-value).
BIRKEN_TABLE_4_2 = [
    (1.0, 0.120, 0.350, 1.140, 0.001615),
    (3.0, 0.135, 0.375, 1.370, 0.007773),
    (6.0, 0.140, 0.385, 1.445, 0.01233),
    (9.0, 0.140, 0.390, 1.450, 0.01486),
    (12.0, 0.145, 0.395, 1.440, 0.01588),
    (24.0, 0.145, 0.395, 1.495, 0.01772),
]


def mu_squared(cfl, alphas, c, n_theta=1001):
    """Birken's objective |P_s|^2 maximised over high frequencies."""
    alphas_t = torch.as_tensor(alphas, dtype=torch.float64)
    c_t = torch.as_tensor(c, dtype=torch.float64)
    mu = smoothing_factor_rk_1d(cfl, alphas_t, c_t, n_theta=n_theta)
    return float(mu) ** 2


class TestReproduceBirkenTables:
    """Our objective evaluated at Birken's optima must reproduce his values.

    Tolerance: Birken's parameters are grid-quantised (0.005 in alpha and c)
    and his max used 200 frequency points, so a small mismatch in either
    direction is expected.
    """

    @pytest.mark.parametrize("cfl,alpha,c,opt", BIRKEN_TABLE_4_1)
    def test_two_stage(self, cfl, alpha, c, opt):
        val = mu_squared(cfl, [alpha], c)
        assert val == pytest.approx(opt, rel=2e-2)

    @pytest.mark.parametrize("cfl,a1,a2,c,opt", BIRKEN_TABLE_4_2)
    def test_three_stage(self, cfl, a1, a2, c, opt):
        val = mu_squared(cfl, [a1, a2], c)
        assert val == pytest.approx(opt, rel=5e-2)


def optimise_smoother(cfl, alphas0, c0):
    """Minimise the differentiable smoothing factor from a given start.

    Two-stage Adam (coarse then fine learning rate) on the same frequency
    grid used for the final evaluation.
    """
    alphas = torch.tensor(alphas0, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(c0, dtype=torch.float64, requires_grad=True)
    for lr, n_steps in ((5e-3, 1500), (2e-4, 1500)):
        opt = torch.optim.Adam([alphas, c], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = smoothing_factor_rk_1d(cfl, alphas, c, n_theta=1001) ** 2
            loss.backward()
            opt.step()
    return alphas.detach(), c.detach()


class TestRecoverBirkenOptima:
    """Continuous optimisation of our loss must match or beat Birken's
    grid-search optimum, with parameters close to his (within a few times
    his grid resolution -- the optimum is slightly flat in c)."""

    @pytest.mark.parametrize(
        "cfl,alpha_b,c_b,opt", [BIRKEN_TABLE_4_1[1], BIRKEN_TABLE_4_1[5]]
    )
    def test_two_stage_from_neutral_start(self, cfl, alpha_b, c_b, opt):
        # Coarse scan over Birken's parameter box, then gradient refinement.
        best, best_ac = np.inf, ([0.3], 0.9)
        for a in np.linspace(0.05, 0.6, 24):
            for c in np.linspace(0.3, 1.2, 24):
                val = mu_squared(cfl, [a], c, n_theta=257)
                if val < best:
                    best, best_ac = val, ([a], c)
        alphas, c = optimise_smoother(cfl, *best_ac)
        final = mu_squared(cfl, alphas.tolist(), float(c))
        # At least as good as the published grid-search optimum...
        assert final <= opt * 1.01
        # ...and at the same location in parameter space.
        assert abs(float(alphas[0]) - alpha_b) < 0.02
        assert abs(float(c) - c_b) < 0.05

    @pytest.mark.parametrize("cfl,a1,a2,c_b,opt", [BIRKEN_TABLE_4_2[1]])
    def test_three_stage_refines_birken(self, cfl, a1, a2, c_b, opt):
        # Start from Birken's published point: refinement must not drift far
        # and must match or slightly beat his grid-quantised objective.
        alphas, c = optimise_smoother(cfl, [a1, a2], c_b)
        final = mu_squared(cfl, alphas.tolist(), float(c))
        assert final <= opt * 1.01
        assert abs(float(alphas[0]) - a1) < 0.02
        assert abs(float(alphas[1]) - a2) < 0.02
        assert abs(float(c) - c_b) < 0.05


def test_gradients_flow():
    """The smoothing factor must be differentiable in alphas and beta."""
    alphas = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    mu = smoothing_factor_rk_1d(3.0, alphas, c)
    mu.backward()
    assert alphas.grad is not None and torch.isfinite(alphas.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert alphas.grad.abs().sum() > 0
    assert c.grad.abs() > 0


def test_stencil_symbol_consistent_with_1d():
    """The 2D stencil symbol of the upwind advection operator embedded in the
    middle row of a 3x3 stencil must agree with the 1D symbol, with the W
    coefficient contributing exp(-i*theta_x) (regression test for the axis
    convention in `stencil_symbol`)."""
    cfl = 3.0
    # Row [W, C, E] = [-cfl, 1 + cfl, 0]: 1D upwind advection along x.
    stencil = torch.tensor(
        [[0.0, 0.0, 0.0], [-cfl, 1.0 + cfl, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    tx = torch.linspace(-np.pi, np.pi, 33, dtype=torch.float64)
    for ty_val in (0.0, 1.0):  # symbol must be independent of theta_y
        thetas = torch.stack([tx, torch.full_like(tx, ty_val)], dim=-1)
        sym_2d = stencil_symbol(stencil, thetas)
        sym_1d = upwind_advection_symbol_1d(cfl, tx)
        assert torch.allclose(sym_2d, sym_1d, atol=1e-12)
