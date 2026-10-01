"""Tests for the solver-in-the-loop validation (dense multigrid cycles).

Cross-validates the recursive V-cycle implementation against the independent
dense two-grid assembly used in test_two_grid_lfa.py, and checks that the
cycle actually solves the model problem.
"""

import numpy as np
import pytest
import torch

from neuralfa.lfa import two_grid_block_symbol_1d
from neuralfa.solver_validation import (
    build_hierarchy,
    cycle_error_matrix,
    measured_factors,
    two_grid_hierarchy,
    vcycle,
)
from .test_two_grid_lfa import direct_two_grid_matrix


@pytest.mark.parametrize("cfl,alphas,c", [
    (3.0, [0.3], 0.93),
    (24.0, [0.135, 0.375], 1.37),
])
def test_two_level_cycle_matches_direct_assembly(cfl, alphas, c):
    """The recursive cycle truncated to two levels must reproduce the
    independently assembled E_TG = (I - P A_c^-1 R A) S exactly."""
    m = 64
    levels = two_grid_hierarchy(build_hierarchy(m, cfl, periodic=True))
    params = [(np.asarray(alphas), c)] * len(levels)
    E_cycle = cycle_error_matrix(levels, params)
    E_direct = direct_two_grid_matrix(m, cfl, alphas, c, nu1=1, nu2=0).numpy()
    assert np.abs(E_cycle - E_direct).max() < 1e-12


def test_two_level_cycle_matches_lfa_blocks():
    """Measured spectral radius of the periodic two-level cycle equals the
    LFA block prediction at the discrete frequencies."""
    m, cfl, alphas, c = 128, 6.0, [0.315], 0.96
    levels = two_grid_hierarchy(build_hierarchy(m, cfl, periodic=True))
    params = [(np.asarray(alphas), c)] * len(levels)
    rho, _ = measured_factors(cycle_error_matrix(levels, params))
    thetas = 2.0 * np.pi * torch.arange(m // 2, dtype=torch.float64) / m
    blocks = two_grid_block_symbol_1d(
        cfl, torch.tensor(alphas, dtype=torch.float64),
        torch.tensor(c, dtype=torch.float64), thetas)
    rho_lfa = float(torch.linalg.eigvals(blocks).abs().amax(dim=-1).max())
    assert rho == pytest.approx(rho_lfa, abs=1e-10)


@pytest.mark.parametrize("periodic", [True, False])
def test_vcycle_solves_model_problem(periodic):
    """Iterated V-cycles reduce the residual of A u = f by many orders."""
    m, cfl = 128, 3.0
    levels = build_hierarchy(m, cfl, periodic=periodic)
    params = [(np.asarray([0.3]), 0.93)] * len(levels)
    rng = np.random.default_rng(0)
    A = levels[0]["A"]
    f = rng.standard_normal(m)
    u = np.zeros(m)
    for _ in range(30):
        u = vcycle(levels, params, 0, u, f)
    assert np.linalg.norm(f - A @ u) < 1e-8 * np.linalg.norm(f)
