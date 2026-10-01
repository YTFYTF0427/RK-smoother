"""Independent checks for the new proofs, convex reference, and audit fixes."""

import numpy as np
import pytest
import torch

from neuralfa.lfa import (two_grid_block_symbol_1d, two_grid_factor_1d,
                          stencil_symbol, two_grid_convergence_factor)
from neuralfa.multigrid import assemble_2d_advection_diffusion
from neuralfa.polynomial import polynomial_reference
from neuralfa.review_validation import multilevel_bound
from neuralfa.solver_validation import (build_hierarchy, cycle_error_matrix,
                                        rk_smooth, vcycle)


@pytest.mark.parametrize("stages,nu", [(2,1/24), (3,1/24), (2,3.), (3,24.)])
def test_convex_reference_bounds_and_rk_recovery(stages, nu):
    r = polynomial_reference(nu, stages, n_theta=129)
    assert r["rk_feasible"]
    eta_rk = float(two_grid_factor_1d(
        nu, torch.tensor(r["alphas"], dtype=torch.float64), r["beta"], n_theta=129))
    assert eta_rk == pytest.approx(r["eta"], rel=1e-8, abs=1e-13)
    assert abs(r["eta"]-r["eta_lower"]) < max(1e-11, 1e-7*r["eta"])
    # The dual lower bound cannot exceed independently sampled RK values.
    rng = np.random.default_rng(4)
    for _ in range(5):
        a = torch.tensor(rng.uniform(.1,.5, stages-1))
        eta = float(two_grid_factor_1d(nu, a, .1*nu/(1+nu), n_theta=129))
        assert r["eta_lower"] <= eta + 1e-12


@pytest.mark.parametrize("pre,post", [(1,0),(2,1),(0,2)])
def test_rank_one_powers(pre, post):
    theta = torch.linspace(0, np.pi, 41, dtype=torch.float64)
    E = two_grid_block_symbol_1d(6., [.14,.38], 1.4, theta, pre, post)
    tr = E.diagonal(dim1=-2,dim2=-1).sum(-1)
    assert torch.linalg.det(E).abs().max() < 1e-14
    assert torch.allclose(torch.linalg.matrix_norm(E,ord=2),
                          torch.linalg.matrix_norm(E,ord="fro"), atol=1e-13)
    for k in (2,3,6):
        assert torch.allclose(torch.linalg.matrix_power(E,k),
                              tr[:,None,None]**(k-1)*E, atol=1e-13)


@pytest.mark.parametrize("periodic,mu", [(True,0.), (False,0.), (True,.3)])
def test_recursive_identity_and_bound(periodic, mu):
    levels = build_hierarchy(64,3.,mu0=mu,periodic=periodic)
    # Scale beta to keep dt*=0.4 and exercise nonzero coarse errors.
    params = [(np.array([.3]), .4*x["scale"]) for x in levels]
    E = cycle_error_matrix(levels,params)
    Ec = cycle_error_matrix(levels[1:],params[1:])
    E2 = cycle_error_matrix(levels[:2],params[:2])
    A,R,P = (levels[0][k] for k in ("A","R","P"))
    S = rk_smooth(A,np.eye(64),np.zeros((64,64)),params[0][0],.4)
    perturb = P @ Ec @ np.linalg.solve(levels[1]["A"],R @ A @ S)
    np.testing.assert_allclose(E,E2+perturb,atol=2e-13)
    q,_ = multilevel_bound(levels,params)
    assert np.linalg.norm(E,2) <= q + 1e-12


@pytest.mark.parametrize("n", [1,2,5])
@pytest.mark.parametrize("b", [(1.,-.7),(-.4,1.)])
def test_2d_periodic_assembly_matches_fourier_symbol(n,b):
    A,stencil = assemble_2d_advection_diffusion(n,eps=.02,b=b,dt=.3)
    np.testing.assert_allclose(A @ np.ones(n*n),np.ones(n*n),atol=1e-13)
    yy,xx = np.meshgrid(np.arange(n),np.arange(n),indexing="ij")
    for kx,ky in [(0,0),(1,0),(0,1),(1,2)]:
        tx,ty = 2*np.pi*np.array([kx,ky])/n
        mode = np.exp(1j*(tx*xx+ty*yy)).ravel()
        sigma = stencil_symbol(torch.tensor(stencil),torch.tensor([[tx,ty]])).item()
        np.testing.assert_allclose(A @ mode,sigma*mode,atol=1e-12)


def test_2d_proxy_rejects_unsupported_learned_transfer():
    with pytest.raises(NotImplementedError,match="4x4"):
        two_grid_convergence_factor(torch.eye(3),[.3],.1,
                                    P_symbol_fn=lambda theta: torch.ones(len(theta)))


def test_coarsest_only_vcycle_is_exact():
    levels = build_hierarchy(8,3.)
    f = np.arange(8.)
    u = vcycle(levels,[],0,np.zeros(8),f)
    np.testing.assert_allclose(levels[0]["A"] @ u,f,atol=1e-13)


def test_invalid_hierarchy_size_is_rejected():
    with pytest.raises(ValueError):
        build_hierarchy(30,3.)
