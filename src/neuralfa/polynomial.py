"""Convex polynomial reference for the pure-advection two-grid norm.

With one pre-smoothing step, no post-smoothing, and Galerkin aggregation,
each block C(theta) q(A(theta)) has rank at most one. Its squared spectral
norm is its squared Frobenius norm, a convex quadratic in the real
coefficients of q. We solve the sampled minimax epigraph problem and report
a dual lower bound as well as a feasible RK upper bound. This is a numerical
optimality check on the sampled problem, not an interval-arithmetic proof.
"""

import numpy as np
import torch
from math import comb
from scipy.optimize import minimize

from .lfa import two_grid_block_symbol_1d


def polynomial_reference(nu, n_stages, n_theta=513):
    """Optimise q(z)=1+sum c_j z^j, z=A/(1+nu), then recover RK stages.

    The unconstrained polynomial class contains the admissible RK class.
    If the recovered dt*>0 and alphas are in [0,1], its solution is also
    admissible for the paper's RK problem. A simplex combination of the
    sampled quadratic objectives supplies a global lower bound.
    """
    if nu <= 0 or n_stages < 1 or n_theta < 2:
        raise ValueError("Require nu > 0, n_stages >= 1, n_theta >= 2")
    theta = np.linspace(0, np.pi / 2, n_theta)
    C = two_grid_block_symbol_1d(
        nu, [], 0.0, torch.tensor(theta), nu1=0, nu2=0).numpy()
    sigma = 1 + nu * (1 - np.exp(-1j * (theta[:, None] + [0, np.pi])))
    z = sigma / (1 + nu)
    # Centre and scale the basis around the Fourier circle z=1+r exp(it).
    # A monomial basis loses relative accuracy when nu is small and the
    # optimal norm is O(r**s). All nonconstant basis corrections vanish at
    # z=0, preserving q(0)=1 exactly at the algebraic level.
    radius = nu / (1 + nu)
    centred = (z - 1) / radius
    norm_scale = radius**n_stages
    polynomials = [(-centred)**n_stages]
    polynomials += [z * centred**j for j in range(n_stages)]
    basis = np.stack([C * q[:, None, :] for q in polynomials], axis=-1)
    # q_i(c) = c^T H_i c + 2 g_i^T c + b_i, all real.
    gram = np.einsum("nabi,nabj->nij", basis.conj(), basis).real
    H, g, b = gram[:, 1:, 1:], gram[:, 1:, 0], gram[:, 0, 0]

    def values(c):
        # Evaluate from the complex blocks to avoid cancellation near q=0.
        E = np.einsum("nabj,j->nab", basis, np.r_[1.0, c])
        return np.sum(np.abs(E)**2, axis=(1, 2))

    def constraint(x):
        return x[-1] - values(x[:-1])

    def jac(x):
        return np.column_stack([-2 * (H @ x[:-1] + g), np.ones(n_theta)])

    # Zero corrections start at (1-z)^s.
    c0 = np.zeros(n_stages)
    x0 = np.r_[c0, values(c0).max()]
    result = minimize(lambda x: x[-1], x0,
                      jac=lambda x: np.r_[np.zeros(n_stages), 1.0],
                      constraints={"type": "ineq", "fun": constraint, "jac": jac},
                      method="SLSQP", options={"ftol": 1e-13, "maxiter": 1000})
    correction = result.x[:-1]
    primal = float(values(correction).max())
    # SLSQP's nonnegative inequality multipliers give a dual feasible
    # simplex after clipping and normalisation, even before convergence.
    weights = np.maximum(result.multipliers, 0)
    if weights.sum() == 0:
        weights[np.argmax(values(correction))] = 1
    weights /= weights.sum()
    Hd = np.einsum("n,nij->ij", weights, H)
    gd = weights @ g
    cd = np.linalg.solve(Hd, -gd)
    # Each q_i is a sum of squares. Evaluate the quadratic minimiser in
    # that form, avoiding b - g^T H^{-1}g cancellation.
    dual = float(weights @ values(cd))
    c = np.array([(-1)**j * comb(n_stages, j)
                  for j in range(1, n_stages + 1)], dtype=float)
    for j, y in enumerate(correction):
        for k in range(j + 1):
            c[k] += radius**(n_stages - j) * y * comb(j, k) * (-1)**(j-k)
    d = -c[0]
    if d > 0:
        alphas = np.array([-c[j] / (d * c[j - 1])
                           for j in range(1, n_stages)])[::-1]
    else:
        alphas = np.full(n_stages - 1, np.nan)
    feasible = bool(d > 0 and np.all(alphas >= 0) and np.all(alphas <= 1))
    return {"nu": float(nu), "n_stages": n_stages,
            "coefficients": c.tolist(), "alphas": alphas.tolist(),
            "beta": float(d * nu / (1 + nu)), "rk_feasible": feasible,
            "eta": float(norm_scale * np.sqrt(primal)),
            "eta_lower": float(norm_scale * np.sqrt(max(dual, 0))),
            "squared_duality_gap": float(norm_scale**2 * (primal - dual)),
            "dual_stationarity": float(np.linalg.norm(Hd @ cd + gd)),
            "iterations": int(result.nit), "n_theta": n_theta,
            "optimizer_success": bool(result.success), "message": result.message}
