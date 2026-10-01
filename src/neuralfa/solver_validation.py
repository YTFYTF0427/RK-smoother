"""Solver-in-the-loop validation of the learned RK smoothers (ROADMAP step 1).

Everything reported by `neuralfa.training` is an LFA quantity.  This module
closes the loop by running actual multigrid cycles with the trained network's
parameters and comparing *measured* convergence factors against the LFA
predictions:

1. Two-grid cycle, periodic BC: by the exact block-diagonalisation
   proposition, the measured spectral radius must equal the LFA prediction at
   the discrete frequencies (sanity row for the whole pipeline).
2. Multilevel V-cycle, periodic BC: measured factor vs. the finest-level
   two-grid prediction (the classical heuristic; close but not identical).
3. Multilevel V-cycle, inflow (non-periodic) BC: the matrix is no longer
   circulant, Fourier modes are not eigenvectors, and the theorem does not
   apply; agreement here is an empirical robustness result.
4. MG-preconditioned GMRES: iteration counts with the learned parameters
   vs. Birken's table values vs. no preconditioner.

Run from the repository root (requires trained networks from `neuralfa.training`):

    uv run python -m neuralfa.solver_validation

Outputs:
  results/solver_validation.json       measured/predicted data + GMRES counts
  figs/solver_validation.png          figure
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .lfa import two_grid_block_symbol_1d, two_grid_factor_1d
from .training import (
    BIRKEN_SMOOTHING_OPTIMA,
    StencilNet,
    stencil_from_cfl,
)

COARSEST = 8  # direct solve below this size


# ----------------------------------------------------------------------------
# Model problem and multigrid components (dense numpy, float64)
# ----------------------------------------------------------------------------

def upwind_matrix(m, periodic=True):
    """B with (Bu)_i = u_i - u_{i-1}; inflow BC drops the wrap-around entry."""
    B = np.eye(m) - np.diag(np.ones(m - 1), -1)
    if periodic:
        B[0, m - 1] -= 1.0
    return B


def diffusion_matrix(m, periodic=True):
    """C with (Cu)_i = 2u_i - u_{i-1} - u_{i+1} (central diffusion)."""
    C = 2.0 * np.eye(m) - np.diag(np.ones(m - 1), -1) - np.diag(np.ones(m - 1), 1)
    if periodic:
        C[0, m - 1] -= 1.0
        C[m - 1, 0] -= 1.0
    return C


def aggregation_transfers(m):
    """R averages fine pairs {2i, 2i+1}; P = 2 R^T (piecewise constant)."""
    R = np.zeros((m // 2, m))
    for i in range(m // 2):
        R[i, 2 * i] = R[i, 2 * i + 1] = 0.5
    return R, 2.0 * R.T


def build_hierarchy(m0, nu0, mu0=0.0, periodic=True):
    """Levels of A_l = I + nu_l B_l + mu_l C_l with coarse operators by
    re-discretisation: (nu, mu) -> (nu/2, mu/4) per level.  (For mu = 0
    re-discretisation and Galerkin coincide; for mu > 0 they do not, and we
    follow the re-discretisation convention of the LFA analysis.)
    Each level stores `scale` = nu + 4*mu, the pseudo-timestep normaliser
    dt* = beta / scale."""
    if m0 < COARSEST or m0 % COARSEST or (m0 // COARSEST) & (m0 // COARSEST - 1):
        raise ValueError("m0 must be 8 times a nonnegative power of two")
    levels = []
    m, nu, mu = m0, nu0, mu0
    while m >= COARSEST:
        A = np.eye(m) + nu * upwind_matrix(m, periodic)
        if mu != 0.0:
            A = A + mu * diffusion_matrix(m, periodic)
        R, P = aggregation_transfers(m) if m // 2 >= COARSEST // 2 else (None, None)
        levels.append({"A": A, "nu": nu, "mu": mu,
                       "scale": nu + 4.0 * mu, "R": R, "P": P, "m": m})
        m, nu, mu = m // 2, nu / 2.0, mu / 4.0
    return levels


def rk_smooth(A, u, f, alphas, dt_star):
    """One s-stage RK smoothing step for the pseudo-time ODE u_t* = f - A u."""
    un = u
    v = un
    for a in alphas:
        v = un + a * dt_star * (f - A @ v)
    return un + dt_star * (f - A @ v)


def vcycle(levels, params, l, u, f):
    """Recursive V-cycle with one pre-smoothing step and no post-smoothing
    (nu1 = 1, nu2 = 0, matching the LFA analysis and Birken's cycle);
    the coarsest level is solved directly."""
    A, scale = levels[l]["A"], levels[l]["scale"]
    if l == len(levels) - 1:
        return np.linalg.solve(A, f)
    alphas, beta = params[l]
    u = rk_smooth(A, u, f, alphas, beta / scale)
    r = f - A @ u
    rc = levels[l]["R"] @ r
    if l + 1 == len(levels) - 1:
        ec = np.linalg.solve(levels[l + 1]["A"], rc)
    else:
        ec = vcycle(levels, params, l + 1, np.zeros_like(rc), rc)
    return u + levels[l]["P"] @ ec


def cycle_error_matrix(levels, params):
    """Error propagation matrix of the cycle: columns are the cycle applied to
    unit vectors with f = 0 (the cycle is linear in u for fixed f)."""
    m = levels[0]["m"]
    E = np.empty((m, m))
    for j in range(m):
        e = np.zeros(m)
        e[j] = 1.0
        E[:, j] = vcycle(levels, params, 0, e, np.zeros(m))
    return E


# ----------------------------------------------------------------------------
# Parameter sources
# ----------------------------------------------------------------------------

def load_net(n_stages, results_dir):
    net = StencilNet(n_stages)
    net.load_state_dict(torch.load(results_dir / f"stencil_net_s{n_stages}.pt",
                                   weights_only=True))
    net.eval()
    return net


def net_params(net, nu_levels):
    """(alphas, beta) per level from the trained network."""
    with torch.no_grad():
        a, b = net(stencil_from_cfl(np.asarray(nu_levels, dtype=np.float64)))
    return [(a[i].numpy(), float(b[i])) for i in range(len(nu_levels))]


def birken_params(n_stages, nu_levels):
    """Birken's smoothing optima at the nearest tabulated CFL per level."""
    table = BIRKEN_SMOOTHING_OPTIMA[n_stages]
    out = []
    for nu in nu_levels:
        _, alphas, c = min(table, key=lambda row: abs(math.log(row[0] / nu)))
        out.append((np.asarray(alphas), c))
    return out


# ----------------------------------------------------------------------------
# Measurements and predictions
# ----------------------------------------------------------------------------

def measured_factors(E):
    rho = float(np.abs(np.linalg.eigvals(E)).max())
    nrm = float(np.linalg.svd(E, compute_uv=False).max())
    return rho, nrm


def lfa_predictions(nu0, alphas, beta, m=None):
    """Sampled approximations of continuum eta_bar and rho_TG; if m is given, also the exact discrete
    spectral radius max_k rho(E_hat(theta_k)) for a periodic grid of size m."""
    a_t = torch.as_tensor(np.asarray(alphas), dtype=torch.float64)
    b_t = torch.as_tensor(beta, dtype=torch.float64)
    eta = float(two_grid_factor_1d(nu0, a_t, b_t, n_theta=513, measure="norm"))
    rho = float(two_grid_factor_1d(nu0, a_t, b_t, n_theta=513,
                                   measure="spectral"))
    out = {"eta_bar": eta, "rho_tg": rho}
    if m is not None:
        thetas = 2.0 * np.pi * torch.arange(m // 2, dtype=torch.float64) / m
        blocks = two_grid_block_symbol_1d(nu0, a_t, b_t, thetas)
        out["rho_tg_discrete"] = float(
            torch.linalg.eigvals(blocks).abs().amax(dim=-1).max())
    return out


def two_grid_hierarchy(levels):
    """Truncate a hierarchy to two levels (coarse level solved directly)."""
    return levels[:2]


def gmres_iterations(A, M_op, b, rtol=1e-8, maxiter=400):
    import scipy.sparse.linalg as spla
    count = {"n": 0}

    def cb(_):
        count["n"] += 1

    x, info = spla.gmres(A, b, M=M_op, rtol=rtol, restart=A.shape[0],
                         maxiter=maxiter, callback=cb,
                         callback_type="pr_norm")
    assert info == 0, f"GMRES did not converge (info={info})"
    return count["n"]


def mg_preconditioner(levels, params):
    import scipy.sparse.linalg as spla
    m = levels[0]["m"]

    def apply(v):
        return vcycle(levels, params, 0, np.zeros(m), v)

    return spla.LinearOperator((m, m), matvec=apply)


# ----------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------

def sweep(net, n_stages, nu0_grid, m0=256):
    rows = []
    for nu0 in nu0_grid:
        n_levels = int(math.log2(m0 // COARSEST)) + 1
        nu_levels = [nu0 / 2**l for l in range(n_levels)]
        params = net_params(net, nu_levels)

        pred = lfa_predictions(nu0, *params[0], m=m0)

        per = build_hierarchy(m0, nu0, periodic=True)
        rho_2g, _ = measured_factors(cycle_error_matrix(two_grid_hierarchy(per), params))
        rho_v_per, nrm_v_per = measured_factors(cycle_error_matrix(per, params))

        rows.append({
            "nu0": nu0, **pred,
            "rho_two_grid_periodic": rho_2g,
            "rho_vcycle_periodic": rho_v_per,
            "rho_jacobi": math.sqrt((1+nu0)**2+2*nu0**2)/3/(1+nu0),
        })
        print(f"  [s={n_stages}] nu0={nu0:6.2f}  rho_TG(LFA)={pred['rho_tg']:.4f}  "
              f"  V-cycle={rho_v_per:.4f}  eta_bar={pred['eta_bar']:.4f}")
    return rows


def gmres_experiment(net, n_stages, nu0_values, m0=256, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for nu0 in nu0_values:
        n_levels = int(math.log2(m0 // COARSEST)) + 1
        nu_levels = [nu0 / 2**l for l in range(n_levels)]
        levels = build_hierarchy(m0, nu0, periodic=True)
        A = levels[0]["A"]
        b = rng.standard_normal(m0)
        res = {"nu0": nu0,
               "unpreconditioned": gmres_iterations(A, None, b)}
        for name, params in (("learned", net_params(net, nu_levels)),
                             ("birken", birken_params(n_stages, nu_levels))):
            res[name] = gmres_iterations(A, mg_preconditioner(levels, params), b)
        out.append(res)
        print(f"  [s={n_stages}] nu0={nu0:6.2f}  GMRES iters: "
              f"none={res['unpreconditioned']}  learned={res['learned']}  "
              f"birken={res['birken']}")
    return out


# Categorical palette (validated, light surface).
C_CERT, C_SPEC, C_2G, C_V, C_J, C_INF = ("#2a78d6", "#eb6834", "#1baf7a",
                                    "#eda100", "#2a78d6", "#e87ba4")
INK, MUTED = "#0b0b0b", "#52514e"


def make_figure(all_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, (n_stages, rows) in zip(axes, sorted(all_rows.items())):
        nu = [r["nu0"] for r in rows]
        ax.set_xscale("log")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(r"finest-level CFL number $\nu_0$", fontsize=9,
                      color=INK)
        ax.plot(nu, [r["rho_tg"] for r in rows], color=C_SPEC, lw=2,
                ls=(0, (4, 2)), label=r"$\rho_{\rm TG}$ (LFA)")
        ax.plot(nu, [r["rho_vcycle_periodic"] for r in rows], ls="none",
                marker="s", ms=6, color=C_V,
                label="measured RK V-cycle")
        ax.plot(nu, [r["rho_jacobi"] for r in rows], ls="none",
                marker="s", ms=6, color=C_J,
                label="measured Jacobi V-cycle")
        ax.set_title(f"{n_stages}-stage smoother", fontsize=10, color=INK,
                     loc="left")
        ax.set_ylabel("convergence factor", fontsize=9, color=INK)
    axes[0].legend(fontsize=7.5, frameon=False, labelcolor=INK)
    fig.suptitle("Measured multigrid convergence vs. LFA prediction "
                 "(learned parameters, $m_0 = 256$)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    here = Path(__file__).resolve()
    results_dir = here.parents[2] / "results"
    figs_dir = here.parents[2] / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    nu0_grid = list(np.logspace(0.0, math.log10(24.0), 7))
    all_rows, summary = {}, {}
    for n_stages in (2, 3):
        print(f"Sweeping {n_stages}-stage network (m0=256, 6 levels) ...")
        net = load_net(n_stages, results_dir)
        rows = sweep(net, n_stages, nu0_grid)
        print(f"GMRES experiment ({n_stages}-stage) ...")
        gmres = gmres_experiment(net, n_stages, [3.0, 24.0])
        all_rows[n_stages] = rows
        summary[f"s{n_stages}"] = {"sweep": rows, "gmres": gmres}

        worst = max(abs(r["rho_two_grid_periodic"] - r["rho_tg_discrete"])
                    for r in rows)
        print(f"  exactness check (two-grid vs discrete LFA): "
              f"max |diff| = {worst:.2e}")
        summary[f"s{n_stages}"]["exactness_max_abs_diff"] = worst

    fig_path = figs_dir / "solver_validation.png"
    make_figure(all_rows, fig_path)
    print(f"wrote {fig_path}")
    with open(results_dir / "solver_validation.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {results_dir/'solver_validation.json'}")


if __name__ == "__main__":
    main()
