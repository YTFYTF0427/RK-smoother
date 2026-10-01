"""Ablation: spectral-radius vs. norm training objective (ROADMAP step 2).

The paper argues that in the advection-dominated (non-normal) regime a
training loss based on the spectral radius rho_TG can prefer parameters with
poor transient behaviour, and that the norm objective eta_bar certifies
contraction from the first cycle.  This module tests that claim: it trains
the same StencilNet architecture, with the same protocol and seed, on

    L_norm(theta)     = E_nu[ log max_theta ||E_hat(theta)||_2 ]   (eta_bar)
    L_spectral(theta) = E_nu[ log max_theta rho(E_hat(theta)) ]    (rho_TG)

and compares the two resulting networks on
  1. the LFA curves eta_bar(nu) and rho_TG(nu),
  2. exact worst-case transients ||E^k||_2 of the *actual* six-level V-cycle
     error propagation matrix (dense, m0 = 256), k = 1..12,
  3. MG-preconditioned GMRES iteration counts.

Run from the repository root (reuses the norm-trained networks written by
`neuralfa.training`; trains the spectral variants if not yet saved):

    uv run python -m neuralfa.ablation

Outputs:
  results/stencil_net_s{2,3}_spectral.pt   spectral-trained weights
  results/ablation_rho_vs_norm.json        all measurements
  figs/ablation_rho_vs_norm.png           figure
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .solver_validation import (
    COARSEST,
    build_hierarchy,
    cycle_error_matrix,
    gmres_iterations,
    load_net,
    mg_preconditioner,
    net_params,
)
from .training import CFL_MAX, CFL_MIN, StencilNet, evaluate, train

M0 = 256
TRANSIENT_NU0 = 24.0
TRANSIENT_KMAX = 12


def load_or_train_spectral(n_stages, results_dir):
    path = results_dir / f"stencil_net_s{n_stages}_spectral.pt"
    if path.exists():
        net = StencilNet(n_stages)
        net.load_state_dict(torch.load(path, weights_only=True))
        net.eval()
        return net
    print(f"Training {n_stages}-stage network on the SPECTRAL loss ...")
    net = train(n_stages, measure="spectral")
    torch.save(net.state_dict(), path)
    return net


def lfa_curves(net, nu_grid):
    """eta_bar(nu) and rho_TG(nu) for a trained network."""
    from .lfa import two_grid_factor_1d
    etas, rhos = [], []
    with torch.no_grad():
        alphas, beta, _ = evaluate(net, nu_grid)
    for i, nu in enumerate(nu_grid):
        a = torch.tensor(alphas[i], dtype=torch.float64)
        b = torch.tensor(beta[i], dtype=torch.float64)
        etas.append(float(two_grid_factor_1d(nu, a, b, measure="norm")))
        rhos.append(float(two_grid_factor_1d(nu, a, b, measure="spectral")))
    return etas, rhos


def vcycle_matrix_for(net, nu0, m0=M0):
    n_levels = int(math.log2(m0 // COARSEST)) + 1
    nu_levels = [nu0 / 2**l for l in range(n_levels)]
    levels = build_hierarchy(m0, nu0, periodic=True)
    params = net_params(net, nu_levels)
    return cycle_error_matrix(levels, params), levels, params


def transient_norms(E, kmax=TRANSIENT_KMAX):
    """Exact worst-case error amplification ||E^k||_2, k = 1..kmax."""
    out, Ek = [], np.eye(E.shape[0])
    for _ in range(kmax):
        Ek = E @ Ek
        out.append(float(np.linalg.svd(Ek, compute_uv=False).max()))
    return out


# Palette (validated categorical slots): norm-trained blue, rho-trained orange.
C_NORM, C_RHO, C_REF = "#2a78d6", "#eb6834", "#52514e"
INK, MUTED = "#0b0b0b", "#52514e"


def make_figure(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), dpi=200)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    for row, n_stages in enumerate((2, 3)):
        d = data[f"s{n_stages}"]
        nu = d["nu_grid"]

        ax = axes[row, 0]
        ax.set_xscale("log")
        ax.plot(nu, d["norm_net"]["eta"], color=C_NORM, lw=2,
                label=r"$\eta_T$, norm-trained")
        ax.plot(nu, d["spec_net"]["eta"], color=C_RHO, lw=2,
                label=r"$\eta_T$, $\rho$-trained")
        ax.plot(nu, d["norm_net"]["rho"], color=C_NORM, lw=2, ls=(0, (4, 2)),
                label=r"$\rho_{\rm TG}$, norm-trained")
        ax.plot(nu, d["spec_net"]["rho"], color=C_RHO, lw=2, ls=(0, (4, 2)),
                label=r"$\rho_{\rm TG}$, $\rho$-trained")
        ax.set_xlabel(r"CFL number $\nu$", fontsize=9, color=INK)
        ax.set_ylabel("two-grid factor (LFA)", fontsize=9, color=INK)
        ax.set_title(f"{n_stages}-stage: LFA objectives", fontsize=10,
                     color=INK, loc="left")
        if row == 0:
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)

        ax = axes[row, 1]
        ks = list(range(1, TRANSIENT_KMAX + 1))
        ax.set_yscale("log")
        ax.plot(ks, d["transient"]["norm_net"], color=C_NORM, lw=2,
                marker="o", ms=4, label="norm-trained")
        ax.plot(ks, d["transient"]["spec_net"], color=C_RHO, lw=2,
                marker="s", ms=4, label=r"$\rho$-trained")
        eta0 = d["transient"]["eta_bar_norm_net"]
        ax.plot(ks, [eta0**k for k in ks], color=C_REF, lw=1.2,
                ls=(0, (2, 2)), label=r"$\eta_T^{\,k}$ (two-grid comparison)")
        ax.set_xlabel(r"cycles $k$", fontsize=9, color=INK)
        ax.set_ylabel(r"$\|E_{\rm V}^{\,k}\|_2$", fontsize=9, color=INK)
        ax.set_title(
            f"{n_stages}-stage: worst-case transient, "
            rf"$\nu_0 = {TRANSIENT_NU0:g}$",
            fontsize=10, color=INK, loc="left")
        if row == 0:
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)

    fig.suptitle("Spectral-radius vs. norm training objective "
                 "(V-cycle, $m_0 = 256$)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    here = Path(__file__).resolve()
    results_dir = here.parents[2] / "results"
    figs_dir = here.parents[2] / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    nu_grid = list(np.logspace(math.log10(CFL_MIN), math.log10(CFL_MAX), 25))
    data = {}
    for n_stages in (2, 3):
        norm_net = load_net(n_stages, results_dir)
        spec_net = load_or_train_spectral(n_stages, results_dir)

        d = {"nu_grid": nu_grid, "norm_net": {}, "spec_net": {}}
        for key, net in (("norm_net", norm_net), ("spec_net", spec_net)):
            eta, rho = lfa_curves(net, nu_grid)
            d[key]["eta"], d[key]["rho"] = eta, rho
        geo = lambda xs: float(np.exp(np.mean(np.log(xs))))
        print(f"[s={n_stages}] geo-mean over nu grid:  "
              f"norm-net: eta={geo(d['norm_net']['eta']):.4f} "
              f"rho={geo(d['norm_net']['rho']):.4f}   "
              f"rho-net: eta={geo(d['spec_net']['eta']):.4f} "
              f"rho={geo(d['spec_net']['rho']):.4f}")

        # Worst-case transients of the actual V-cycle at large CFL.
        trans = {}
        for key, net in (("norm_net", norm_net), ("spec_net", spec_net)):
            E, levels, params = vcycle_matrix_for(net, TRANSIENT_NU0)
            trans[key] = transient_norms(E)
            if key == "norm_net":
                from .lfa import two_grid_factor_1d
                a = torch.tensor(params[0][0], dtype=torch.float64)
                b = torch.tensor(params[0][1], dtype=torch.float64)
                trans["eta_bar_norm_net"] = float(
                    two_grid_factor_1d(TRANSIENT_NU0, a, b, measure="norm"))
        d["transient"] = trans
        print(f"[s={n_stages}] ||E_V^k||_2 at nu0={TRANSIENT_NU0:g}, "
              f"k=1..4:  norm-net {[round(x,3) for x in trans['norm_net'][:4]]}"
              f"  rho-net {[round(x,3) for x in trans['spec_net'][:4]]}")

        # GMRES with either preconditioner.
        rng = np.random.default_rng(0)
        gmres = []
        for nu0 in (3.0, 24.0):
            n_levels = int(math.log2(M0 // COARSEST)) + 1
            nus = [nu0 / 2**l for l in range(n_levels)]
            levels = build_hierarchy(M0, nu0, periodic=True)
            b_rhs = rng.standard_normal(M0)
            row = {"nu0": nu0}
            for key, net in (("norm_net", norm_net), ("spec_net", spec_net)):
                M_op = mg_preconditioner(levels, net_params(net, nus))
                row[key] = gmres_iterations(levels[0]["A"], M_op, b_rhs)
            gmres.append(row)
            print(f"[s={n_stages}] GMRES nu0={nu0:5.1f}: "
                  f"norm-net={row['norm_net']}  rho-net={row['spec_net']}")
        d["gmres"] = gmres
        data[f"s{n_stages}"] = d

    fig_path = figs_dir / "ablation_rho_vs_norm.png"
    make_figure(data, fig_path)
    print(f"wrote {fig_path}")
    with open(results_dir / "ablation_rho_vs_norm.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {results_dir/'ablation_rho_vs_norm.json'}")


if __name__ == "__main__":
    main()
