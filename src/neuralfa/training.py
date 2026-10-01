"""Train the stencil -> RK-parameter network on the two-grid LFA norm loss.

This implements the amortisation experiment of the paper: a small network
maps the (CFL-parameterised) 3-point stencil of the model problem
A = I + nu*B to RK smoother coefficients (alpha, beta), trained by
minimising the two-grid norm loss

    L_TG(theta) = E_nu [ max_theta || E_hat_{N(x(nu);theta)}(theta; nu) ||_2 ],

The implementation trains on the logarithm of a sampled approximation to
this norm objective. Sampling does not certify a continuum supremum. The
CFL reduction identifies the input family shared across levels; it does not
prove a uniform approximation or full V-cycle convergence guarantee.

Run from the repository root:

    uv run python -m neuralfa.training

Outputs:
  results/stencil_net_s{2,3}.pt        trained weights
  results/tg_training_summary.json     evaluation data (learned vs oracle)
  figs/tg_learned_{2,3}stage.png      figures
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .lfa import two_grid_block_symbol_1d, two_grid_block_symbol_1d_S

CFL_MIN, CFL_MAX = 1.0 / 24.0, 24.0
# Train on one extra octave each side of the evaluation range so that the
# endpoints of [CFL_MIN, CFL_MAX] are interior points of the training
# distribution (edge-of-support extrapolation error otherwise dominates the
# amortisation gap at nu = CFL_MIN).
TRAIN_CFL_MIN, TRAIN_CFL_MAX = CFL_MIN / 2.0, CFL_MAX * 2.0

# Birken (2012) smoothing-factor optima (Tables 4.1 / 4.2), used as the
# classical baseline: parameters optimised for the smoother alone, without
# the coarse-grid coupling and without the norm objective.
BIRKEN_SMOOTHING_OPTIMA = {
    2: [(1.0, [0.275], 0.745), (3.0, [0.3], 0.93), (6.0, [0.315], 0.96),
        (9.0, [0.32], 0.975), (12.0, [0.325], 0.97), (24.0, [0.33], 0.98)],
    3: [(1.0, [0.12, 0.35], 1.14), (3.0, [0.135, 0.375], 1.37),
        (6.0, [0.14, 0.385], 1.445), (9.0, [0.14, 0.39], 1.45),
        (12.0, [0.145, 0.395], 1.44), (24.0, [0.145, 0.395], 1.495)],
}


def stencil_from_cfl(nu):
    """Model-problem stencil rows [-nu, 1+nu, 0] for a batch of CFL numbers."""
    nu = torch.as_tensor(nu, dtype=torch.float64).reshape(-1)
    return torch.stack([-nu, 1.0 + nu, torch.zeros_like(nu)], dim=-1)


def batched_two_grid_norm(nu, alphas, beta, nu1=1, nu2=0, n_theta=64):
    """eta_bar = max_theta ||E_hat(theta)||_2 for a batch of CFL numbers.

    Shapes: nu (B,), alphas (B, s-1), beta (B,).  Returns (B,).
    Broadcasting through `two_grid_block_symbol_1d` requires cfl and beta as
    (B, 1) and the stage coefficients iterable over stages, i.e. (s-1, B, 1).
    """
    thetas = torch.linspace(0.0, math.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_1d(
        nu.reshape(-1, 1),
        alphas.T.unsqueeze(-1),
        beta.reshape(-1, 1),
        thetas,
        nu1=nu1,
        nu2=nu2,
    )
    return torch.linalg.matrix_norm(E, ord=2).amax(dim=-1)


def batched_two_grid_spectral(nu, alphas, beta, nu1=1, nu2=0, n_theta=64):
    """Sampled rho_TG = max_theta rho(E_hat(theta)) for a batch of CFL numbers.

    Same shapes as `batched_two_grid_norm`.  Uses the closed-form eigenvalues
    of the 2x2 blocks, lambda = (tr +- sqrt(tr^2 - 4 det))/2, which is
    differentiable wherever the eigenvalues are simple (generic here) and
    avoids the fragility of eigendecomposition backward passes.
    """
    thetas = torch.linspace(0.0, math.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_1d(
        nu.reshape(-1, 1),
        alphas.T.unsqueeze(-1),
        beta.reshape(-1, 1),
        thetas,
        nu1=nu1,
        nu2=nu2,
    )
    tr = E[..., 0, 0] + E[..., 1, 1]
    det = E[..., 0, 0] * E[..., 1, 1] - E[..., 0, 1] * E[..., 1, 0]
    disc = torch.sqrt(tr * tr - 4.0 * det)
    rho = torch.maximum(((tr + disc) / 2).abs(), ((tr - disc) / 2).abs())
    return rho.amax(dim=-1)

def batched_two_grid_norm_S(nu, alphas, beta, nu1=1, nu2=0, n_theta=64):
    """eta_bar = max_theta ||E_hat(theta)||_2 for a batch of CFL numbers.

    Shapes: nu (B,), alphas (B, s-1), beta (B,).  Returns (B,).
    Broadcasting through `two_grid_block_symbol_1d` requires cfl and beta as
    (B, 1) and the stage coefficients iterable over stages, i.e. (s-1, B, 1).
    """
    thetas = torch.linspace(0.0, math.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_1d_S(
        nu.reshape(-1, 1),
        alphas.T.unsqueeze(-1),
        beta.reshape(-1, 1),
        thetas,
        nu1=nu1,
        nu2=nu2,
    )
    return torch.linalg.matrix_norm(E, ord=2).amax(dim=-1)


def batched_two_grid_spectral_S(nu, alphas, beta, nu1=1, nu2=0, n_theta=64):
    """Sampled rho_TG = max_theta rho(E_hat(theta)) for a batch of CFL numbers.

    Same shapes as `batched_two_grid_norm`.  Uses the closed-form eigenvalues
    of the 2x2 blocks, lambda = (tr +- sqrt(tr^2 - 4 det))/2, which is
    differentiable wherever the eigenvalues are simple (generic here) and
    avoids the fragility of eigendecomposition backward passes.
    """
    thetas = torch.linspace(0.0, math.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_1d_S(
        nu.reshape(-1, 1),
        alphas.T.unsqueeze(-1),
        beta.reshape(-1, 1),
        thetas,
        nu1=nu1,
        nu2=nu2,
    )
    tr = E[..., 0, 0] + E[..., 1, 1]
    det = E[..., 0, 0] * E[..., 1, 1] - E[..., 0, 1] * E[..., 1, 0]
    disc = torch.sqrt(tr * tr - 4.0 * det)
    rho = torch.maximum(((tr + disc) / 2).abs(), ((tr - disc) / 2).abs())
    return rho.amax(dim=-1)



class StencilNet(nn.Module):
    """MLP mapping the model-problem stencil to RK smoother parameters.

    Input: stencil rows (B, 3) of the form [-nu, 1+nu, 0].  The forward pass
    extracts the features [nu, log nu] (the stencil is in bijection with nu,
    and the CFL range spans three decades, so a log feature is essential).
    Output: alphas in (0,1)^(s-1) via sigmoid, beta > 0 via softplus.
    """

    def __init__(self, n_stages=2, hidden=64):
        super().__init__()
        self.n_stages = n_stages
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_stages),
        ).double()

    def forward(self, stencil):
        nu = -stencil[..., 0:1]
        feats = torch.cat([nu / CFL_MAX, torch.log(nu)], dim=-1)
        out = self.mlp(feats)
        alphas = torch.sigmoid(out[..., : self.n_stages - 1])
        beta = nn.functional.softplus(out[..., -1])
        return alphas, beta


def train(n_stages, n_steps=3000, batch=64, lr=1e-3, seed=0, verbose=True,
          measure="norm"):
    """Train a StencilNet on E_nu[log f(nu)], with f the two-grid norm
    (measure="norm", the sampled norm estimate) or spectral radius
    (measure="spectral", the asymptotic factor rho_TG)."""
    loss_fn = {"norm": batched_two_grid_norm,
               "spectral": batched_two_grid_spectral}[measure]
    torch.manual_seed(seed)
    net = StencilNet(n_stages)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    log_lo, log_hi = math.log(TRAIN_CFL_MIN), math.log(TRAIN_CFL_MAX)
    for step in range(n_steps):
        nu = torch.exp(torch.empty(batch, dtype=torch.float64)
                       .uniform_(log_lo, log_hi))
        alphas, beta = net(stencil_from_cfl(nu))
        eta = loss_fn(nu, alphas, beta)
        # Train on log eta_bar (the convergence *rate*): the arithmetic mean
        # of eta_bar gives negligible gradient weight to small-CFL samples,
        # where the optimal factor is ~1e-3, and the network would fit that
        # tail poorly in relative terms.  The log equalises relative accuracy
        # across the CFL range (geometric-mean contraction factor).
        loss = torch.log(eta).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == n_steps - 1):
            print(f"  [s={n_stages}/{measure}] step {step:4d}  geo-mean = "
                  f"{float(loss.detach().exp()):.4f}")
    return net


def oracle(nu_values, n_stages, n_theta=513):
    """Per-CFL local optimisation of the sampled norm (historical reference).

    Marches from the largest to the smallest CFL with warm starts
    (continuation), optimising unconstrained parameters through the same
    sigmoid/softplus transforms as the network.
    """
    a_raw = torch.full((n_stages - 1,), -0.8, dtype=torch.float64,
                       requires_grad=True)
    b_raw = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    out = {}
    for nu in sorted(nu_values, reverse=True):
        for lr, steps in ((1e-2, 800), (1e-3, 300)):
            opt = torch.optim.Adam([a_raw, b_raw], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                alphas = torch.sigmoid(a_raw).unsqueeze(0)
                beta = nn.functional.softplus(b_raw).reshape(1)
                loss = batched_two_grid_norm(
                    torch.tensor([nu], dtype=torch.float64), alphas, beta,
                    n_theta=n_theta)
                loss.backward()
                opt.step()
        with torch.no_grad():
            alphas = torch.sigmoid(a_raw)
            beta = nn.functional.softplus(b_raw)
            eta = batched_two_grid_norm(
                torch.tensor([nu], dtype=torch.float64),
                alphas.unsqueeze(0), beta.reshape(1), n_theta=n_theta)
        out[nu] = (alphas.tolist(), float(beta), float(eta))
    return out


def evaluate(net, nu_values, n_theta=513):
    with torch.no_grad():
        nu = torch.tensor(nu_values, dtype=torch.float64)
        alphas, beta = net(stencil_from_cfl(nu))
        eta = batched_two_grid_norm(nu, alphas, beta, n_theta=n_theta)
    return alphas.numpy(), beta.numpy(), eta.numpy()

def evaluate_S(net, nu_values, n_theta=513):
    with torch.no_grad():
        nu = torch.tensor(nu_values, dtype=torch.float64)
        alphas, beta = net(stencil_from_cfl(nu))
        eta = batched_two_grid_norm_S(nu, alphas, beta, n_theta=n_theta)
    return alphas.numpy(), beta.numpy(), eta.numpy()


def birken_baseline(n_stages, n_theta=513):
    pts = []
    for nu, alphas, c in BIRKEN_SMOOTHING_OPTIMA[n_stages]:
        eta = batched_two_grid_norm(
            torch.tensor([nu], dtype=torch.float64),
            torch.tensor([alphas], dtype=torch.float64),
            torch.tensor([c], dtype=torch.float64), n_theta=n_theta)
        pts.append((nu, alphas, c, float(eta)))
    return pts


# Categorical palette (validated, light surface): network / oracle / Birken.
C_NET, C_ORACLE, C_BIRKEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


def make_figure(n_stages, nu_grid, net_eval, oracle_out, birken_pts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net_alphas, net_beta, net_eta = net_eval
    o_alphas = np.array([oracle_out[nu][0] for nu in nu_grid])
    o_beta = np.array([oracle_out[nu][1] for nu in nu_grid])
    o_eta = np.array([oracle_out[nu][2] for nu in nu_grid])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(r"CFL number $\nu$", fontsize=9, color=INK)

    # (a) stage coefficients alpha_j
    ax = axes[0, 0]
    for j in range(n_stages - 1):
        ls = "-" if j == 0 else (0, (4, 2))
        ax.plot(nu_grid, net_alphas[:, j], color=C_NET, lw=2, ls=ls)
        ax.plot(nu_grid, o_alphas[:, j], color=C_ORACLE, lw=2, ls=ls,
                alpha=0.9)
        ax.annotate(rf"$\alpha_{j+1}$", (nu_grid[-1], net_alphas[-1, j]),
                    textcoords="offset points", xytext=(5, 0),
                    fontsize=9, color=INK)
    for nu, alphas, _, _ in birken_pts:
        for j, a in enumerate(alphas):
            ax.plot(nu, a, marker="D", ms=5, color=C_BIRKEN, ls="none")
    ax.set_ylabel(r"$\alpha_j$", fontsize=10, color=INK)
    ax.set_title("Stage coefficients", fontsize=10, color=INK, loc="left")

    # (b) pseudo-timestep scaling beta
    ax = axes[0, 1]
    ax.plot(nu_grid, net_beta, color=C_NET, lw=2, label="network")
    ax.plot(nu_grid, o_beta, color=C_ORACLE, lw=2, ls=(0, (4, 2)),
            label="oracle (per-CFL opt.)")
    ax.plot([p[0] for p in birken_pts], [p[2] for p in birken_pts],
            marker="D", ms=5, color=C_BIRKEN, ls="none",
            label="Birken smoothing optima")
    ax.set_ylabel(r"$\beta$", fontsize=10, color=INK)
    ax.set_title("Pseudo-timestep scaling", fontsize=10, color=INK,
                 loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)

    # (c) achieved two-grid factor eta_bar
    ax = axes[1, 0]
    ax.plot(nu_grid, net_eta, color=C_NET, lw=2, label="network")
    ax.plot(nu_grid, o_eta, color=C_ORACLE, lw=2, ls=(0, (4, 2)),
            label="oracle")
    ax.plot([p[0] for p in birken_pts], [p[3] for p in birken_pts],
            marker="D", ms=5, color=C_BIRKEN, ls="none",
            label="Birken smoothing optima")
    ax.set_ylabel(r"$\bar\eta$ (two-grid norm)", fontsize=10, color=INK)
    ax.set_title("Certified two-grid contraction factor", fontsize=10,
                 color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)

    # (d) amortisation gap
    ax = axes[1, 1]
    gap = 100.0 * (net_eta - o_eta) / o_eta
    ax.plot(nu_grid, gap, color=C_NET, lw=2)
    ax.axhline(0.0, color=MUTED, lw=1)
    ax.set_ylabel(r"$(\bar\eta_{\rm net} - \bar\eta^*)/\bar\eta^*$ [%]",
                  fontsize=10, color=INK)
    ax.set_title("Amortisation gap vs oracle", fontsize=10, color=INK,
                 loc="left")

    fig.suptitle(
        f"{n_stages}-stage RK smoother learned from the two-grid LFA norm "
        f"loss (model problem)", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    here = Path(__file__).resolve()
    results_dir = here.parents[2] / "results"
    figs_dir = here.parents[2] / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    nu_grid = list(np.logspace(math.log10(CFL_MIN), math.log10(CFL_MAX), 25))
    summary = {}
    for n_stages in (2, 3):
        print(f"Training {n_stages}-stage network on L_TG ...")
        net = train(n_stages)
        torch.save(net.state_dict(),
                   results_dir / f"stencil_net_s{n_stages}.pt")

        print(f"Computing per-CFL oracle ({len(nu_grid)} points) ...")
        oracle_out = oracle(nu_grid, n_stages)
        net_eval = evaluate(net, nu_grid)
        birken_pts = birken_baseline(n_stages)

        fig_path = figs_dir / f"tg_learned_{n_stages}stage.png"
        make_figure(n_stages, np.array(nu_grid), net_eval, oracle_out,
                    birken_pts, fig_path)
        print(f"  wrote {fig_path}")

        net_alphas, net_beta, net_eta = net_eval
        o_eta = np.array([oracle_out[nu][2] for nu in nu_grid])
        gap = (net_eta - o_eta) / o_eta
        summary[f"s{n_stages}"] = {
            "nu_grid": nu_grid,
            "net_alphas": net_alphas.tolist(),
            "net_beta": net_beta.tolist(),
            "net_eta": net_eta.tolist(),
            "oracle": {str(k): v for k, v in oracle_out.items()},
            "birken_smoothing_pts": birken_pts,
            "max_rel_gap": float(gap.max()),
            "median_rel_gap": float(np.median(gap)),
        }
        print(f"  amortisation gap: median {np.median(gap):+.2%}, "
              f"max {gap.max():+.2%}")

        # Hierarchy walk: one network serving every level, finest CFL = 24.
        print(f"  hierarchy walk (nu_0 = 24, {n_stages}-stage):")
        nus = [24.0 / 2**l for l in range(9)]
        a, b, e = evaluate(net, nus)
        a, b, e_S = evaluate_S(net,nus)
        for l, nu in enumerate(nus):
            o_e = oracle_out.get(nu, (None, None, None))[2]
            oe_str = f"  oracle {o_e:.4f}" if o_e is not None else ""
            print(f"    level {l}: nu = {nu:7.3f}  alpha = "
                  f"{np.round(a[l], 3).tolist()}  beta = {b[l]:.3f}  "
                  f"eta_bar = {e[l]:.4f}{oe_str}   eta_S = {e_S[l]:.4f}")

    with open(results_dir / "tg_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {results_dir/'tg_training_summary.json'}")


if __name__ == "__main__":
    main()
