"""Advection-diffusion extension: a two-parameter family (ROADMAP step 3).

Adds diffusion to the model problem: u_t - eps*u_xx + b*u_x = f, implicit
Euler, upwind advection, central diffusion, periodic BC.  The system matrix
is A = I + nu*B + mu*C with the CFL number nu = b*dt/dx and the diffusion
number mu = eps*dt/dx^2; the mesh Peclet number is Pe = nu/mu.  Standard
coarsening maps (nu, mu) -> (nu/2, mu/4), i.e. Pe doubles per level, so by
the advection-diffusion corollary of the CFL proposition a single network
trained over the (nu, mu) region serves every level of every hierarchy whose
per-level parameters stay in that region.

The network now receives a two-dimensional family. This tests amortisation
across two physical groups, without establishing a need for neural networks
over interpolation. Training uses a sampled norm on (nu, Pe) in
[1/48, 48] x [0.5, 2e4], followed by comparison with a local optimisation
reference and dense solver checks. See REVIEW.md for the limitations.

Run from the repository root:

    uv run python -m neuralfa.advdiff

Outputs:
  results/stencil_net_ad_s{2,3}.pt     trained weights
  results/advdiff.json                 evaluation data
  figs/advdiff_learned.png            figure (3-stage)
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .lfa import two_grid_block_symbol_ad_1d, two_grid_factor_ad_1d, two_grid_block_symbol_1d_S
from .solver_validation import (
    COARSEST,
    build_hierarchy,
    cycle_error_matrix,
    measured_factors,
    two_grid_hierarchy,
    vcycle,
    rk_smooth
)

NU_MIN, NU_MAX = 1.0 / 24.0, 24.0
PE_MIN, PE_MAX = 1.0, 1.0e4
# Train one octave wider than the evaluation region in each direction.
TRAIN_LOG_NU = (math.log(NU_MIN / 2), math.log(NU_MAX * 2))
TRAIN_LOG_PE = (math.log(PE_MIN / 2), math.log(PE_MAX * 2))
M0 = 256
TRANSIENT_NU0 = 24.0
TRANSIENT_KMAX = 12


def stencil_from_params(nu, mu):
    """Advection-diffusion stencil rows [-nu - mu, 1 + nu + 2*mu, -mu]."""
    nu = torch.as_tensor(nu, dtype=torch.float64).reshape(-1)
    mu = torch.as_tensor(mu, dtype=torch.float64).reshape(-1)
    return torch.stack([-nu - mu, 1.0 + nu + 2.0 * mu, -mu], dim=-1)


def batched_norm_ad(nu, mu, alphas, beta, n_theta=64):
    """eta_bar over a batch; shapes nu, mu, beta (B,), alphas (B, s-1)."""
    thetas = torch.linspace(0.0, math.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_ad_1d(
        nu.reshape(-1, 1), mu.reshape(-1, 1),
        alphas.T.unsqueeze(-1), beta.reshape(-1, 1), thetas)
    return torch.linalg.matrix_norm(E, ord=2).amax(dim=-1)


class StencilNetAD(nn.Module):
    """MLP mapping the advection-diffusion stencil to RK parameters.

    The stencil [-nu - mu, 1 + nu + 2*mu, -mu] is in bijection with
    (nu, mu); the forward pass extracts the features
    (nu/24, log nu, log(1 + mu), mu/(1 + mu)), covering both the
    advection-dominated (mu -> 0) and diffusion-dominated (mu >> nu)
    regimes without singularities.
    """

    def __init__(self, n_stages=3, hidden=64):
        super().__init__()
        self.n_stages = n_stages
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_stages),
        ).double()

    def forward(self, stencil):
        mu = -stencil[..., 2:3]
        nu = -stencil[..., 0:1] - mu
        feats = torch.cat([nu / 24.0, torch.log(nu),
                           torch.log1p(mu), mu / (1.0 + mu)], dim=-1)
        out = self.mlp(feats)
        alphas = torch.sigmoid(out[..., : self.n_stages - 1])
        beta = nn.functional.softplus(out[..., -1])
        return alphas, beta


def sample_params(batch):
    log_nu = torch.empty(batch, dtype=torch.float64).uniform_(*TRAIN_LOG_NU)
    log_pe = torch.empty(batch, dtype=torch.float64).uniform_(*TRAIN_LOG_PE)
    nu = torch.exp(log_nu)
    mu = nu / torch.exp(log_pe)
    return nu, mu


def train_ad(n_stages, n_steps=4000, batch=64, lr=1e-3, seed=0, verbose=True):
    torch.manual_seed(seed)
    net = StencilNetAD(n_stages)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    for step in range(n_steps):
        nu, mu = sample_params(batch)
        alphas, beta = net(stencil_from_params(nu, mu))
        loss = torch.log(batched_norm_ad(nu, mu, alphas, beta)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 1000 == 0 or step == n_steps - 1):
            print(f"  [ad s={n_stages}] step {step:4d}  geo-mean eta_bar = "
                  f"{float(loss.detach().exp()):.4f}")
    return net


def net_params_ad(net, nus, mus):
    with torch.no_grad():
        a, b = net(stencil_from_params(np.asarray(nus), np.asarray(mus)))
    return [(a[i].numpy(), float(b[i])) for i in range(len(nus))]


def eval_eta(net, nu, mu, n_theta=513):
    with torch.no_grad():
        a, b = net(stencil_from_params(nu, mu))
    return float(two_grid_factor_ad_1d(
        nu, mu, a[0], b[0], n_theta=n_theta, measure="norm"))

def batched_two_grid_norm_S(nu, mu, alphas, beta, nu1=1, nu2=0, n_theta=64):
    """eta_bar = max_theta ||E_hat(theta)||_2 for a batch of CFL numbers.

    Shapes: nu (B,), alphas (B, s-1), beta (B,).  Returns (B,).
    Broadcasting through `two_grid_block_symbol_1d` requires cfl and beta as
    (B, 1) and the stage coefficients iterable over stages, i.e. (s-1, B, 1).
    """
    thetas = torch.linspace(0.0, math.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_1d_S(
        nu.reshape(-1, 1),
        mu.reshape(-1, 1), 
        alphas.T.unsqueeze(-1),
        beta.reshape(-1, 1),
        thetas,
        nu1=nu1,
        nu2=nu2,
    )
    return torch.linalg.matrix_norm(E, ord=2).amax(dim=-1)


def eval_S(net, nu, mu, n_theta=513):
    with torch.no_grad():
        a, b = net(stencil_from_params(np.asarray(nu), np.asarray(mu)))
    nu = torch.tensor(nu, dtype=torch.float64)
    mu = torch.tensor(mu, dtype=torch.float64)
    eta = batched_two_grid_norm_S(
        nu, mu, a, b, n_theta=n_theta)
    return a.numpy(), b.numpy(), eta.numpy()

def vcycle_matrix_for(net, nu0, mu0, m0=M0):
    n_levels = int(math.log2(m0 // COARSEST)) + 1
    nu_levels = [nu0 / 2**l for l in range(n_levels)]
    mu_levels = [mu0 / 4**l for l in range(n_levels)]
    levels = build_hierarchy(m0, nu0, mu0, periodic=True)
    params = net_params_ad(net, nu_levels, mu_levels)
    return cycle_error_matrix(levels, params), levels, params


def transient_norms(E, kmax=TRANSIENT_KMAX):
    """Exact worst-case error amplification ||E^k||_2, k = 1..kmax."""
    out, Ek = [], np.eye(E.shape[0])
    for _ in range(kmax):
        Ek = E @ Ek
        out.append(float(np.linalg.svd(Ek, compute_uv=False).max()))
    return out

def Jacobi_norms(nu, mu, kmax=TRANSIENT_KMAX):
    """Exact worst-case error amplification ||E^k||_2, k = 1..kmax."""
    out = []
    fac = math.sqrt((1+nu+2*mu)**2+4*nu**2)/3/(1+nu+2*mu)
    Ek = 1.0
    for _ in range(kmax):
        Ek = Ek * fac
        out.append(Ek)
    return out


def oracle_ad(points, n_stages, n_theta=257):
    """Per-(nu, mu) direct optimisation of eta_bar, warm-started along the
    scan order (nu descending, then Pe descending)."""
    a_raw = torch.full((n_stages - 1,), -0.8, dtype=torch.float64,
                       requires_grad=True)
    b_raw = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    out = {}
    for nu, mu in points:
        for lr, steps in ((1e-2, 800), (1e-3, 300)):
            opt = torch.optim.Adam([a_raw, b_raw], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                alphas = torch.sigmoid(a_raw)
                beta = nn.functional.softplus(b_raw)
                loss = two_grid_factor_ad_1d(nu, mu, alphas, beta,
                                             n_theta=n_theta, measure="norm")
                loss.backward()
                opt.step()
        with torch.no_grad():
            alphas = torch.sigmoid(a_raw)
            beta = nn.functional.softplus(b_raw)
            eta = two_grid_factor_ad_1d(nu, mu, alphas, beta,
                                        n_theta=513, measure="norm")
        out[(nu, mu)] = (alphas.tolist(), float(beta), float(eta))
    return out


def solver_check(net, configs, m0=256):
    """Measured two-grid and V-cycle factors vs. LFA predictions on dense
    matrices, for a few (nu0, Pe0) configurations."""
    rows = []
    for nu0, pe0 in configs:
        mu0 = nu0 / pe0
        n_levels = int(math.log2(m0 // COARSEST)) + 1
        nus = [nu0 / 2**l for l in range(n_levels)]
        mus = [mu0 / 4**l for l in range(n_levels)]
        params = net_params_ad(net, nus, mus)

        a = torch.tensor(params[0][0], dtype=torch.float64)
        b = torch.tensor(params[0][1], dtype=torch.float64)
        pred_rho = float(two_grid_factor_ad_1d(nu0, mu0, a, b, n_theta=513,
                                               measure="spectral"))
        pred_eta = float(two_grid_factor_ad_1d(nu0, mu0, a, b, n_theta=513,
                                               measure="norm"))
        thetas = 2.0 * np.pi * torch.arange(m0 // 2, dtype=torch.float64) / m0
        blocks = two_grid_block_symbol_ad_1d(nu0, mu0, a, b, thetas)
        pred_rho_disc = float(
            torch.linalg.eigvals(blocks).abs().amax(dim=-1).max())

        levels = build_hierarchy(m0, nu0, mu0=mu0, periodic=True)
        rho_2g, _ = measured_factors(
            cycle_error_matrix(two_grid_hierarchy(levels), params))
        rho_v, _ = measured_factors(cycle_error_matrix(levels, params))
        rows.append({"nu0": nu0, "pe0": pe0, "rho_tg": pred_rho,
                     "eta_bar": pred_eta, "rho_tg_discrete": pred_rho_disc,
                     "rho_two_grid": rho_2g, "rho_vcycle": rho_v,
                     "exactness_diff": abs(rho_2g - pred_rho_disc)})
        print(f"  (nu0={nu0:5.1f}, Pe0={pe0:7.1f})  rho_TG={pred_rho:.4f}  "
              f"2-grid={rho_2g:.4f} (|diff|={abs(rho_2g-pred_rho_disc):.1e})  "
              f"V-cycle={rho_v:.4f}  eta_bar={pred_eta:.4f}")
    return rows

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

def gmres_experiment(net, n_stages, nu0_values, mu0_values, m0=256, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(3):
        nu0 = nu0_values[i]
        mu0 = mu0_values[i]
        n_levels = int(math.log2(m0 // COARSEST)) + 1
        nu_levels = [nu0 / 2**l for l in range(n_levels)]
        mu_levels = [mu0 / 4**l for l in range(n_levels)]
        levels = build_hierarchy(m0, nu0, mu0, periodic=True)
        A = levels[0]["A"]
        b = rng.standard_normal(m0)
        res = {"nu0": nu0,
               "mu0": mu0,
               "unpreconditioned": gmres_iterations(A, None, b)}
        name = "learned"
        params = net_params_ad(net, nu_levels, mu_levels)
        #for name, params in (("learned", net_params_ad(net, nu_levels, mu_levels))):
        res[name] = gmres_iterations(A, mg_preconditioner(levels, params), b)
        out.append(res)
        print(f"  [s={n_stages}] nu0={nu0:6.2f}  GMRES iters: "
              f"none={res['unpreconditioned']}  learned={res['learned']}  ")
    return out



C_NET, C_ORACLE = "#2a78d6", "#eb6834"
SLICE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#0b0b0b", "#52514e"


def _geom_edges(vals):
    """Cell edges at geometric midpoints, for pcolormesh on log axes."""
    v = np.asarray(vals, dtype=float)
    inner = np.sqrt(v[1:] * v[:-1])
    return np.concatenate([[v[0] ** 2 / inner[0]], inner,
                           [v[-1] ** 2 / inner[-1]]])

C_NORM, C_RHO, C_REF = "#2a78d6", "#eb6834", "#52514e"
INK, MUTED = "#0b0b0b", "#52514e"

def make_figure(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nus, pes = np.array(d["nu_grid"]), np.array(d["pe_grid"])
    eta_net = np.array(d["eta_net"])          # (n_pe, n_nu)
    gap = 100.0 * (np.array(d["eta_net"]) - np.array(d["eta_oracle"])) \
        / np.array(d["eta_oracle"])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), dpi=200)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)

    for ax, Z, cmap, title in ((axes[0, 0], eta_net, "Blues",
                                "sampled two-grid norm (network)"),
                               (axes[0, 1], gap, "Oranges",
                                "gap to local reference [%]")):
        pc = ax.pcolormesh(_geom_edges(nus), _geom_edges(pes), Z, cmap=cmap,
                           shading="flat")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"CFL number $\nu$", fontsize=9, color=INK)
        ax.set_ylabel("mesh Péclet number Pe", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        fig.colorbar(pc, ax=ax).ax.tick_params(labelsize=7, colors=MUTED)

    ax = axes[1, 0]
    ax.set_xscale("log")
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for color, pe in zip(SLICE_COLORS, d["slice_pes"]):
        i = list(pes).index(pe)
        ax.plot(nus, eta_net[i], color=color, lw=2,
                label=f"network, Pe = {pe:g}")
        ax.plot(nus, np.array(d["eta_oracle"])[i], color=color, lw=2,
                ls=(0, (4, 2)), label=f"local ref., Pe = {pe:g}")
    ax.set_xlabel(r"CFL number $\nu$", fontsize=9, color=INK)
    ax.set_ylabel(r"$\eta_T$", fontsize=9, color=INK)
    ax.set_title("slices across the Péclet range", fontsize=10,
                 color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncols=2)

    ax = axes[1, 1]
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    sv = d["solver_check"]
    pred = [r["rho_tg"] for r in sv]
    ax.plot([0, 0.5], [0, 0.5], color=MUTED, lw=1)
    ax.plot(pred, [r["rho_two_grid"] for r in sv], ls="none", marker="o",
            ms=7, mfc="none", mew=1.6, color=C_NET,
            label="measured two-grid")
    ax.plot(pred, [r["rho_vcycle"] for r in sv], ls="none", marker="s",
            ms=6, color=C_ORACLE, label="measured V-cycle")
    lim = 1.1 * max(max(pred), max(r["rho_vcycle"] for r in sv))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"LFA prediction $\rho_{\rm TG}$", fontsize=9, color=INK)
    ax.set_ylabel("measured factor", fontsize=9, color=INK)
    ax.set_title("solver check ($m_0 = 256$)", fontsize=10, color=INK,
                 loc="left")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)

    fig.suptitle("Advection-diffusion: one network over the "
                 r"$(\nu, {\rm Pe})$ plane (3-stage)", fontsize=11,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def make_figure_V(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), dpi=200)
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
        ks = list(range(1, TRANSIENT_KMAX + 1))
        ax.set_yscale("log")
        ax.plot(ks, d["transient"]["240"], color=C_NORM, lw=2,
                marker="o", ms=4, label="RK smoother")
        ax.plot(ks, d["transient"]["240_Jacobi"], color=C_RHO, lw=2,
                marker="s", ms=4, label="Jacobi smoother")
        ax.set_xlabel(r"cycles $k$", fontsize=9, color=INK)
        ax.set_ylabel(r"$\|E_{\rm V}^{\,k}\|_2$", fontsize=9, color=INK)
        ax.set_title(
            f"{n_stages}-stage: worst-case transient, "
            rf"$\nu_0 = {24:g}$, $\mu_0 = {0.1:g}$",
            fontsize=7, color=INK, loc="left")
        if row == 0:
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)

        ax = axes[row, 1]
        ks = list(range(1, TRANSIENT_KMAX + 1))
        ax.set_yscale("log")
        ax.plot(ks, d["transient"]["60"], color=C_NORM, lw=2,
                marker="o", ms=4, label="RK smoother")
        ax.plot(ks, d["transient"]["60_Jacobi"], color=C_RHO, lw=2,
                marker="s", ms=4, label="Jacobi smoother")
        ax.set_xlabel(r"cycles $k$", fontsize=9, color=INK)
        ax.set_ylabel(r"$\|E_{\rm V}^{\,k}\|_2$", fontsize=9, color=INK)
        ax.set_title(
            f"{n_stages}-stage: worst-case transient, "
            rf"$\nu_0 = {3:g}$, $\mu_0 = {0.05:g}$",
            fontsize=7, color=INK, loc="left")
        if row == 0:
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)

        ax = axes[row, 2]
        ks = list(range(1, TRANSIENT_KMAX + 1))
        ax.set_yscale("log")
        ax.plot(ks, d["transient"]["37.5"], color=C_NORM, lw=2,
                marker="o", ms=4, label="RK smoother")
        ax.plot(ks, d["transient"]["37.5_Jacobi"], color=C_RHO, lw=2,
                marker="s", ms=4, label="Jacobi smoother")
        ax.set_xlabel(r"cycles $k$", fontsize=9, color=INK)
        ax.set_ylabel(r"$\|E_{\rm V}^{\,k}\|_2$", fontsize=9, color=INK)
        ax.set_title(
            f"{n_stages}-stage: worst-case transient, "
            rf"$\nu_0 = {0.375:g}$, $\mu_0 = {0.01:g}$",
            fontsize=7, color=INK, loc="left")
        if row == 0:
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK)

    fig.suptitle("Measured multigrid convergence "
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

    nu_grid = list(np.logspace(math.log10(NU_MIN), math.log10(NU_MAX), 7))
    pe_grid = list(np.logspace(math.log10(PE_MIN), math.log10(PE_MAX), 5))
    summary, data = {}, {}
    for n_stages in (2, 3):
        print(f"Training {n_stages}-stage advection-diffusion network ...")
        net = train_ad(n_stages, n_steps=4000)
        torch.save(net.state_dict(),
                   results_dir / f"stencil_net_ad_s{n_stages}.pt")

        # Hierarchy walk: one network serving every level, finest CFL = 24.
        print(f"  hierarchy walk (nu_0 = 24, {n_stages}-stage):")
        nus = [24.0 / 8**l for l in range(3)]
        mus = [0.1, 0.05, 0.01] 
        keys = ["240", "60", "37.5"]
        keys2 = ["240_Jacobi", "60_Jacobi", "37.5_Jacobi"]

        a, b, e_S = eval_S(net, nus, mus)
        for l in range(3):
            print(f"    level {l}: nu = {nus[l]:7.3f} mu = {mus[l]:7.3f} alpha = "
                  f"{np.round(a[l], 3).tolist()}  beta = {b[l]:.3f}  "
                  f"eta_S = {e_S[l]:.4f}")

        trans = {}
        for l in range(3):
            nu = nus[l]
            mu = mus[l]
            key = keys[l]
            key2 = keys2[l]
            E, levels, params = vcycle_matrix_for(net, nu, mu)
            trans[key] = transient_norms(E)
            trans[key2] = Jacobi_norms(nu, mu)
        #d["transient"] = trans


        print(f"GMRES experiment ({n_stages}-stage) ...")
        gmres = gmres_experiment(net, n_stages, [24.0, 3.0, 0.375], [0.1, 0.05, 0.01])


        print(f"Oracle over the (nu, Pe) grid "
              f"({len(nu_grid) * len(pe_grid)} points) ...")
        points = [(nu, nu / pe) for pe in sorted(pe_grid, reverse=True)
                  for nu in sorted(nu_grid, reverse=True)]
        oracle = oracle_ad(points, n_stages)

        eta_net = [[eval_eta(net, nu, nu / pe) for nu in nu_grid]
                   for pe in pe_grid]
        eta_oracle = [[oracle[(nu, nu / pe)][2] for nu in nu_grid]
                      for pe in pe_grid]
        gap = (np.array(eta_net) - np.array(eta_oracle)) / np.array(eta_oracle)
        print(f"  [s={n_stages}] gap to oracle: median {np.median(gap):+.2%}, "
              f"max {gap.max():+.2%}")

        print(f"Solver check ({n_stages}-stage) ...")
        sv = solver_check(net, [(3.0, 10.0), (24.0, 10.0),
                                (24.0, 1000.0), (1.0, 2.0)])

        d = {"nu_grid": nu_grid, "pe_grid": pe_grid,
             "slice_pes": [pe_grid[0], pe_grid[2], pe_grid[4]],
             "eta_net": eta_net, "eta_oracle": eta_oracle,
             "median_gap": float(np.median(gap)),
             "max_gap": float(gap.max()), "solver_check": sv,
             "transient": trans}
        summary[f"s{n_stages}"] = d
        data[f"s{n_stages}"] = d
        if n_stages == 3:
            fig_path = figs_dir / "advdiff_learned.png"
            make_figure(d, fig_path)
            print(f"wrote {fig_path}")
    
    fig_path = figs_dir / "advdiff_V.png"
    make_figure_V(data,fig_path)

    with open(results_dir / "advdiff.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {results_dir/'advdiff.json'}")


if __name__ == "__main__":
    main()
