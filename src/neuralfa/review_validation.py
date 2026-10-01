"""Reproduce the September 2026 mathematical and numerical manuscript audit.

Reads existing checkpoints without retraining or overwriting them. Writes
review_validation.json, a convex-reference comparison figure, and LaTeX
tables included by the manuscript. Run: uv run python -m neuralfa.review_validation
"""

import json
from pathlib import Path
import numpy as np
import torch

from .lfa import two_grid_block_symbol_1d, two_grid_factor_1d
from .polynomial import polynomial_reference
from .solver_validation import (build_hierarchy, cycle_error_matrix, load_net,
                                net_params, rk_smooth)
from .training import evaluate


def multilevel_bound(levels, params):
    """q_l <= eta_l + kappa_l q_(l+1), with exact finite-grid norms.

    Returns the recursively computed sufficient bound and its ingredients.
    Uses dense matrices as an independent check of the paper's identity.
    One pre-smoothing step and no post-smoothing match solver_validation.
    """
    q, rows = 0.0, []
    for l in range(len(levels) - 2, -1, -1):
        A, R, P = (levels[l][k] for k in ("A", "R", "P"))
        Ac = levels[l + 1]["A"]
        I = np.eye(len(A))
        a, b = params[l]
        S = rk_smooth(A, I, np.zeros_like(I), a, b / levels[l]["scale"])
        T = np.linalg.solve(Ac, R @ A @ S)
        E2 = S - P @ T
        eta = float(np.linalg.norm(E2, 2))
        kappa = float(np.linalg.norm(P, 2) * np.linalg.norm(T, 2))
        q = eta + kappa * q
        rows.append({"level": l, "eta": eta, "kappa": kappa, "bound": q})
    return q, rows[::-1]


def run(results, root):
    old = json.loads((results / "tg_training_summary.json").read_text())
    summary = {}
    for s in (2, 3):
        net = load_net(s, results)
        nus = old[f"s{s}"]["nu_grid"]
        a, b, eta_net = evaluate(net, nus)
        refs = [polynomial_reference(nu, s) for nu in nus]
        if not all(r["rk_feasible"] for r in refs):
            raise RuntimeError("A polynomial reference is outside the RK family")
        eta_ref = np.array([r["eta"] for r in refs])
        # Independently evaluate the recovered RK polynomial on a much
        # denser grid. This measures frequency sampling error, not a proof
        # of a continuum-frequency upper bound.
        dense_eta = [float(two_grid_factor_1d(
            nu, torch.tensor(r["alphas"], dtype=torch.float64), r["beta"],
            n_theta=8193)) for nu, r in zip(nus, refs)]
        dense_net = [float(two_grid_factor_1d(
            nu, torch.tensor(ai), bi, n_theta=8193))
            for nu, ai, bi in zip(nus, a, b)]
        bounds = []
        for nu in (1., 3., 6., 12., 24.):
            levels = build_hierarchy(256, nu)
            params = net_params(net, [x["nu"] for x in levels])
            E = cycle_error_matrix(levels, params)
            bound, details = multilevel_bound(levels, params)
            bounds.append({"nu": nu, "norm_v": float(np.linalg.norm(E, 2)),
                           "bound": bound, "levels": details})
        # Finite-cycle identity checked against full matrix powers.
        levels = build_hierarchy(64, 24.)[:2]
        params = net_params(net, [x["nu"] for x in levels])
        E = cycle_error_matrix(levels, params)
        th = torch.arange(32, dtype=torch.float64) * (2*np.pi/64)
        blocks = two_grid_block_symbol_1d(24., torch.tensor(params[0][0]),
                                          params[0][1], th).numpy()
        tr = np.trace(blocks, axis1=-2, axis2=-1)
        block_norm = np.linalg.norm(blocks, axis=(-2,-1))
        errors = [abs(np.linalg.norm(np.linalg.matrix_power(E, k), 2)
                      - np.max(np.abs(tr)**(k-1) * block_norm))
                  for k in (1, 2, 3, 6)]
        d = {"references": refs, "eta_network": eta_net.tolist(),
             "eta_dense_reference": dense_eta, "eta_dense_network": dense_net,
             "median_relative_gap": float(np.median(eta_net/eta_ref-1)),
             "max_relative_gap": float(np.max(eta_net/eta_ref-1)),
             "max_squared_duality_gap": max(abs(r["squared_duality_gap"]) for r in refs),
             "max_frequency_refinement_change": float(np.max(np.abs(
                 np.array(dense_eta)-eta_ref))),
             "max_network_frequency_refinement_change": float(np.max(np.abs(
                 np.array(dense_net)-eta_net))),
             "rank_one_power_max_error": max(errors), "multilevel": bounds}
        summary[f"s{s}"] = d
        print(f"s={s}: median gap {d['median_relative_gap']:.3%}, "
              f"max gap {d['max_relative_gap']:.3%}; "
              f"duality gap {d['max_squared_duality_gap']:.2e}", flush=True)
        print("  multilevel (nu, measured norm, bound):", [
            (x["nu"], round(x["norm_v"],4), round(x["bound"],4)) for x in bounds], flush=True)
    (results / "review_validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_tables(summary, root / "sections", old)
    plot(summary, root / "figs" / "convex_reference.png")
    return summary


def write_tables(summary, sections, training_summary):
    sections.mkdir(parents=True, exist_ok=True)
    rows = [r"\begin{tabular}{crrrr}", r"\hline",
            r"Stages & $\nu$ & Network & Local search & Convex reference \\", r"\hline"]
    old = training_summary
    def tex_number(value):
        value = f"{value:.6g}"
        if "e" in value:
            mantissa, exponent = value.split("e")
            return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"
        return "$" + value + "$"

    for s in (2, 3):
        d = summary[f"s{s}"]
        for i in (0, 12, 24):
            r = d["references"][i]
            local = old[f"s{s}"]["oracle"][str(r["nu"])][2]
            nu_label = r"$1/24$" if i == 0 else tex_number(r["nu"])
            rows.append(f"{s} & {nu_label} & {tex_number(d['eta_network'][i])} & "
                        f"{tex_number(local)} & {tex_number(r['eta'])}" + r" \\")
    rows += [r"\hline", r"\end{tabular}"]
    (sections / "table_convex.tex").write_text("\n".join(rows)+"\n")
    rows = [r"\begin{tabular}{crrrr}", r"\hline",
            r" & \multicolumn{2}{c}{Two stages} & \multicolumn{2}{c}{Three stages} \\",
            r"$\nu_0$ & $\|E_{\rm V}\|_2$ & $q_0$ & $\|E_{\rm V}\|_2$ & $q_0$ \\", r"\hline"]
    for x,y in zip(summary["s2"]["multilevel"], summary["s3"]["multilevel"]):
        rows.append(f"{x['nu']:g} & {x['norm_v']:.3f} & {x['bound']:.3f} & "
                    f"{y['norm_v']:.3f} & {y['bound']:.3f}" + r" \\")
    rows += [r"\hline", r"\end{tabular}"]
    (sections / "table_multilevel.tex").write_text("\n".join(rows)+"\n")


def plot(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=200)
    for ax,s in zip(axes,(2,3)):
        d = summary[f"s{s}"]
        nu = [r["nu"] for r in d["references"]]
        ax.loglog(nu,d["eta_network"], color="#2266aa", label="Network")
        ax.loglog(nu,[r["eta"] for r in d["references"]], "--",
                  color="#bb5522", label="Convex reference")
        ax.set(xlabel=r"CFL number $\nu$", ylabel="Sampled two-grid norm",
               title=f"{s}-stage smoother")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    run(root / "results", root)


if __name__ == "__main__":
    main()
