# neuralfa

Neural architectures for learning multigrid components via local Fourier
analysis.

## Manuscript revision (5 September 2026)

This standalone package was extracted from the
[Neurogrid manuscript project](https://github.com/Tripudium/neurogrid).
The active paper uses the validated 1D implementation. The 2D/D4/GNN/attention
modules are prototypes, not evidence for the paper's numerical claims.

```bash
uv run python -m neuralfa.review_validation
```

This reads the saved weights without retraining, solves the convex polynomial
reference, verifies finite-cycle identities and a recursive V-cycle norm bound,
and writes `results/review_validation.json`, two generated LaTeX tables, and
`figs/convex_reference.png`. The tables are written to `sections/` inside this
checkout. The reference requires SciPy >= 1.16 for
SLSQP multipliers. Its floating-point primal–dual gaps concern the sampled
objective; they are not interval-arithmetic continuum certificates.

A frequency sample approximates the continuous supremum. For an exact
finite-grid two-grid norm, evaluate all discrete harmonic pairs for that grid.
No two-grid quantity alone certifies the full multilevel cycle.

## Quick start

```bash
git clone git@github.com:Tripudium/neuralfa.git
cd neuralfa
uv sync --locked --group dev         # install the locked dependencies and tests
uv run python -m neuralfa.demo      # run all demos
uv run pytest                       # run the test suite
uv run python -m neuralfa.training  # train stencil->RK-parameter nets on the
                                    # two-grid LFA norm loss (~2 min on CPU);
                                    # writes results/ and figs/tg_learned_*.png
uv run python -m neuralfa.solver_validation
                                    # run actual multigrid cycles with the
                                    # trained nets; measured vs LFA-predicted
                                    # factors, inflow-BC robustness, GMRES
                                    # counts -> figs/solver_validation.png
uv run python -m neuralfa.ablation  # spectral-radius vs norm training
                                    # objective: LFA curves, worst-case
                                    # V-cycle transients ||E^k||, GMRES
                                    # -> figs/ablation_rho_vs_norm.png
uv run python -m neuralfa.advdiff   # advection-diffusion: one network over
                                    # the (CFL, Peclet) plane vs per-instance
                                    # oracle + solver check
                                    # -> figs/advdiff_learned.png
```

## Tests

Run with `uv run pytest`.  The suite anchors the LFA machinery to two
independent ground truths:

- `test_birken_lfa.py` — validates against Birken (2012), "Optimizing
  Runge-Kutta smoothers for unsteady flow problems" (ETNA 39;
  reference values are embedded in the tests): evaluating our smoothing
  factor at the published optimal `(alpha, c)` of his Tables 4.1/4.2
  reproduces his objective values, and continuously optimising our
  differentiable loss recovers his grid-search optima.  This pins down the
  `rk_smoother_symbol` recurrence (it matches Birken's stability
  polynomials) and the CFL parameterisation `dt* = c/CFL`.
- `test_two_grid_lfa.py` — assembles the two-grid operator
  `E_TG = S^nu2 (I - P A_c^-1 R A) S^nu1` as a dense matrix and checks that
  spectral radius and spectral norm agree with the maxima over the LFA
  blocks to ~1e-10 (the exact block-diagonalisation proposition of the
  paper), plus mesh-independence, the norm >= spectral-radius gap, and
  differentiability of the norm loss.
- `test_training.py` — batched loss agrees with the unbatched reference;
  a short training run contracts on average.
- `test_solver_validation.py` — the recursive V-cycle truncated to two
  levels reproduces the independent dense two-grid assembly to 1e-12 and
  matches the LFA blocks; iterated V-cycles solve the model problem under
  both periodic and inflow boundary conditions.

## Package structure

```
src/neuralfa/
  __init__.py         # public API re-exports
  multigrid.py        # multigrid primitives (assembly, V-cycle, smoothers)
  lfa.py              # local Fourier analysis loss functions
  training.py         # stencil -> RK-parameter network trained on the
                      #   two-grid LFA norm loss (amortisation experiment)
  equivariant.py      # D4-equivariant stencil network
  gnn.py              # GNN for unstructured grids
  attention.py        # attention-based smoother selection
  demo.py             # demo script exercising all modules
tests/
  test_birken_lfa.py  # validates the smoothing-factor machinery against the
                      #   published optima of Birken (2012), Tables 4.1/4.2
  test_two_grid_lfa.py# validates the two-grid block symbol against direct
                      #   matrix assembly of the two-grid operator
  test_training.py    # smoke tests for the training pipeline
```

## Standalone repository layout

The Python source, tests, locked environment, and small reference checkpoints
are self-contained in this repository. No manuscript checkout or local PDF is
required to run the tests or reproduce the numerical results.

- `results/` contains the reference checkpoints and JSON summaries. Training
  commands overwrite the corresponding outputs; `review_validation` reads the
  existing weights without retraining.
- `figs/` and `sections/` are created locally for generated figures and LaTeX
  tables and are excluded from Git.
- The surrounding manuscript and its research notes remain in
  [Tripudium/neurogrid](https://github.com/Tripudium/neurogrid).

## Modules

### `lfa` — Local Fourier analysis loss functions

Instead of the supervised loss
`||u* - MG(N(x; theta))||^2` (which requires ground-truth solutions), we
provide differentiable LFA-based losses that directly target the convergence
factor.

**Core functions:**

- `stencil_symbol(stencil_3x3, thetas)` — Fourier symbol of a 3x3 stencil at
  given frequencies (E coefficient contributes `exp(+i*theta_x)`, N
  contributes `exp(+i*theta_y)`).
- `rk_smoother_symbol(alphas, beta, A_symbol)` — amplification factor of an
  s-stage explicit RK smoother, built via the stage recursion
  `g_0 = 1; g_j = 1 - alpha_j*z*g_{j-1}; g_s = 1 - z*g_{s-1}` with
  `z = beta*sigma_A`.  Equals Birken's stability polynomial `P_s(-z)`.
- `smoothing_factor_rk(stencil, alphas, beta)` — maximum amplification over
  high-frequency modes (the smoothing factor mu).  Differentiable w.r.t.
  `alphas` and `beta` for use as a training loss.
- `two_grid_convergence_factor(stencil, alphas, beta)` — ideal-coarse-correction
  smoothing proxy in 2D, not full two-grid LFA. A supplied prolongation symbol
  raises `NotImplementedError`; four-harmonic transfer analysis is not implemented.

**1D model problem (Birken 2012)** — the implicit-Euler upwind advection
problem `A = I + cfl*B` with periodic boundary conditions, the setting of the
paper's CFL-parameterisation and exact block-diagonalisation propositions:

- `upwind_advection_symbol_1d(cfl, thetas)` — the symbol
  `sigma(theta) = 1 + cfl*(1 - exp(-i*theta))`.
- `smoothing_factor_rk_1d(cfl, alphas, beta)` — smoothing factor
  `mu = max_{theta in [pi/2, pi]} |g_s(dt* sigma)|` with the pseudo-timestep
  parameterised as `dt* = beta/cfl`, so `beta` is Birken's parameter `c`.
  `mu**2` is the "Opt-value" objective of his eq. (4.2) and Tables 4.1-4.3.
- `two_grid_block_symbol_1d(cfl, alphas, beta, thetas, nu1, nu2)` — the 2x2
  harmonic-pair block symbol `E_hat(theta)` of the two-grid operator with
  aggregation transfers.  For a periodic grid with m cells the DFT
  block-diagonalises the two-grid operator *exactly* into these blocks at
  `theta_k = 2*pi*k/m, k < m/2` (verified against dense matrices in the
  tests).  Broadcasts over a batch when `cfl`/`beta` are shaped `(B, 1)` and
  the stage coefficients `(s-1, B, 1)`.
- `two_grid_factor_1d(cfl, alphas, beta, measure=...)` — the reduction over
  frequencies: `measure="norm"` gives a sampled maximum of `||E_hat||_2`;
  `measure="spectral"` gives a sampled maximum of `rho(E_hat)`. The defaults
  approximate continuous-frequency objectives and are not certified upper
  bounds over unsampled frequencies.
  Always `norm >= spectral`; the gap is genuine for these non-normal blocks.

The smoothing factor is defined as

    mu = max_{theta in Theta_high} |sigma_RK(theta)|

where `sigma_RK` is the amplification polynomial built stage by stage:

    g_0 = 1
    g_j = 1 - alpha_j * beta * sigma_A * g_{j-1}
    g_s = 1 - beta * sigma_A * g_{s-1}

and `sigma_A(theta)` is the Fourier symbol of the discretisation stencil.

### `training` — Amortised smoother selection on the two-grid loss

Implements the paper's amortisation experiment (Section "Amortised smoother
selection with the two-grid LFA loss" in the numerics): a single network
learns the map from the model-problem stencil to RK smoother parameters by
minimising the log of a sampled two-grid norm.

```bash
uv run python -m neuralfa.training
```

trains a 2-stage and a 3-stage network (~1 minute each on CPU, deterministic
seed) and writes

- `results/stencil_net_s{2,3}.pt` — trained weights,
- `results/tg_training_summary.json` — learned/local-reference/Birken data;
  the historical `oracle` field is a local optimisation,
- `figs/tg_learned_{2,3}stage.png` — historical local-reference figures.
  The revised paper uses the convex-reference figure instead.

**Model.** `StencilNet`: MLP with two hidden layers of 64 tanh units on the
features `(nu/24, log nu)` extracted from the stencil `[-nu, 1+nu, 0]`;
sigmoid head for `alpha in (0,1)^(s-1)`, softplus head for `beta > 0`.

**Loss.** `E_nu[ log eta_bar ]` with `nu` log-uniform on `[1/48, 48]`
(Adam, 3000 steps, batch 64, cosine-annealed lr 1e-3).  Two deliberate
choices, both learned the hard way:

1. *Train on `log eta_bar`, not `eta_bar`.*  The arithmetic mean gives
   negligible gradient weight to small CFL numbers, where the optimal factor
   is ~1e-3; the log equalises relative accuracy across three decades of CFL
   (it minimises the geometric-mean contraction factor).
2. *Train one octave beyond the evaluation range on each side.*  Otherwise
   edge-of-support extrapolation error dominates the gap at `nu = 1/24`.

**Results.** Against the stronger convex polynomial reference, median excess
sampled norms are 0.59% (2-stage) and 7.54% (3-stage). Maximum relative gaps
are 48.4% and 800.6% at CFL 1/24, despite small absolute norms. The earlier
0.6%/6.6% figures used a less accurate local reference. One network can be
queried at all level CFL values inside its parameter domain; this does not
prove uniform approximation or multilevel optimality. See the generated
review JSON and the revised manuscript for the precise numerical checks.

### `multigrid` — Core multigrid utilities

Shared infrastructure used by all architectures:

- `assemble_2d_advection_diffusion(n, eps, b, dt)` — assembles the implicit
  Euler discretisation of `-eps * Laplacian(u) + b . grad(u) = f` on an n x n
  periodic grid.  Returns the sparse matrix A and the constant-coefficient 3x3
  stencil.
- `bilinear_prolongation_2d(nc)` — standard bilinear interpolation from a
  coarse grid to a fine grid.
- `restriction_2d(P)` — full-weighting restriction (scaled transpose of P).
- `weighted_jacobi(A, b, x, omega, nu)` — nu steps of weighted Jacobi
  smoothing.
- `vcycle(A_levels, P_levels, R_levels, b, x, ...)` — recursive V-cycle with
  optional custom smoother callback.

## PDE test cases

The following test problems are recommended for evaluating the learned
multigrid components, organised by difficulty.  See Section 6.4 of the paper
for the full mathematical formulations and rationale.

### Tier 1 — Directly accessible

These build on the existing `assemble_2d_advection_diffusion` and require
minimal code changes.

#### Advection-diffusion with varying Peclet number

```
-eps * Laplacian(u) + b . grad(u) = f    on [0,1]^2
```

The central benchmark.  Sweep the parameter space:

| Parameter | Values |
|-----------|--------|
| eps (diffusion) | 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6 |
| alpha (advection angle) | 0, pi/6, pi/4, pi/3, pi/2 |
| N (grid size) | 32, 64, 128, 256 |

As eps decreases, the problem becomes advection-dominated and standard
multigrid with Jacobi/Gauss-Seidel smoothers degrades sharply.  This is where
learned RK smoothers should show the largest gains.  The advection angle sweep
tests generalisation — the D4-equivariant network should handle angle variation
much better than a plain dense network.

**Assembly:** Use `assemble_2d_advection_diffusion(n, eps, b=(cos(alpha),
sin(alpha)), dt)` directly.

**Baseline comparison:** Birken's RK smoother (grid-search optimised) and
Huang et al.'s CNN smoother.


## Dependencies

- Python >= 3.13
- PyTorch >= 2.6
- NumPy >= 2.4
- SciPy >= 1.16
- Matplotlib >= 3.10

## References

- Birken (2012) — RK smoother parameter optimisation via grid search
- Greenfeld et al. (2019) — learned prolongation for variable-coefficient
  diffusion
- Luz et al. (2020) — GNN-based algebraic multigrid
- Huang et al. (2023) — CNN smoother learning using supervised multigrid error
- Azulay & Treister (2024) — multigrid-augmented deep learning for Helmholtz
