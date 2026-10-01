"""Local Fourier analysis (LFA) tools for multigrid convergence estimation.

Provides differentiable (PyTorch) routines for computing:
- The smoothing factor of an s-stage RK smoother given its Fourier symbol.
- Exact 1D two-grid symbols and a separate idealised 2D smoothing proxy.

These serve as unsupervised loss functions for training smoother and
prolongation networks without requiring ground-truth solutions.
"""

import torch
import numpy as np


def _fourier_grid(n_theta=64):
    """Return a 2D grid of Fourier frequencies theta in (-pi, pi]^2.

    Returns (n_theta^2, 2) tensor.
    """
    t = torch.linspace(-np.pi + 2 * np.pi / n_theta, np.pi, n_theta,
                       dtype=torch.float64)
    tx, ty = torch.meshgrid(t, t, indexing="ij")
    return torch.stack([tx.reshape(-1), ty.reshape(-1)], dim=-1)


def high_frequency_mask(thetas):
    """Boolean mask selecting high-frequency modes (at least one component
    in (pi/2, pi] or [-pi, -pi/2))."""
    abs_t = thetas.abs()
    return (abs_t[:, 0] > np.pi / 2) | (abs_t[:, 1] > np.pi / 2)


def stencil_symbol(stencil_3x3, thetas):
    """Compute the Fourier symbol of a 3x3 stencil at frequencies thetas.

    Parameters
    ----------
    stencil_3x3 : torch.Tensor of shape (3, 3)
        Stencil coefficients in row-major order:
        [[NW, N, NE], [W, C, E], [SW, S, SE]].
    thetas : torch.Tensor of shape (M, 2)

    Returns
    -------
    torch.Tensor of shape (M,), complex-valued Fourier symbol.
    """
    tx, ty = thetas[:, 0], thetas[:, 1]
    # Offsets for a 3x3 stencil centred at (0,0): di is the y-offset (rows,
    # north positive), dj is the x-offset (columns, east positive), so the
    # E coefficient contributes exp(+i*theta_x) and N contributes
    # exp(+i*theta_y).
    symbol = torch.zeros(thetas.shape[0], dtype=torch.complex128)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            coeff = stencil_3x3[1 - di, 1 + dj]  # map stencil index to offset
            phase = torch.exp(1j * (dj * tx + di * ty))
            symbol = symbol + coeff * phase
    return symbol


def rk_smoother_symbol(alphas, beta, A_symbol):
    """Fourier symbol of an s-stage explicit RK smoother.

    The RK iteration on the pseudo-time ODE u_{t*} = b - A u is:
        u_0 = u^n
        u_j = u^n + alpha_j * dt* * (b - A u_{j-1}),   j = 1, ..., s-1
        u_s = u^n + dt* * (b - A u_{s-1})

    where dt* = beta * dx / (|b_adv| * dt).

    Here beta is the actual pseudo-time step multiplying A_symbol.
    The recurrence g_j = 1 - alpha_j*z*g_{j-1}, with the final alpha=1,
    gives a polynomial in z=beta*A_symbol; it is not a product of Euler
    factors. The 1D wrapper functions apply the CFL normalisation.

    Parameters
    ----------
    alphas : torch.Tensor of shape (s-1,)
        RK stage coefficients.
    beta : torch.Tensor, scalar
        Actual pseudo-time step (normalisation is the caller's responsibility).
    A_symbol : torch.Tensor of shape (M,), complex
        Fourier symbol of A at each frequency.

    Returns
    -------
    torch.Tensor of shape (M,), complex amplification factor.
    """
    z = beta * A_symbol  # (M,)
    # Build the amplification polynomial stage by stage
    # Stage 1: g_1 = 1 - alpha_1 * z
    # Stage j: g_j = 1 - alpha_j * z * g_{j-1}  (simplified Euler-like form)
    # Final:   g_s = 1 - z * g_{s-1}
    g = torch.ones_like(z)
    for alpha_j in alphas:
        g = 1.0 - alpha_j * z * g
    # Final stage (alpha = 1)
    g = 1.0 - z * g
    return g


def smoothing_factor_rk(stencil_3x3, alphas, beta, n_theta=64):
    """Compute the LFA smoothing factor for an RK smoother.

    This is the maximum absolute value of the RK amplification factor over
    high-frequency Fourier modes.  It is differentiable w.r.t. alphas and beta.

    Parameters
    ----------
    stencil_3x3 : torch.Tensor of shape (3, 3)
    alphas : torch.Tensor of shape (s-1,)
    beta : torch.Tensor, scalar
    n_theta : int, resolution of the Fourier grid

    Returns
    -------
    mu : torch.Tensor, nonnegative scalar smoothing factor (may exceed one).
    """
    thetas = _fourier_grid(n_theta)
    A_sym = stencil_symbol(stencil_3x3, thetas)
    sigma = rk_smoother_symbol(alphas, beta, A_sym)
    mask = high_frequency_mask(thetas)
    high_freq_amp = sigma[mask].abs()
    mu = high_freq_amp.max()
    return mu


def upwind_advection_symbol_1d(cfl, thetas):
    """Fourier symbol of the 1D implicit-Euler upwind advection matrix.

    This is the model problem of Birken, "Optimizing Runge-Kutta smoothers
    for unsteady flow problems", ETNA 39 (2012): u_t + b u_x = 0 with b > 0,
    first-order upwind in space and implicit Euler in time, giving
    A = I + cfl * B with (B u)_i = u_i - u_{i-1} (periodic), eq. (2.4).
    The eigenvalues of A are

        sigma(theta) = 1 + cfl * (1 - exp(-i*theta)).

    Parameters
    ----------
    cfl : float
        CFL number b * dt / dx of the level.
    thetas : torch.Tensor of shape (M,)
        Frequencies in (-pi, pi].

    Returns
    -------
    torch.Tensor of shape (M,), complex.
    """
    return 1.0 + cfl * (1.0 - torch.exp(-1j * thetas))


def smoothing_factor_rk_1d(cfl, alphas, beta, n_theta=257):
    """LFA smoothing factor of the s-stage RK smoother for 1D upwind advection.

    Computes mu = max_{theta in [pi/2, pi]} |g_s(dt* sigma_A(theta))| for the
    Birken (2012) model problem, where the pseudo-timestep is parameterised as
    dt* = beta * dx / (b*dt) = beta / cfl, so `beta` is exactly Birken's
    parameter `c` (and the beta of the paper draft).  By conjugate symmetry
    |g_s| is even in theta, so maximising over [pi/2, pi] covers the full
    high-frequency range pi/2 <= |theta| <= pi.

    mu**2 corresponds to the "Opt-value" objective of Birken's eq. (4.2)
    and Tables 4.1-4.3.  Differentiable w.r.t. alphas and beta.

    Parameters
    ----------
    cfl : float
        CFL number of the level.  Note that standard coarsening halves it.
    alphas : torch.Tensor of shape (s-1,)
        RK stage coefficients (alpha_1, ..., alpha_{s-1}).
    beta : torch.Tensor, scalar
        Pseudo-timestep scaling (Birken's c).
    n_theta : int
        Resolution of the frequency grid on [pi/2, pi].

    Returns
    -------
    mu : torch.Tensor, scalar smoothing factor.
    """
    thetas = torch.linspace(np.pi / 2, np.pi, n_theta, dtype=torch.float64)
    sigma = upwind_advection_symbol_1d(cfl, thetas)
    g = rk_smoother_symbol(alphas, beta / cfl, sigma)
    return g.abs().max()


def two_grid_block_symbol_1d(cfl, alphas, beta, thetas, nu1=1, nu2=0):
    """Harmonic-pair block symbol of the two-grid operator, Birken model problem.

    For the 1D implicit-Euler upwind advection problem A = I + cfl*B with
    aggregation transfers R (pairwise averaging) and P = 2*R^T (piecewise
    constant), the two-grid error propagation operator

        E_TG = S^nu2 (I - P A_c^{-1} R A) S^nu1

    leaves the harmonic-pair spaces span{phi^theta, phi^(theta+pi)} invariant
    and acts on them by the 2x2 block

        E_hat(theta) = S_hat^nu2 (I_2 - p_hat sigma_c^{-1} r_hat A_hat) S_hat^nu1

    with transfer symbols r_hat = ((1+e^{i th})/2, (1-e^{i th})/2),
    p_hat = conj(r_hat)^T, coarse symbol
    sigma_c(2 th) = 1 + (cfl/2)(1 - e^{-2i th}) (Galerkin and re-discretised
    coarse operators coincide for this transfer pair), and S_hat the diagonal
    of RK amplification factors.  For a periodic grid with m cells the DFT
    block-diagonalises E_TG exactly into these blocks at
    theta_k = 2 pi k / m, k = 0, ..., m/2 - 1.

    Parameters
    ----------
    cfl : float
        CFL number of the fine level.
    alphas : torch.Tensor of shape (s-1,)
    beta : torch.Tensor, scalar
        Pseudo-timestep scaling (Birken's c): dt* = beta / cfl.
    thetas : torch.Tensor of shape (M,)
        Pair representatives; {theta, theta + pi} enumerates each pair once.
    nu1, nu2 : int
        Pre-/post-smoothing steps (Birken's V-cycle uses nu1=1, nu2=0).

    Returns
    -------
    torch.Tensor of shape (M, 2, 2), complex.  Differentiable w.r.t.
    alphas and beta.
    """
    return two_grid_block_symbol_ad_1d(cfl, 0.0, alphas, beta, thetas,
                                       nu1=nu1, nu2=nu2)

def two_grid_block_symbol_1d_S(nu, mu, alphas, beta, thetas, nu1=1, nu2=0):
    """Harmonic-pair block symbol of the two-grid operator, Birken model problem.

    For the 1D implicit-Euler upwind advection problem A = I + cfl*B with
    aggregation transfers R (pairwise averaging) and P = 2*R^T (piecewise
    constant), the two-grid error propagation operator

        E_TG = S^nu1

    leaves the harmonic-pair spaces span{phi^theta, phi^(theta+pi)} invariant
    and acts on them by the 2x2 block

        E_hat(theta) = S_hat^nu1

    with transfer symbols r_hat = ((1+e^{i th})/2, (1-e^{i th})/2),
    p_hat = conj(r_hat)^T, coarse symbol
    sigma_c(2 th) = 1 + (cfl/2)(1 - e^{-2i th}) (Galerkin and re-discretised
    coarse operators coincide for this transfer pair), and S_hat the diagonal
    of RK amplification factors.  For a periodic grid with m cells the DFT
    block-diagonalises E_TG exactly into these blocks at
    theta_k = 2 pi k / m, k = 0, ..., m/2 - 1.

    Parameters
    ----------
    cfl : float
        CFL number of the fine level.
    alphas : torch.Tensor of shape (s-1,)
    beta : torch.Tensor, scalar
        Pseudo-timestep scaling (Birken's c): dt* = beta / cfl.
    thetas : torch.Tensor of shape (M,)
        Pair representatives; {theta, theta + pi} enumerates each pair once.
    nu1, nu2 : int
        Pre-/post-smoothing steps (Birken's V-cycle uses nu1=1, nu2=0).

    Returns
    -------
    torch.Tensor of shape (M, 2, 2), complex.  Differentiable w.r.t.
    alphas and beta.
    """
    t0 = thetas
    t1 = thetas + np.pi
    scale = nu + 4.0 * mu
    sig0 = advection_diffusion_symbol_1d(nu, mu, t0)
    sig1 = advection_diffusion_symbol_1d(nu, mu, t1)
    g0 = rk_smoother_symbol(alphas, beta / scale, sig0)
    g1 = rk_smoother_symbol(alphas, beta / scale, sig1)

    S1 = torch.diag_embed(torch.stack([g0**nu1, g1**nu1], dim=-1))
    return S1


def advection_diffusion_symbol_1d(nu, mu, thetas):
    """Fourier symbol of A = I + nu*B + mu*C, 1D advection-diffusion.

    Implicit-Euler discretisation of u_t - eps*u_xx + b*u_x = f with
    first-order upwind advection ((Bu)_i = u_i - u_{i-1}) and central
    diffusion ((Cu)_i = 2u_i - u_{i-1} - u_{i+1}), periodic BC:

        sigma(theta) = 1 + nu*(1 - exp(-i*theta)) + 2*mu*(1 - cos(theta)),

    with nu = b*dt/dx (CFL number) and mu = eps*dt/dx^2 (diffusion number).
    The mesh Peclet number is Pe = b*dx/eps = nu/mu.  mu = 0 recovers the
    pure-advection model problem.
    """
    return (1.0 + nu * (1.0 - torch.exp(-1j * thetas))
            + 2.0 * mu * (1.0 - torch.cos(thetas)))


def two_grid_block_symbol_ad_1d(nu, mu, alphas, beta, thetas, nu1=1, nu2=0):
    """Harmonic-pair block symbol for the 1D advection-diffusion problem.

    Same construction as `two_grid_block_symbol_1d` (which is the mu = 0
    special case), with the pseudo-timestep normalised as

        dt* = beta / (nu + 4*mu),

    which reduces to Birken's dt* = beta/nu at mu = 0 and keeps
    dt* * sigma(theta) of order beta at the highest frequency in both the
    advection- and diffusion-dominated limits (sigma(pi) = 1 + 2nu + 4mu).
    The coarse operator is the re-discretisation with (nu/2, mu/4); note
    that for mu > 0 this no longer coincides with the Galerkin operator,
    which would carry mu/2 (piecewise-constant aggregation halves rather
    than quarters the diffusion term).

    Broadcasts like `two_grid_block_symbol_1d`: nu, mu, beta as (B, 1),
    stage coefficients iterable over stages, i.e. (s-1, B, 1).
    """
    t0 = thetas
    t1 = thetas + np.pi
    scale = nu + 4.0 * mu
    sig0 = advection_diffusion_symbol_1d(nu, mu, t0)
    sig1 = advection_diffusion_symbol_1d(nu, mu, t1)
    g0 = rk_smoother_symbol(alphas, beta / scale, sig0)
    g1 = rk_smoother_symbol(alphas, beta / scale, sig1)

    r0 = 0.5 * (1.0 + torch.exp(1j * t0))
    r1 = 0.5 * (1.0 - torch.exp(1j * t0))
    p0, p1 = r0.conj(), r1.conj()
    inv_sig_c = 1.0 / advection_diffusion_symbol_1d(nu / 2.0, mu / 4.0,
                                                    2.0 * t0)

    # K = p_hat sigma_c^{-1} r_hat A_hat, entries K[a, b] = p_a r_b sig_b / sig_c
    K = torch.stack(
        [
            torch.stack([p0 * inv_sig_c * r0 * sig0,
                         p0 * inv_sig_c * r1 * sig1], dim=-1),
            torch.stack([p1 * inv_sig_c * r0 * sig0,
                         p1 * inv_sig_c * r1 * sig1], dim=-1),
        ],
        dim=-2,
    )
    cgc = torch.eye(2, dtype=K.dtype) - K
    S1 = torch.diag_embed(torch.stack([g0**nu1, g1**nu1], dim=-1))
    S2 = torch.diag_embed(torch.stack([g0**nu2, g1**nu2], dim=-1))
    return S2 @ cgc @ S1


def two_grid_factor_1d(cfl, alphas, beta, nu1=1, nu2=0, n_theta=257,
                       measure="norm"):
    """Two-grid LFA convergence measure for the Birken model problem.

    measure="norm" returns a SAMPLED maximum of ||E_hat(theta)||_2.
    This approximates the continuum supremum; sampling alone is not a
    continuum certificate. For an exact finite-grid norm, evaluate the
    blocks at all that grid's discrete Fourier pair representatives.
    measure="spectral" returns max_theta rho(E_hat(theta)), the asymptotic
    two-grid convergence factor.  Always norm >= spectral.

    By conjugate symmetry, pair representatives theta in [0, pi/2] cover all
    harmonic pairs.  Differentiable w.r.t. alphas and beta (the spectral
    measure only where eigenvalues are simple).
    """
    thetas = torch.linspace(0.0, np.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_1d(cfl, alphas, beta, thetas, nu1=nu1, nu2=nu2)
    if measure == "norm":
        vals = torch.linalg.matrix_norm(E, ord=2)
    elif measure == "spectral":
        vals = torch.linalg.eigvals(E).abs().amax(dim=-1)
    else:
        raise ValueError(f"unknown measure: {measure!r}")
    return vals.max()


def two_grid_factor_ad_1d(nu, mu, alphas, beta, nu1=1, nu2=0, n_theta=257,
                          measure="norm"):
    """Two-grid LFA convergence measure for 1D advection-diffusion.

    Generalises `two_grid_factor_1d` (its mu = 0 special case); see there
    for the meaning of `measure`.
    """
    thetas = torch.linspace(0.0, np.pi / 2, n_theta, dtype=torch.float64)
    E = two_grid_block_symbol_ad_1d(nu, mu, alphas, beta, thetas,
                                    nu1=nu1, nu2=nu2)
    if measure == "norm":
        vals = torch.linalg.matrix_norm(E, ord=2)
    elif measure == "spectral":
        vals = torch.linalg.eigvals(E).abs().amax(dim=-1)
    else:
        raise ValueError(f"unknown measure: {measure!r}")
    return vals.max()


def two_grid_convergence_factor(stencil_3x3, alphas, beta,
                                P_symbol_fn=None, nu1=2, nu2=2,
                                n_theta=64):
    """Ideal-coarse-correction smoothing proxy, not full 2D two-grid LFA.

    Computes rho = max_theta |sigma_S^{nu2} * sigma_CGC * sigma_S^{nu1}|
    where sigma_S is the smoother symbol and sigma_CGC is the coarse-grid
    correction symbol.

    For simplicity, if P_symbol_fn is None, we use ideal coarse-grid
    correction (CGC kills all low-frequency error perfectly), so the
    convergence factor reduces to the smoothing factor raised to (nu1 + nu2).

    Parameters
    ----------
    stencil_3x3 : torch.Tensor of shape (3, 3)
    alphas : torch.Tensor of shape (s-1,)
    beta : torch.Tensor, scalar
    P_symbol_fn : optional callable(thetas) -> complex tensor of shape (M,)
        Fourier symbol of the prolongation operator.
    nu1, nu2 : int, pre/post smoothing steps.
    n_theta : int

    Returns
    -------
    rho : torch.Tensor, scalar convergence factor.
    """
    if P_symbol_fn is not None:
        raise NotImplementedError(
            "2D two-grid LFA requires a 4x4 harmonic block; learned transfers "
            "are not implemented by this ideal-coarse-correction proxy."
        )
    thetas = _fourier_grid(n_theta)
    A_sym = stencil_symbol(stencil_3x3, thetas)
    sigma_S = rk_smoother_symbol(alphas, beta, A_sym)

    if P_symbol_fn is None:
        # Ideal CGC: kills low-frequency error, so two-grid factor is
        # determined entirely by high-frequency smoothing
        mask = high_frequency_mask(thetas)
        amp = (sigma_S[mask].abs() ** (nu1 + nu2))
        return amp.max()
