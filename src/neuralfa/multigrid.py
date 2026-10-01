"""Core multigrid utilities for 2D structured grids.

Provides assembly of the 2D advection-diffusion operator, standard
prolongation/restriction, weighted Jacobi smoothing, and a V-cycle solver.
"""

import numpy as np
import scipy.sparse as sp


def assemble_2d_advection_diffusion(n, eps=1.0, b=(1.0, 0.0), dt=1.0):
    """Assemble the 2D advection-diffusion operator on an n x n interior grid.

    Discretises  u - dt*(eps*Laplacian(u) - b . grad(u)) = rhs
    using backward Euler in time, 5-point Laplacian, and first-order upwind
    advection. Periodic boundary conditions.

    Returns the sparse matrix A of size (n*n, n*n) and the 3x3 stencil at an
    interior point.
    """
    if n < 1:
        raise ValueError("n must be positive")
    h = 1.0 / n
    bx, by = b

    # Diffusion contributions (5-point stencil)
    c = eps * dt / h**2
    # Upwind advection contributions
    ax_p = max(bx, 0.0) * dt / h
    ax_m = max(-bx, 0.0) * dt / h
    ay_p = max(by, 0.0) * dt / h
    ay_m = max(-by, 0.0) * dt / h

    centre = 1.0 + 4.0 * c + ax_p + ax_m + ay_p + ay_m
    west = -c - ax_p
    east = -c - ax_m
    south = -c - ay_p
    north = -c - ay_m

    # Assemble by grid coordinates: flattened +/-1 diagonals connect the
    # wrong rows at horizontal boundaries. COO also sums coincident
    # neighbours correctly for n=1 or n=2.
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            row = i * n + j
            for di, dj, value in ((0, 0, centre), (0, -1, west),
                                   (0, 1, east), (-1, 0, south), (1, 0, north)):
                rows.append(row)
                cols.append(((i + di) % n) * n + (j + dj) % n)
                vals.append(value)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n*n, n*n)).tocsr()

    # 3x3 stencil (row-major: NW N NE / W C E / SW S SE)
    stencil = np.array([
        [0.0, north, 0.0],
        [west, centre, east],
        [0.0, south, 0.0],
    ])

    return A, stencil


def bilinear_prolongation_2d(nc):
    """Standard bilinear prolongation from nc x nc to (2*nc) x (2*nc) grid.

    Returns the prolongation matrix P of size (nf*nf, nc*nc) where nf = 2*nc.
    """
    nf = 2 * nc
    Nc = nc * nc
    Nf = nf * nf
    rows, cols, vals = [], [], []

    for ic in range(nc):
        for jc in range(nc):
            c_idx = ic * nc + jc
            # The four fine-grid points influenced by coarse point (ic, jc)
            # using periodic wrapping
            for di, dj, w in [
                (0, 0, 1.0),
                (0, 1, 0.5),
                (1, 0, 0.5),
                (1, 1, 0.25),
                (0, -1, 0.5),
                (-1, 0, 0.5),
                (-1, -1, 0.25),
                (-1, 1, 0.25),
                (1, -1, 0.25),
            ]:
                fi = (2 * ic + di) % nf
                fj = (2 * jc + dj) % nf
                f_idx = fi * nf + fj
                rows.append(f_idx)
                cols.append(c_idx)
                vals.append(w)

    P = sp.csr_matrix((vals, (rows, cols)), shape=(Nf, Nc))
    # Normalise so that constant vectors are preserved
    row_sums = np.array(P.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    P = sp.diags(1.0 / row_sums) @ P
    return P


def restriction_2d(P):
    """Full-weighting restriction R = P^T (scaled)."""
    R = P.T.tocsr()
    row_sums = np.array(R.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    R = sp.diags(1.0 / row_sums) @ R
    return R


def weighted_jacobi(A, b, x, omega=2.0 / 3.0, nu=1):
    """Perform nu steps of weighted Jacobi smoothing."""
    d_inv = 1.0 / A.diagonal()
    for _ in range(nu):
        r = b - A @ x
        x = x + omega * d_inv * r
    return x


def vcycle(A_levels, P_levels, R_levels, b, x, nu1=2, nu2=2, omega=2.0 / 3.0,
           smoother_fn=None):
    """Recursive V-cycle multigrid solver.

    Parameters
    ----------
    A_levels : list of sparse matrices, coarsest last.
    P_levels : list of prolongation operators.
    R_levels : list of restriction operators.
    b : right-hand side on the finest level.
    x : initial guess on the finest level.
    nu1, nu2 : pre- and post-smoothing steps.
    omega : Jacobi relaxation weight (used if smoother_fn is None).
    smoother_fn : optional callable(A, b, x, level) -> x for custom smoothing.
    """
    return _vcycle_recursive(A_levels, P_levels, R_levels, b, x,
                             0, nu1, nu2, omega, smoother_fn)


def _vcycle_recursive(As, Ps, Rs, b, x, level, nu1, nu2, omega, smoother_fn):
    if level == len(As) - 1:
        # Coarsest level: direct solve
        return sp.linalg.spsolve(As[level], b)

    # Pre-smooth
    if smoother_fn is not None:
        x = smoother_fn(As[level], b, x, level)
    else:
        x = weighted_jacobi(As[level], b, x, omega, nu1)

    # Restrict residual
    r = b - As[level] @ x
    rc = Rs[level] @ r

    # Recurse on coarse grid
    ec = np.zeros(As[level + 1].shape[0])
    ec = _vcycle_recursive(As, Ps, Rs, rc, ec, level + 1, nu1, nu2,
                           omega, smoother_fn)

    # Prolongate and correct
    x = x + Ps[level] @ ec

    # Post-smooth
    if smoother_fn is not None:
        x = smoother_fn(As[level], b, x, level)
    else:
        x = weighted_jacobi(As[level], b, x, omega, nu2)

    return x
