"""Attention-based smoother selection for spatially varying parameters.

For variable-coefficient problems, the optimal smoother adapts locally — near
boundaries, coefficient discontinuities, or regions of strong advection.  This
module uses cross-attention between local stencil patches and a learned set of
"prototype" smoother templates to produce spatially varying smoother parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SmootherPrototypes(nn.Module):
    """A bank of K learnable smoother prototypes.

    Each prototype is a vector of smoother parameters (e.g., RK alpha/beta
    values).  During inference, the network computes attention weights over
    these prototypes based on the local stencil, then outputs a weighted
    combination.
    """

    def __init__(self, n_prototypes, n_params, embed_dim):
        """
        Parameters
        ----------
        n_prototypes : int
            Number of prototype smoothers (K).
        n_params : int
            Dimensionality of smoother parameters (e.g., s-1 alphas + 1 beta).
        embed_dim : int
            Dimension of the key/value embeddings.
        """
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_params = n_params
        # Learnable prototype parameters
        self.proto_params = nn.Parameter(
            torch.randn(n_prototypes, n_params) * 0.1
        )
        # Learnable keys for attention
        self.proto_keys = nn.Parameter(
            torch.randn(n_prototypes, embed_dim) * 0.1
        )

    def get_keys(self):
        """Return prototype keys: (K, embed_dim)."""
        return self.proto_keys

    def get_params(self):
        """Return prototype parameters: (K, n_params)."""
        return self.proto_params


class StencilPatchEncoder(nn.Module):
    """Encode a local stencil patch into a query vector for attention.

    Takes a local p x p patch of stencil coefficients around a grid point
    and maps it to an embedding suitable for cross-attention with smoother
    prototypes.
    """

    def __init__(self, patch_size=3, embed_dim=32):
        """
        Parameters
        ----------
        patch_size : int
            Size of the local stencil patch (3 or 5).
        embed_dim : int
            Output embedding dimension.
        """
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Linear(patch_size * patch_size, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, patches):
        """
        Parameters
        ----------
        patches : (B, patch_size, patch_size) or (B, patch_size^2)

        Returns
        -------
        queries : (B, embed_dim)
        """
        B = patches.shape[0]
        x = patches.reshape(B, -1)
        return self.encoder(x)


class AttentionSmootherSelector(nn.Module):
    """Cross-attention network for spatially varying smoother selection.

    Architecture:
        1. Encode local stencil patch -> query vector.
        2. Cross-attend over K smoother prototypes using learned keys.
        3. Output = weighted combination of prototype parameters.

    This produces a smooth, spatially varying field of smoother parameters
    that adapts to local operator structure while maintaining the sparsity
    pattern of the smoother (it only changes scalar parameters, not the
    smoother's graph structure).

    Parameters
    ----------
    n_prototypes : int
        Number of prototype smoothers.  More prototypes = more expressive
        but harder to train.  8-16 is a good starting point.
    n_params : int
        Number of smoother parameters per grid point.
    patch_size : int
        Local stencil patch size (3 or 5).
    embed_dim : int
        Embedding dimension for attention.
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, n_prototypes=8, n_params=3, patch_size=3,
                 embed_dim=32, n_heads=4):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_params = n_params
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.head_dim = embed_dim // n_heads

        self.patch_encoder = StencilPatchEncoder(patch_size, embed_dim)
        self.prototypes = SmootherPrototypes(n_prototypes, n_params, embed_dim)

        # Multi-head attention projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)

        # Temperature parameter for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, patches):
        """
        Parameters
        ----------
        patches : (B, patch_size, patch_size) or (B, patch_size^2)
            Local stencil patches for each grid point.

        Returns
        -------
        params : (B, n_params)
            Spatially varying smoother parameters.
        attn_weights : (B, n_prototypes)
            Attention weights showing which prototypes are active.
        """
        # Encode stencil patches to queries
        queries = self.patch_encoder(patches)  # (B, embed_dim)
        q = self.W_q(queries)                  # (B, embed_dim)

        # Get prototype keys
        keys = self.prototypes.get_keys()      # (K, embed_dim)
        k = self.W_k(keys)                     # (K, embed_dim)

        # Multi-head attention
        B, K = q.shape[0], k.shape[0]
        q = q.reshape(B, self.n_heads, self.head_dim)  # (B, H, d)
        k = k.reshape(K, self.n_heads, self.head_dim)  # (K, H, d)

        # Scaled dot-product attention per head
        # (B, H, d) x (K, H, d) -> (B, H, K) via einsum
        attn_logits = torch.einsum("bhd,khd->bhk", q, k)
        attn_logits = attn_logits / (self.head_dim ** 0.5 * self.temperature)

        # Average over heads, then softmax
        attn_logits = attn_logits.mean(dim=1)  # (B, K)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, K)

        # Weighted combination of prototype parameters
        proto_params = self.prototypes.get_params()  # (K, n_params)
        params_raw = torch.einsum("bk,kp->bp", attn_weights, proto_params)

        # Enforce parameter constraints
        params = torch.cat([
            torch.sigmoid(params_raw[:, :-1]),   # alphas in (0, 1)
            F.softplus(params_raw[:, -1:]),   # beta > 0
        ], dim=-1)

        return params, attn_weights

    def predict_grid(self, stencil_field):
        """Predict spatially varying parameters for an entire grid.

        Parameters
        ----------
        stencil_field : (ny, nx, 3, 3) tensor
            The local 3x3 stencil at each grid point.

        Returns
        -------
        param_field : (ny, nx, n_params)
        attn_field : (ny, nx, n_prototypes)
        """
        ny, nx = stencil_field.shape[:2]
        patches = stencil_field.reshape(ny * nx, 3, 3)
        params, attn = self(patches)
        return params.reshape(ny, nx, -1), attn.reshape(ny, nx, -1)


def extract_stencil_patches(A_sparse, n, patch_size=3):
    """Extract local stencil patches from a sparse matrix on an n x n grid.

    For each grid point (i, j), extracts the patch_size x patch_size block
    of the operator stencil centered at that point.  Uses periodic wrapping.

    Parameters
    ----------
    A_sparse : scipy sparse matrix of shape (n*n, n*n).
    n : int, grid size.
    patch_size : int, must be odd.

    Returns
    -------
    patches : (n*n, patch_size, patch_size) numpy array.
    """
    import scipy.sparse as sp
    A = A_sparse.toarray() if sp.issparse(A_sparse) else A_sparse
    half = patch_size // 2
    N = n * n
    patches = np.zeros((N, patch_size, patch_size))

    for i in range(n):
        for j in range(n):
            row_idx = i * n + j
            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    ni = (i + di) % n
                    nj = (j + dj) % n
                    col_idx = ni * n + nj
                    patches[row_idx, di + half, dj + half] = A[row_idx, col_idx]

    return patches


def demo_attention():
    """Demo: attention-based smoother selection on variable-coefficient problem."""
    from .multigrid import assemble_2d_advection_diffusion

    n = 16
    # Build a problem with spatially varying advection direction
    A, base_stencil = assemble_2d_advection_diffusion(
        n, eps=0.01, b=(1.0, 0.5)
    )

    # Extract local stencil patches
    patches_np = extract_stencil_patches(A, n, patch_size=3)
    patches = torch.tensor(patches_np, dtype=torch.float32)

    net = AttentionSmootherSelector(
        n_prototypes=8, n_params=3, patch_size=3,
        embed_dim=32, n_heads=4,
    )
    net.eval()

    with torch.no_grad():
        params, attn_weights = net(patches)

    print(f"Attention smoother demo on {n}x{n} grid:")
    print(f"  Parameters shape: {params.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    # Show which prototypes are most active
    mean_attn = attn_weights.mean(dim=0)
    top_k = mean_attn.argsort(descending=True)[:3]
    print(f"  Top 3 active prototypes: {top_k.tolist()}")
    print(f"  Mean attention weights: {mean_attn[top_k].tolist()}")

    # Show spatial variation of beta
    beta_field = params[:, -1].reshape(n, n)
    print(f"  Beta range: [{beta_field.min():.4f}, {beta_field.max():.4f}]")
    print(f"  Beta std: {beta_field.std():.4f}")
    return params, attn_weights


if __name__ == "__main__":
    demo_attention()
