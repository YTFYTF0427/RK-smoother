"""Demo script showcasing all three neural architectures for multigrid.

Runs each architecture on a small advection-diffusion problem and demonstrates:
1. D4-equivariant stencil network (smoother parameter prediction)
2. GNN-based prolongation/smoother learning on the matrix graph
3. Attention-based spatially varying smoother selection
4. LFA-based loss computation for training

Usage:
    cd code && uv run python -m neuralfa.demo
"""

import torch
import numpy as np


def run_lfa_demo():
    """Demonstrate LFA-based loss computation for RK smoother training."""
    from .lfa import smoothing_factor_rk, two_grid_convergence_factor

    print("=" * 60)
    print("1. LFA-based loss function")
    print("=" * 60)

    # Advection-diffusion stencil (eps=0.01, b=(1,0), dt=0.1, h=1/32)
    h = 1.0 / 32
    eps = 0.01
    dt = 0.1
    c = eps * dt / h**2
    a = dt / h  # upwind advection

    stencil = torch.tensor([
        [0.0, -c, 0.0],
        [-c - a, 1.0 + 4 * c + a, -c],
        [0.0, -c, 0.0],
    ], dtype=torch.float64, requires_grad=False)

    # Learnable RK parameters (2-stage)
    alphas = torch.tensor([0.5, 0.7], dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)

    # Compute smoothing factor
    mu = smoothing_factor_rk(stencil, alphas, beta, n_theta=32)
    print(f"  Stencil (advection-diffusion, eps={eps}, h={h:.4f}):")
    print(f"    {stencil[0].tolist()}")
    print(f"    {stencil[1].tolist()}")
    print(f"    {stencil[2].tolist()}")
    print(f"  RK params: alphas={alphas.detach().tolist()}, beta={beta.item():.4f}")
    print(f"  Smoothing factor mu = {mu.item():.6f}")

    # Compute gradient (this is what we'd use for training)
    mu.backward()
    print(f"  d(mu)/d(alphas) = {alphas.grad.tolist()}")
    print(f"  d(mu)/d(beta)   = {beta.grad.item():.6f}")

    # Two-grid convergence factor
    alphas2 = torch.tensor([0.5, 0.7], dtype=torch.float64)
    beta2 = torch.tensor(0.8, dtype=torch.float64)
    rho = two_grid_convergence_factor(stencil, alphas2, beta2, nu1=2, nu2=2)
    print(f"  Ideal-coarse-correction smoothing proxy = {rho.item():.6f}")
    print()


def run_equivariant_demo():
    """Demonstrate D4-equivariant stencil network."""
    from .equivariant import D4StencilNet, D4_PERMS

    print("=" * 60)
    print("2. D4-equivariant stencil network")
    print("=" * 60)

    net = D4StencilNet(n_stages=3, hidden_channels=16)
    net.eval()

    # Non-symmetric advection stencil
    stencil = torch.tensor([
        [0.0, -0.3, 0.0],
        [-0.8, 3.0, -0.5],
        [0.0, -0.4, 0.0],
    ], dtype=torch.float32)

    with torch.no_grad():
        alphas, beta = net(stencil.unsqueeze(0))
    print(f"  Input stencil (advection-diffusion):")
    print(f"    {stencil[0].tolist()}")
    print(f"    {stencil[1].tolist()}")
    print(f"    {stencil[2].tolist()}")
    print(f"  Predicted: alphas={alphas[0].tolist()}, beta={beta.item():.4f}")

    # Verify D4 invariance
    flat = stencil.reshape(9)
    results = []
    for perm in D4_PERMS:
        rotated = flat[torch.tensor(perm)].reshape(1, 3, 3)
        with torch.no_grad():
            a, b = net(rotated)
        results.append((a[0].numpy(), b.item()))

    alpha_vals = np.array([r[0] for r in results])
    beta_vals = np.array([r[1] for r in results])
    print(f"  D4 invariance check (should be ~0):")
    print(f"    Alpha std across D4: {alpha_vals.std(axis=0).max():.8f}")
    print(f"    Beta std across D4:  {beta_vals.std():.8f}")

    # Batch prediction
    batch = torch.randn(16, 3, 3)
    with torch.no_grad():
        alphas_batch, beta_batch = net(batch)
    print(f"  Batch prediction: {batch.shape[0]} stencils -> "
          f"alphas {alphas_batch.shape}, beta {beta_batch.shape}")
    print()


def run_gnn_demo():
    """Demonstrate GNN on a sparse matrix graph."""
    from .gnn import MultigridGNN, sparse_to_graph
    from .multigrid import assemble_2d_advection_diffusion

    print("=" * 60)
    print("3. GNN-based multigrid component learning")
    print("=" * 60)

    n = 8
    A, _ = assemble_2d_advection_diffusion(n, eps=1.0, b=(0.0, 0.0))
    node_feat, edge_idx, edge_attr = sparse_to_graph(A)

    net = MultigridGNN(n_mp_rounds=4, n_smoother_params=3)
    net.eval()

    with torch.no_grad():
        smoother_params, _ = net(node_feat, edge_idx, edge_attr)

    print(f"  Input: {n}x{n} Poisson grid -> {A.shape[0]} nodes, "
          f"{edge_idx.shape[1]} edges")
    print(f"  Smoother params shape: {smoother_params.shape}")
    print(f"  Sample output (node 0):")
    print(f"    alpha1={smoother_params[0, 0]:.4f}, "
          f"alpha2={smoother_params[0, 1]:.4f}, "
          f"beta={smoother_params[0, 2]:.4f}")
    print(f"  Parameter statistics:")
    print(f"    alpha1: mean={smoother_params[:, 0].mean():.4f}, "
          f"std={smoother_params[:, 0].std():.4f}")
    print(f"    beta:   mean={smoother_params[:, 2].mean():.4f}, "
          f"std={smoother_params[:, 2].std():.4f}")
    print()


def run_attention_demo():
    """Demonstrate attention-based smoother selection."""
    from .attention import AttentionSmootherSelector, extract_stencil_patches
    from .multigrid import assemble_2d_advection_diffusion

    print("=" * 60)
    print("4. Attention-based smoother selection")
    print("=" * 60)

    n = 16
    A, _ = assemble_2d_advection_diffusion(n, eps=0.01, b=(1.0, 0.5))
    patches_np = extract_stencil_patches(A, n, patch_size=3)
    patches = torch.tensor(patches_np, dtype=torch.float32)

    net = AttentionSmootherSelector(
        n_prototypes=8, n_params=3, patch_size=3,
        embed_dim=32, n_heads=4,
    )
    net.eval()

    with torch.no_grad():
        params, attn_weights = net(patches)

    print(f"  Input: {n}x{n} advection-diffusion grid (eps=0.01)")
    print(f"  {net.n_prototypes} smoother prototypes, {net.n_heads} attention heads")
    print(f"  Parameters shape: {params.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    # Spatial variation
    beta_field = params[:, -1].reshape(n, n)
    print(f"  Beta field statistics:")
    print(f"    min={beta_field.min():.4f}, max={beta_field.max():.4f}, "
          f"std={beta_field.std():.4f}")

    # Prototype activation
    mean_attn = attn_weights.mean(dim=0)
    top_k = mean_attn.argsort(descending=True)[:3]
    print(f"  Top 3 active prototypes: {top_k.tolist()}")
    for idx in top_k:
        proto = net.prototypes.get_params()[idx]
        print(f"    Prototype {idx.item()}: raw params = "
              f"{proto.detach().tolist()}, "
              f"weight = {mean_attn[idx]:.4f}")
    print()


def main():
    print()
    print("Neural Architectures for Multigrid via Local Fourier Analysis")
    print("=" * 60)
    print()

    run_lfa_demo()
    run_equivariant_demo()
    run_gnn_demo()
    run_attention_demo()

    print("All demos completed.")


if __name__ == "__main__":
    main()
