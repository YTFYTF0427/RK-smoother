"""GNN-based multigrid component learning for unstructured grids.

Implements a message-passing graph neural network (following Luz et al. 2020)
that takes the sparse matrix graph of a PDE operator as input and produces
prolongation weights and/or smoother parameters as output.

The GNN is permutation-equivariant and naturally handles arbitrary graph
topologies (unstructured meshes, AMG coarsening, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class MessagePassingLayer(nn.Module):
    """A single round of message passing on the matrix graph.

    For each edge (i, j) with edge feature e_{ij}:
        m_{ij} = MLP_msg([h_i, h_j, e_{ij}])
    For each node i:
        h_i' = MLP_upd([h_i, aggr_j m_{ij}])
    """

    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        """
        Parameters
        ----------
        h : (N, node_dim) node features.
        edge_index : (2, E) long tensor, source and target indices.
        edge_attr : (E, edge_dim) edge features.

        Returns
        -------
        h_new : (N, node_dim) updated node features.
        """
        src, dst = edge_index
        # Compute messages
        msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        messages = self.msg_mlp(msg_input)  # (E, hidden_dim)

        # Aggregate: sum messages per destination node
        N = h.shape[0]
        agg = torch.zeros(N, messages.shape[-1],
                          device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update node features
        upd_input = torch.cat([h, agg], dim=-1)
        h_new = self.upd_mlp(upd_input)
        return h_new


class MultigridGNN(nn.Module):
    """GNN for learning multigrid prolongation weights and smoother parameters.

    Architecture:
        1. Embed node features (diagonal of A, degree) and edge features
           (off-diagonal entries of A).
        2. K rounds of message passing.
        3. Per-node readout heads for:
           a) Prolongation weights to coarse-grid neighbours.
           b) Local smoother parameters (e.g., Jacobi omega or RK coefficients).

    The number of message-passing rounds K should match the effective stencil
    radius to capture sufficient neighbourhood information.
    """

    def __init__(self, n_mp_rounds=4, node_dim=32, edge_dim=1,
                 hidden_dim=64, n_smoother_params=3):
        """
        Parameters
        ----------
        n_mp_rounds : int
            Number of message-passing rounds (4-8 recommended).
        node_dim : int
            Dimension of node feature embeddings.
        edge_dim : int
            Dimension of raw edge features (typically 1: the off-diagonal entry).
        hidden_dim : int
            Hidden dimension in message/update MLPs.
        n_smoother_params : int
            Number of smoother parameters per node (e.g., 3 for a 3-stage RK).
        """
        super().__init__()
        self.n_mp_rounds = n_mp_rounds
        # Node embedding: (diag_entry, degree) -> node_dim
        self.node_embed = nn.Sequential(
            nn.Linear(2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        # Edge embedding
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Message-passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(node_dim, hidden_dim, hidden_dim)
            for _ in range(n_mp_rounds)
        ])
        # Layer norms for stability
        self.norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(n_mp_rounds)
        ])
        # Readout: prolongation weights (per-edge output)
        self.prolong_head = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Readout: smoother parameters (per-node output)
        self.smoother_head = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_smoother_params),
        )

    def forward(self, node_features, edge_index, edge_attr,
                coarse_fine_edges=None):
        """
        Parameters
        ----------
        node_features : (N, 2) tensor — [diagonal entry, degree] per node.
        edge_index : (2, E) long tensor — sparse matrix graph.
        edge_attr : (E, 1) tensor — off-diagonal entries.
        coarse_fine_edges : optional (2, E_cf) long tensor — edges between
            fine and coarse nodes for prolongation weight prediction.

        Returns
        -------
        smoother_params : (N, n_smoother_params) per-node smoother parameters.
        prolong_weights : (E_cf,) prolongation weights (if coarse_fine_edges
            is given), else None.
        """
        # Embed
        h = self.node_embed(node_features)
        e = self.edge_embed(edge_attr)

        # Message passing with residual connections
        for mp, norm in zip(self.mp_layers, self.norms):
            h_new = mp(h, edge_index, e)
            h = norm(h + h_new)  # residual + layer norm

        # Smoother parameters (per-node)
        smoother_raw = self.smoother_head(h)
        # alphas via sigmoid, beta via softplus
        smoother_params = torch.cat([
            torch.sigmoid(smoother_raw[:, :-1]),
            F.softplus(smoother_raw[:, -1:]),
        ], dim=-1)

        # Prolongation weights (per-edge between coarse and fine)
        prolong_weights = None
        if coarse_fine_edges is not None:
            src, dst = coarse_fine_edges
            edge_feat = torch.cat([h[src], h[dst]], dim=-1)
            prolong_weights = self.prolong_head(edge_feat).squeeze(-1)

        return smoother_params, prolong_weights


def sparse_to_graph(A_sparse):
    """Convert a scipy sparse matrix to graph tensors for the GNN.

    Parameters
    ----------
    A_sparse : scipy sparse matrix (CSR or COO).

    Returns
    -------
    node_features : (N, 2) tensor — [diagonal, degree].
    edge_index : (2, E) long tensor.
    edge_attr : (E, 1) tensor.
    """
    A = sp.coo_matrix(A_sparse)
    N = A.shape[0]

    # Node features: diagonal entry and degree
    diag = np.array(A_sparse.diagonal()).flatten()
    degree = np.array(np.abs(A_sparse).sum(axis=1)).flatten()
    node_features = torch.tensor(
        np.stack([diag, degree], axis=-1), dtype=torch.float32
    )

    # Edge index and attributes (exclude self-loops for edges)
    mask = A.row != A.col
    edge_index = torch.tensor(
        np.stack([A.row[mask], A.col[mask]]), dtype=torch.long
    )
    edge_attr = torch.tensor(
        A.data[mask].reshape(-1, 1), dtype=torch.float32
    )

    return node_features, edge_index, edge_attr


def demo_gnn():
    """Demo: apply the GNN to a small 2D Poisson matrix graph."""
    from .multigrid import assemble_2d_advection_diffusion

    n = 8
    A, _ = assemble_2d_advection_diffusion(n, eps=1.0, b=(0.0, 0.0))
    node_feat, edge_idx, edge_attr = sparse_to_graph(A)

    net = MultigridGNN(n_mp_rounds=4, n_smoother_params=3)
    net.eval()
    with torch.no_grad():
        smoother_params, _ = net(node_feat, edge_idx, edge_attr)

    print(f"GNN demo on {n}x{n} Poisson grid ({A.shape[0]} nodes):")
    print(f"  Smoother params shape: {smoother_params.shape}")
    print(f"  Sample params (node 0): alpha1={smoother_params[0, 0]:.4f}, "
          f"alpha2={smoother_params[0, 1]:.4f}, "
          f"beta={smoother_params[0, 2]:.4f}")
    return smoother_params


if __name__ == "__main__":
    demo_gnn()
