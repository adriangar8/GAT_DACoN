"""
Graph Attention Network (GAT) module for topology-aware segment correspondence.

After DACoN's MLP fusion produces per-segment features, this module enriches
each segment with information from its spatially adjacent segments. Two
segments are adjacent if they share at least one boundary pixel (8-connected,
equivalent to 3×3 morphological dilation).

Architecture:
    Layer 1 – 4 heads × 32 dims/head (concat → 128-D), ELU, dropout 0.1
    Layer 2 – 1 head  × 128 dims (mean pooling),         no activation
    Residual – enriched + original features
    LayerNorm after residual

The module is applied independently to each (batch, frame) pair, preserving
DACoN's key property that reference and target images are processed separately
(enabling any number of reference images).

Reference:
    Veličković et al., "Graph Attention Networks", ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_adjacency(seg_image: torch.Tensor, seg_num: int) -> torch.Tensor:
    """Build a symmetric binary adjacency matrix from a segment label image.

    Two segments are considered adjacent when they share at least one pair of
    8-connected pixels (i.e., the boundary of one segment touches the other).
    Every segment also receives a self-loop.

    Edge cases handled:
    - Segments with no spatial neighbours: only the self-loop is present.
    - Label indices that exceed seg_num (should not occur in practice):
      clamped to stay in range.
    - Images smaller than 2×2: only self-loops are returned.

    Args:
        seg_image: (H, W) tensor (float32 or int) with integer segment labels
                   in the range 1..seg_num. Pixels with label 0 are treated as
                   background/unlabelled and are ignored during adjacency
                   construction.
        seg_num:   Number of segments (graph nodes), i.e. L.

    Returns:
        adj: (L, L) float32 tensor – symmetric binary adjacency matrix with
             ones on the diagonal (self-loops included).
    """
    device = seg_image.device
    L = seg_num

    if L == 0:
        return torch.zeros(0, 0, device=device)

    seg = seg_image.long()          # ensure integer labels
    H, W = seg.shape

    adj = torch.zeros(L, L, device=device)

    # Self-loops
    arange = torch.arange(L, device=device)
    adj[arange, arange] = 1.0

    if H < 2 or W < 2:
        return adj

    def _add_edges(a: torch.Tensor, b: torch.Tensor) -> None:
        """Mark all adjacent pairs (a[k], b[k]) in the adjacency matrix."""
        valid = (a > 0) & (b > 0) & (a != b)
        if not valid.any():
            return
        ai = (a[valid] - 1).clamp(0, L - 1)
        bi = (b[valid] - 1).clamp(0, L - 1)
        adj[ai, bi] = 1.0
        adj[bi, ai] = 1.0

    # 4-connected (horizontal and vertical neighbours)
    _add_edges(seg[:, :-1].reshape(-1), seg[:, 1:].reshape(-1))   # left–right
    _add_edges(seg[:-1, :].reshape(-1), seg[1:, :].reshape(-1))   # top–bottom

    # Diagonal neighbours → full 8-connectivity (matches 3×3 dilation)
    _add_edges(seg[:-1, :-1].reshape(-1), seg[1:, 1:].reshape(-1))  # TL–BR
    _add_edges(seg[:-1, 1:].reshape(-1),  seg[1:, :-1].reshape(-1)) # TR–BL

    return adj


# ---------------------------------------------------------------------------
# GAT layers
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """Single multi-head graph attention layer (dense adjacency matrix version).

    Uses dense (N×N) attention matrices; efficient for small graphs (N ≲ 300).

    Args:
        in_dim:    Input feature dimension.
        out_dim:   Output feature dimension *per head*.
        num_heads: Number of attention heads.
        dropout:   Dropout probability on attention weights.
        concat:    If True, concatenate head outputs (output dim = num_heads*out_dim).
                   If False, average them (output dim = out_dim).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat

        # Shared linear projection W : R^{in_dim} → R^{num_heads * out_dim}
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        # Per-head attention parameters (split into source and destination vectors)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (N, in_dim) node features.
            adj: (N, N) binary adjacency matrix (self-loops already present).

        Returns:
            out: (N, num_heads * out_dim) if concat=True,
                 (N, out_dim)             if concat=False.
        """
        N = x.size(0)

        # Linear transform → (N, H, D)  where H = num_heads, D = out_dim
        Wf = self.W(x).view(N, self.num_heads, self.out_dim)

        # Attention logits: e[i, j, h] = LeakyReLU(a_src[h]·Wf[i,h] + a_dst[h]·Wf[j,h])
        e_src = torch.einsum("nhd,hd->nh", Wf, self.a_src)   # (N, H)
        e_dst = torch.einsum("nhd,hd->nh", Wf, self.a_dst)   # (N, H)
        # Broadcast to (N_i, N_j, H)
        e = self.leaky_relu(e_src.unsqueeze(1) + e_dst.unsqueeze(0))  # (N, N, H)

        # Mask non-edges: set to -inf so softmax assigns zero weight
        e = e.masked_fill((adj == 0).unsqueeze(-1), float("-inf"))

        # Softmax over incoming neighbours for each node i  (dim=1 → the j-axis)
        alpha = F.softmax(e, dim=1)   # (N, N, H) — rows of adj correspond to i
        alpha = self.dropout(alpha)

        # Weighted aggregation: h[i, h, d] = Σ_j  α[i, j, h] * Wf[j, h, d]
        h = torch.einsum("ijh,jhd->ihd", alpha, Wf)  # (N, H, D)

        if self.concat:
            # Concatenate heads and apply ELU
            return self.elu(h.reshape(N, self.num_heads * self.out_dim))
        else:
            # Average over heads (no activation at final layer)
            return h.mean(dim=1)   # (N, D)


# ---------------------------------------------------------------------------
# Full 2-layer GAT module
# ---------------------------------------------------------------------------

class GATModule(nn.Module):
    """Topology-aware segment feature enrichment via a 2-layer GAT.

    Enriches DACoN's per-segment features (output of the MLP fusion step) with
    spatial context from neighbouring segments before cosine-similarity matching.

    Spec (from the paper):
        Layer 1 : 4 heads, 32 dims/head (concat → 128), ELU, dropout 0.1
        Layer 2 : 1 head,  128 dims (mean pool),         no activation
        Residual : enriched + original; LayerNorm after

    Args:
        feat_dim: Segment feature dimension (must be divisible by 4). Default 128.
        dropout:  Dropout rate for GAT attention weights. Default 0.1.
    """

    def __init__(self, feat_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()

        if feat_dim % 4 != 0:
            raise ValueError(
                f"feat_dim must be divisible by 4 (for 4 attention heads); got {feat_dim}."
            )

        per_head_dim = feat_dim // 4  # 32 when feat_dim=128

        # Layer 1: 4 heads, concatenated → feat_dim
        self.gat1 = GATLayer(feat_dim, per_head_dim, num_heads=4, dropout=dropout, concat=True)

        # Layer 2: 1 head averaged → feat_dim
        self.gat2 = GATLayer(feat_dim, feat_dim, num_heads=1, dropout=dropout, concat=False)

        self.norm = nn.LayerNorm(feat_dim)

    def _enrich_single(self, feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply 2-layer GAT + residual to one image's segment features.

        Args:
            feats: (N, C) – features for N segments.
            adj:   (N, N) – adjacency matrix with self-loops.

        Returns:
            (N, C) topology-enriched features.
        """
        h = self.gat1(feats, adj)    # (N, C)  ELU applied inside
        h = self.gat2(h, adj)         # (N, C)  no activation
        return self.norm(feats + h)   # residual + LayerNorm

    def forward(
        self,
        seg_feats: torch.Tensor,
        seg_images: torch.Tensor,
        seg_nums: torch.Tensor,
    ) -> torch.Tensor:
        """Enrich segment features with topology information.

        Processes each (batch item, frame) independently: the adjacency graph
        for image (b, s) cannot influence the features of image (b', s').

        Args:
            seg_feats:  (B, S, L, C) – fused segment features from DACoN's MLP.
            seg_images: (B, S, H, W) – integer segment-label images (stored as
                        float32 after move_data_to_device; labels are 1-indexed).
            seg_nums:   (B, S)       – actual segment count per image.

        Returns:
            enriched: (B, S, L, C) – topology-enriched features, same shape.
        """
        B, S, L, C = seg_feats.shape
        output = seg_feats.clone()

        for b in range(B):
            for s in range(S):
                n = int(seg_nums[b, s].item())
                if n == 0:
                    continue

                feats   = seg_feats[b, s, :n]         # (n, C)
                seg_img = seg_images[b, s]             # (H, W)
                adj     = build_adjacency(seg_img, n)  # (n, n)

                output[b, s, :n] = self._enrich_single(feats, adj)

        return output
