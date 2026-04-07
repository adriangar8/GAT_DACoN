"""
GAT Attention Visualization for GAT-DACoN paper figures.

Generates three types of figures per interesting sample:
  1. graph_overlay/   – adjacency graph drawn on top of the target sketch
  2. attention/       – attention FROM a query segment TO its neighbours
  3. matching/        – baseline vs GAT matching, side-by-side

"Interesting" samples are those where the baseline gets the segment match
wrong but GAT gets it right — exactly the case the paper argues about.

Usage (from project root):
    python dacon/visualize_gat_attention.py \\
        --checkpoint checkpoints/gat_aug/dacon_20260403_145224.pth \\
        --baseline_checkpoint checkpoints/aug_only/dacon_XXXXXXXX_XXXXXX.pth \\
        --config configs/test_gat_aug.yaml \\
        --baseline_config configs/test_aug_only.yaml \\
        --data_root dataset/PaintBucket_Char_v2/test/PaintBucket_Char_v2/ \\
        --output_dir visualizations/ \\
        --num_samples 8
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on compute nodes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# The script lives in dacon/ so models/, data/, utils/ are importable directly.

from models import DACoNModel
from data import DACoNSingleDataset, dacon_single_pad_collate_fn
from utils import (
    load_config,
    make_single_data_list,
    get_folder_names,
)
from models.gat_module import build_adjacency

# ── visual style ────────────────────────────────────────────────────────────
# Paul Tol colorblind-friendly palette
_COL_QUERY    = "#4477AA"   # blue  — query segment
_COL_CORRECT  = "#228833"   # green — GAT / correct match
_COL_WRONG    = "#CC3311"   # red   — baseline / wrong match
_COL_EDGE     = "#BBBBBB"   # light gray — adjacency edges
_COL_NODE_BDR = "#333333"   # dark gray  — node borders

_DPI          = 300
_FIG_W_1COL   = 3.5         # inches, 1-column width
_FIG_W_2COL   = 7.0         # inches, 2-column width

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "figure.dpi":        _DPI,
    "savefig.dpi":       _DPI,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

# ── helpers ─────────────────────────────────────────────────────────────────

def _tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float [0,1] tensor to (H, W, 3) numpy [0,1]."""
    arr = t.cpu().float().numpy()
    if arr.ndim == 2:               # already grayscale (H, W)
        return np.stack([arr] * 3, axis=-1)
    arr = arr[:3].transpose(1, 2, 0)  # take first 3 channels, → (H,W,3)
    return np.clip(arr, 0.0, 1.0)


def _seg_to_masks(seg_image_np: np.ndarray, n: int) -> np.ndarray:
    """Convert (H, W) integer label image to (n, H, W) boolean masks.

    Labels in seg_image are 1-indexed (1..n); background is 0.
    """
    return np.stack([(seg_image_np == (i + 1)) for i in range(n)], axis=0)


def _compute_centroids(masks: np.ndarray) -> np.ndarray:
    """Compute centroids from (n, H, W) boolean masks.

    Returns (n, 2) array of [row, col] centroids.
    """
    n, H, W = masks.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    centroids = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        m = masks[i]
        if m.any():
            centroids[i] = [yy[m].mean(), xx[m].mean()]
        else:
            centroids[i] = [H / 2.0, W / 2.0]
    return centroids


def _find_gt_matches(tgt_colors: np.ndarray, ref_colors: np.ndarray) -> np.ndarray:
    """Nearest-neighbour match in RGB space.

    Args:
        tgt_colors: (n_tgt, 3) float [0,1]
        ref_colors: (n_ref, 3) float [0,1]

    Returns:
        (n_tgt,) int array of reference segment indices (0-indexed).
    """
    dists = np.linalg.norm(
        tgt_colors[:, np.newaxis] - ref_colors[np.newaxis], axis=-1
    )  # (n_tgt, n_ref)
    return dists.argmin(axis=1)


def _save_both(fig: plt.Figure, path_no_ext: str) -> None:
    """Save figure as both PDF (vector) and PNG (quick view)."""
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png")
    plt.close(fig)


# ── Visualization 1: Graph Overlay ─────────────────────────────────────────

def visualize_graph_overlay(
    line_drawing: np.ndarray,
    segment_masks: np.ndarray,
    segment_colors: np.ndarray,
    adjacency_matrix: np.ndarray,
    save_path: str,
) -> None:
    """Draw the segment adjacency graph overlaid on a line drawing.

    Args:
        line_drawing:     (H, W, 3) float [0,1] — the sketch.
        segment_masks:    (M, H, W) bool — binary mask per segment.
        segment_colors:   (M, 3) float [0,1] — RGB colour of each segment.
        adjacency_matrix: (M, M) float — symmetric binary adjacency.
        save_path:        Output path WITHOUT extension (both .pdf and .png saved).
    """
    M = segment_masks.shape[0]
    centroids = _compute_centroids(segment_masks)   # (M, 2) [row, col]

    fig, ax = plt.subplots(figsize=(_FIG_W_1COL, _FIG_W_1COL))
    ax.imshow(line_drawing, interpolation="bilinear")
    ax.axis("off")

    # Draw edges first (so nodes render on top)
    for i in range(M):
        for j in range(i + 1, M):
            if adjacency_matrix[i, j] > 0:
                ry, rx = centroids[i]
                sy, sx = centroids[j]
                ax.plot([rx, sx], [ry, sy],
                        color=_COL_EDGE, linewidth=0.6, alpha=0.8, zorder=2)

    # Draw nodes coloured by segment ground-truth colour
    for i in range(M):
        cy, cx = centroids[i]
        color = np.clip(segment_colors[i], 0, 1)
        ax.scatter(cx, cy, s=18, color=color,
                   edgecolors=_COL_NODE_BDR, linewidths=0.5,
                   zorder=3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    _save_both(fig, save_path)


# ── Visualization 2: Attention Weights ─────────────────────────────────────

def visualize_attention_weights(
    line_drawing: np.ndarray,
    segment_masks: np.ndarray,
    segment_colors: np.ndarray,
    adjacency_matrix: np.ndarray,
    attention_weights,
    query_segment_idx: int,
    save_path: str,
    layer_idx: int = 0,
    head_idx: int = 0,
) -> None:
    """Show which neighbours a query segment attends to.

    Args:
        line_drawing:       (H, W, 3) float [0,1].
        segment_masks:      (M, H, W) bool.
        segment_colors:     (M, 3) float [0,1].
        adjacency_matrix:   (M, M) float.
        attention_weights:  list of 2 numpy arrays returned by GATModule —
                            [alpha_l1 (n,n,4), alpha_l2 (n,n,1)].
                            alpha[i,j,h] = attention FROM i TO j in head h.
        query_segment_idx:  0-indexed segment whose attention is visualised.
        save_path:          Output path WITHOUT extension.
        layer_idx:          Which GAT layer to visualise (0 or 1).
        head_idx:           Which attention head (0..3 for layer 0; 0 for layer 1).
    """
    M = segment_masks.shape[0]
    H_img, W_img = line_drawing.shape[:2]
    centroids = _compute_centroids(segment_masks)

    # Extract per-neighbour attention weights for the query node
    alpha_full = np.nan_to_num(attention_weights[layer_idx][:, :, head_idx])
    query_weights = alpha_full[query_segment_idx]   # (n,) sums to ~1

    # ── figure layout: image panel + colorbar ──────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(_FIG_W_1COL + 0.6, _FIG_W_1COL),
        gridspec_kw={"width_ratios": [1, 0.06], "wspace": 0.08},
    )
    ax, ax_cb = axes

    # Background sketch
    ax.imshow(line_drawing, interpolation="bilinear")

    # Colour overlay: neighbours filled proportionally to attention weight
    attn_cmap = plt.cm.YlOrRd
    overlay = np.zeros((H_img, W_img, 4), dtype=np.float32)
    max_w = query_weights.max() if query_weights.max() > 0 else 1.0

    for j in range(M):
        if j == query_segment_idx:
            continue
        w = float(query_weights[j])
        if w < 1e-6:
            continue
        rgba = attn_cmap(w / max_w)
        mask = segment_masks[j]
        overlay[mask, :3] = rgba[:3]
        overlay[mask, 3]  = 0.65 * (w / max_w)

    ax.imshow(overlay, interpolation="none")

    # Query segment: solid blue fill
    q_overlay = np.zeros((H_img, W_img, 4), dtype=np.float32)
    q_overlay[segment_masks[query_segment_idx]] = [
        *[c / 255 for c in (0x44, 0x77, 0xAA)], 0.55
    ]
    ax.imshow(q_overlay, interpolation="none")

    # Draw edges from query to neighbours, thickness ∝ weight
    qy, qx = centroids[query_segment_idx]
    for j in range(M):
        if j == query_segment_idx:
            continue
        w = float(query_weights[j])
        if adjacency_matrix[query_segment_idx, j] < 1 or w < 1e-6:
            continue
        jy, jx = centroids[j]
        lw = 0.5 + 4.0 * (w / max_w)
        rgba = attn_cmap(w / max_w)
        ax.plot([qx, jx], [qy, jy], color=rgba[:3], linewidth=lw,
                alpha=0.9, zorder=4)
        # Label the weight
        mx, my = (qx + jx) / 2, (qy + jy) / 2
        ax.text(mx, my, f"{w:.2f}", fontsize=5, ha="center", va="center",
                color="black", zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))

    # Query centroid marker
    ax.scatter(qx, qy, s=40, color=_COL_QUERY,
               edgecolors="white", linewidths=0.8, zorder=6)

    ax.set_title(
        f"Query seg {query_segment_idx} — layer {layer_idx + 1}, head {head_idx}",
        fontsize=8,
    )
    ax.axis("off")

    # Colorbar
    sm = ScalarMappable(cmap=attn_cmap, norm=Normalize(vmin=0, vmax=max_w))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("Attention weight", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    _save_both(fig, save_path)


# ── Visualization 3: Matching Comparison ───────────────────────────────────

def visualize_matching_comparison(
    target_line_drawing: np.ndarray,
    target_masks: np.ndarray,
    ref_color_image: np.ndarray,
    ref_masks: np.ndarray,
    query_seg_idx: int,
    baseline_match_idx: int,
    gat_match_idx: int,
    gt_match_idx: int,
    save_path: str,
) -> None:
    """Side-by-side comparison of baseline vs GAT matching.

    Left:  target sketch with the query segment highlighted.
    Right: reference colour image with the baseline match (red) and GAT match
           (green) highlighted.  Since we only call this for "interesting"
           samples, gat_match_idx == gt_match_idx is always true here.
    Arrows cross the panel boundary: red to baseline, green to GAT.

    Args:
        target_line_drawing: (H, W, 3) float [0,1].
        target_masks:        (M_tgt, H, W) bool.
        ref_color_image:     (H, W, 3) float [0,1].
        ref_masks:           (M_ref, H, W) bool.
        query_seg_idx:       0-indexed target segment.
        baseline_match_idx:  0-indexed reference segment predicted by baseline.
        gat_match_idx:       0-indexed reference segment predicted by GAT.
        gt_match_idx:        0-indexed ground-truth reference segment.
        save_path:           Output path WITHOUT extension.
    """
    tgt_centroids = _compute_centroids(target_masks)
    ref_centroids = _compute_centroids(ref_masks)
    H_t, W_t = target_line_drawing.shape[:2]
    H_r, W_r = ref_color_image.shape[:2]

    fig, (ax_tgt, ax_ref) = plt.subplots(
        1, 2,
        figsize=(_FIG_W_2COL, _FIG_W_2COL * 0.55),
        gridspec_kw={"wspace": 0.12},
    )

    # ── Left panel: target sketch + query highlight ─────────────────────
    ax_tgt.imshow(target_line_drawing, interpolation="bilinear")
    q_ov = np.zeros((H_t, W_t, 4), dtype=np.float32)
    q_ov[target_masks[query_seg_idx]] = [0.267, 0.467, 0.667, 0.50]
    ax_tgt.imshow(q_ov, interpolation="none")
    qy, qx = tgt_centroids[query_seg_idx]
    ax_tgt.scatter(qx, qy, s=30, color=_COL_QUERY,
                   edgecolors="white", linewidths=0.7, zorder=5)
    ax_tgt.set_title("Target sketch  (query = blue)", fontsize=8)
    ax_tgt.axis("off")

    # ── Right panel: reference image + match highlights ──────────────────
    ax_ref.imshow(ref_color_image, interpolation="bilinear")

    def _highlight_ref(seg_idx, color_hex, alpha=0.55):
        ov = np.zeros((H_r, W_r, 4), dtype=np.float32)
        c = tuple(int(color_hex[i:i+2], 16) / 255 for i in (1, 3, 5))
        ov[ref_masks[seg_idx]] = [*c, alpha]
        ax_ref.imshow(ov, interpolation="none")
        cy, cx = ref_centroids[seg_idx]
        ax_ref.scatter(cx, cy, s=30, color=color_hex,
                       edgecolors="white", linewidths=0.7, zorder=5)

    # Draw baseline match (red) first so GAT overlaps if they coincide
    _highlight_ref(baseline_match_idx, _COL_WRONG)
    _highlight_ref(gat_match_idx,      _COL_CORRECT)

    legend_handles = [
        mpatches.Patch(color=_COL_WRONG,   label="Baseline (wrong)"),
        mpatches.Patch(color=_COL_CORRECT, label="GAT-DACoN (correct)"),
    ]
    ax_ref.legend(
        handles=legend_handles,
        fontsize=6, loc="lower right",
        framealpha=0.85, edgecolor="gray",
    )
    ax_ref.set_title("Reference image  (match highlights)", fontsize=8)
    ax_ref.axis("off")

    # ── Cross-panel arrows ───────────────────────────────────────────────
    base_cy, base_cx = ref_centroids[baseline_match_idx]
    gat_cy,  gat_cx  = ref_centroids[gat_match_idx]

    for ref_cx_arrow, ref_cy_arrow, color in [
        (base_cx, base_cy, _COL_WRONG),
        (gat_cx,  gat_cy,  _COL_CORRECT),
    ]:
        con = ConnectionPatch(
            xyA=(qx,           qy),
            xyB=(ref_cx_arrow, ref_cy_arrow),
            coordsA="data", coordsB="data",
            axesA=ax_tgt,   axesB=ax_ref,
            color=color, linewidth=1.4,
            arrowstyle="->", mutation_scale=10,
            zorder=10,
        )
        fig.add_artist(con)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    _save_both(fig, save_path)


# ── Model helpers ────────────────────────────────────────────────────────────

def _load_model(config: dict, checkpoint_path: str, device: torch.device) -> DACoNModel:
    version = config.get("version", "1_1")
    model = DACoNModel(config["network"], version).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _get_features(
    model: DACoNModel,
    line_image: torch.Tensor,
    seg_image: torch.Tensor,
    seg_num: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run model._process_single; returns (B, L, C) segment features."""
    line_image = line_image.to(device).float()
    seg_image  = seg_image.to(device).float()
    seg_num    = seg_num.to(device)
    with torch.no_grad():
        feats, _ = model._process_single(line_image, seg_image, seg_num)
    return feats  # (B, L, C)


def _get_features_and_attention(
    model_gat: DACoNModel,
    line_image: torch.Tensor,
    seg_image: torch.Tensor,
    seg_num: torch.Tensor,
    device: torch.device,
):
    """Run GAT model, returning (B, L, C) enriched features + attention dict.

    Returns:
        enriched_feats: (B, L, C)
        attn_data: dict with keys 'adj' and 'layer_alphas' for image (0, 0).
    """
    line_image = line_image.to(device).float()
    seg_image  = seg_image.to(device).float()
    seg_num    = seg_num.to(device)

    # Step 1 – get pre-GAT features by temporarily disabling the GAT pass
    was_gat = model_gat.use_gat
    model_gat.use_gat = False
    with torch.no_grad():
        pre_gat_feats, _ = model_gat._process_single(line_image, seg_image, seg_num)
    model_gat.use_gat = was_gat

    # Step 2 – run GAT with attention capture
    feats_4d    = pre_gat_feats.unsqueeze(1)    # (B, 1, L, C)
    seg_imgs_4d = seg_image.unsqueeze(1)        # (B, 1, H, W)
    seg_nums_2d = seg_num.unsqueeze(1)          # (B, 1)

    with torch.no_grad():
        enriched_4d, attentions = model_gat.gat(
            feats_4d, seg_imgs_4d, seg_nums_2d, return_attention=True
        )

    enriched_feats = enriched_4d.squeeze(1)     # (B, L, C)
    attn_data      = attentions.get((0, 0), None)
    return enriched_feats, attn_data


def _compute_sim_map(
    model: DACoNModel,
    feats_ref: torch.Tensor,
    feats_tgt: torch.Tensor,
    n_ref: int,
    n_tgt: int,
) -> np.ndarray:
    """Cosine similarity map, clipped to valid segments.

    Args:
        feats_ref: (1, L_ref, C)
        feats_tgt: (1, L_tgt, C)

    Returns:
        (n_tgt, n_ref) numpy float32.
    """
    sim = model.get_seg_cos_sim(
        feats_ref.unsqueeze(1),   # (1, 1, L_ref, C)
        feats_tgt.unsqueeze(1),   # (1, 1, L_tgt, C)
    )  # → (1, L_tgt, L_ref)
    return sim[0, :n_tgt, :n_ref].cpu().numpy()


# ── Main batch script ────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ("graph_overlay", "attention", "matching"):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ── Load both models ─────────────────────────────────────────────────
    cfg_gat  = load_config(args.config)
    cfg_base = load_config(args.baseline_config)

    print("Loading GAT model …")
    model_gat  = _load_model(cfg_gat,  args.checkpoint,          device)
    print("Loading baseline model …")
    model_base = _load_model(cfg_base, args.baseline_checkpoint, device)

    data_root  = args.data_root
    char_names = get_folder_names(data_root)

    samples_found = 0

    for char_name in char_names:
        if samples_found >= args.num_samples:
            break
        print(f"\nCharacter: {char_name}")

        # ── Load reference frame (1-shot) ─────────────────────────────
        ref_list    = make_single_data_list(data_root, char_name, ref_shot=1, is_ref=True)
        ref_dataset = DACoNSingleDataset(ref_list, data_root, is_ref=True, mode="val_kf")
        ref_loader  = DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=dacon_single_pad_collate_fn)
        ref_data = next(iter(ref_loader))

        n_ref       = int(ref_data["seg_num"][0])
        ref_img     = _tensor_to_rgb(ref_data["color_image"][0])   # (H,W,3)
        ref_seg_img = ref_data["seg_image"][0].numpy().astype(int)  # (H,W)
        ref_masks   = _seg_to_masks(ref_seg_img, n_ref)
        ref_colors  = ref_data["seg_colors"][0, :n_ref, :3].numpy()  # (n_ref,3)

        # Reference features (both models)
        feats_ref_gat  = _get_features(model_gat,  ref_data["line_image"],
                                        ref_data["seg_image"], ref_data["seg_num"], device)
        feats_ref_base = _get_features(model_base, ref_data["line_image"],
                                        ref_data["seg_image"], ref_data["seg_num"], device)

        # ── Iterate target frames ──────────────────────────────────────
        tgt_list    = make_single_data_list(data_root, char_name, ref_shot=1, is_ref=False)
        tgt_dataset = DACoNSingleDataset(tgt_list, data_root, is_ref=False, mode="val_kf")
        tgt_loader  = DataLoader(tgt_dataset, batch_size=1, shuffle=False, num_workers=0,
                                  collate_fn=dacon_single_pad_collate_fn)

        for tgt_data in tgt_loader:
            if samples_found >= args.num_samples:
                break

            n_tgt        = int(tgt_data["seg_num"][0])
            frame_name   = tgt_data["frame_name"][0]
            tgt_line_img = _tensor_to_rgb(tgt_data["line_image"][0])  # (H,W,3)
            tgt_seg_img  = tgt_data["seg_image"][0].numpy().astype(int)
            tgt_masks    = _seg_to_masks(tgt_seg_img, n_tgt)
            tgt_colors   = tgt_data["seg_colors"][0, :n_tgt, :3].numpy()

            # Similarity maps and predicted matches
            feats_tgt_gat, attn_data = _get_features_and_attention(
                model_gat,
                tgt_data["line_image"], tgt_data["seg_image"], tgt_data["seg_num"],
                device,
            )
            feats_tgt_base = _get_features(
                model_base,
                tgt_data["line_image"], tgt_data["seg_image"], tgt_data["seg_num"],
                device,
            )

            sim_gat  = _compute_sim_map(model_gat,  feats_ref_gat,  feats_tgt_gat,  n_ref, n_tgt)
            sim_base = _compute_sim_map(model_base, feats_ref_base, feats_tgt_base, n_ref, n_tgt)

            pred_gat  = sim_gat.argmax(axis=1)   # (n_tgt,)
            pred_base = sim_base.argmax(axis=1)  # (n_tgt,)
            gt_match  = _find_gt_matches(tgt_colors, ref_colors)  # (n_tgt,)

            # Segments where baseline is wrong and GAT is right
            interesting_segs = np.where(
                (pred_base != gt_match) & (pred_gat == gt_match)
            )[0]

            if len(interesting_segs) == 0:
                continue

            # Pick the segment with the largest baseline–GAT disagreement
            # (use the one with highest GAT confidence as a tie-breaker)
            gat_conf  = sim_gat[interesting_segs, pred_gat[interesting_segs]]
            query_seg = int(interesting_segs[gat_conf.argmax()])

            tag = f"{char_name}_f{frame_name}_seg{query_seg}"
            print(f"  Found interesting segment: {tag}")

            # ── Build adjacency for this target frame ───────────────────
            seg_img_t  = tgt_data["seg_image"][0].to(device).float()
            adj_tensor = build_adjacency(seg_img_t, n_tgt)
            adj_np     = adj_tensor.cpu().numpy()

            # ── Figure 1: Graph overlay ─────────────────────────────────
            vis_path = os.path.join(args.output_dir, "graph_overlay", tag)
            visualize_graph_overlay(
                tgt_line_img, tgt_masks, tgt_colors, adj_np, vis_path
            )

            # ── Figure 2: Attention weights ─────────────────────────────
            if attn_data is not None:
                for layer_idx, n_heads in enumerate([4, 1]):
                    for head_idx in range(n_heads):
                        attn_tag  = f"{tag}_l{layer_idx}_h{head_idx}"
                        attn_path = os.path.join(
                            args.output_dir, "attention", attn_tag
                        )
                        visualize_attention_weights(
                            tgt_line_img, tgt_masks, tgt_colors, adj_np,
                            attn_data["layer_alphas"],
                            query_seg, attn_path,
                            layer_idx=layer_idx, head_idx=head_idx,
                        )

            # ── Figure 3: Matching comparison ───────────────────────────
            match_path = os.path.join(args.output_dir, "matching", tag)
            visualize_matching_comparison(
                tgt_line_img, tgt_masks,
                ref_img,      ref_masks,
                query_seg,
                int(pred_base[query_seg]),
                int(pred_gat[query_seg]),
                int(gt_match[query_seg]),
                match_path,
            )

            samples_found += 1
            print(f"  Saved visualizations ({samples_found}/{args.num_samples})")

    print(f"\nDone. {samples_found} sample(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    from utils import load_config  # already imported above; kept for clarity

    parser = argparse.ArgumentParser(
        description="Generate GAT attention visualizations for paper figures."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the trained GAT-DACoN .pth file.",
    )
    parser.add_argument(
        "--baseline_checkpoint", required=True,
        help="Path to the baseline (no-GAT) .pth file.",
    )
    parser.add_argument(
        "--config", default="configs/test_gat_aug.yaml",
        help="Test config for the GAT model.",
    )
    parser.add_argument(
        "--baseline_config", default="configs/test_aug_only.yaml",
        help="Test config for the baseline model.",
    )
    parser.add_argument(
        "--data_root",
        default="dataset/PaintBucket_Char_v2/test/PaintBucket_Char_v2/",
        help="Path to the test dataset root.",
    )
    parser.add_argument(
        "--output_dir", default="visualizations/",
        help="Directory to save all figures.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=8,
        help="Number of interesting segments to visualise.",
    )
    args = parser.parse_args()
    main(args)
