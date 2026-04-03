"""
Sketch robustness augmentation for DACoN training.

Applied to target line images only (not reference images, not at eval time).
Simulates common hand-drawn imperfections so the model generalises from
3D-rendered training data to real hand-drawn test data.

Sub-augmentations (each applied independently with its own probability):
    2a. Line gap simulation   – randomly removes 5–15 % of line pixels.
    2b. Line thickness variation – patch-wise morphological erosion/dilation.
    2c. Positional jitter      – smooth elastic deformation.
    2d. Line colour variation  – converts coloured line pixels to black.

The full pipeline fires with probability `apply_prob` (default 0.5);
if it fires, each sub-augmentation is then tried independently.
"""

import random

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a normalised 1-D Gaussian kernel tensor."""
    radius = max(1, int(3.0 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=dtype, device=device) - radius
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _elastic_deform(image: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
    """Apply smooth elastic deformation to a single (C, H, W) tensor.

    A random displacement field is generated and smoothed with a Gaussian
    filter, then used via bilinear grid_sample.

    Args:
        image: (C, H, W) float tensor in [0, 1].
        alpha: Maximum displacement magnitude in pixels.
        sigma: Gaussian smoothing standard deviation; larger → smoother.

    Returns:
        Deformed (C, H, W) tensor.
    """
    C, H, W = image.shape
    device  = image.device
    dtype   = image.dtype

    # Random displacement fields: (1, 1, H, W)
    dx = torch.randn(1, 1, H, W, device=device, dtype=dtype) * alpha
    dy = torch.randn(1, 1, H, W, device=device, dtype=dtype) * alpha

    # Smooth displacement fields with separable Gaussian
    kernel = _gaussian_kernel_1d(sigma, device, dtype)
    k_h = kernel.view(1, 1, -1, 1)
    k_w = kernel.view(1, 1, 1, -1)
    pad = len(kernel) // 2

    dx = F.conv2d(F.conv2d(dx, k_h, padding=(pad, 0)), k_w, padding=(0, pad))
    dy = F.conv2d(F.conv2d(dy, k_h, padding=(pad, 0)), k_w, padding=(0, pad))

    # Normalise displacements from pixels to grid-sample coordinates ([-1, 1])
    dx = dx.squeeze() / (W / 2.0)
    dy = dy.squeeze() / (H / 2.0)

    # Base regular sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )

    flow = torch.stack([grid_x + dx, grid_y + dy], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    deformed = F.grid_sample(
        image.unsqueeze(0), flow,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return deformed.squeeze(0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SketchAugmentation:
    """Stochastic pipeline of line-art imperfection augmentations.

    Usage in the training loop::

        augmenter = SketchAugmentation()
        data['line_images_tgt'] = augmenter.augment_batch(data['line_images_tgt'])

    Args:
        apply_prob:   Probability of applying *any* augmentation to a given image.
        gap_prob:     Probability of line gap simulation (Contribution 2a).
        thick_prob:   Probability of thickness variation (Contribution 2b).
        jitter_prob:  Probability of elastic jitter (Contribution 2c).
        color_prob:   Probability of colour variation (Contribution 2d).
        line_thresh:  Pixel value threshold below which a pixel is treated as a
                      line pixel (checked per RGB channel). Default 0.3.
    """

    def __init__(
        self,
        apply_prob:  float = 0.5,
        gap_prob:    float = 0.3,
        thick_prob:  float = 0.3,
        jitter_prob: float = 0.2,
        color_prob:  float = 0.2,
        line_thresh: float = 0.3,
    ) -> None:
        self.apply_prob  = apply_prob
        self.gap_prob    = gap_prob
        self.thick_prob  = thick_prob
        self.jitter_prob = jitter_prob
        self.color_prob  = color_prob
        self.line_thresh = line_thresh

    # ------------------------------------------------------------------
    # 2a: Line Gap Simulation
    # ------------------------------------------------------------------
    def _line_gap(self, image: torch.Tensor) -> torch.Tensor:
        """Remove 5–15 % of line pixels to simulate incomplete line closure.

        The segment masks are intentionally left unchanged so the model must
        learn to handle sketches that do not perfectly agree with the segments.
        """
        is_line = (image[:3] < self.line_thresh).all(dim=0)   # (H, W)
        coords  = is_line.nonzero(as_tuple=False)              # (N, 2)
        if len(coords) == 0:
            return image

        frac       = random.uniform(0.05, 0.15)
        num_remove = int(frac * len(coords))
        if num_remove == 0:
            return image

        perm   = torch.randperm(len(coords), device=image.device)[:num_remove]
        remove = coords[perm]
        out    = image.clone()
        out[:, remove[:, 0], remove[:, 1]] = 1.0   # set removed pixels to white
        return out

    # ------------------------------------------------------------------
    # 2b: Line Thickness Variation
    # ------------------------------------------------------------------
    def _thickness_variation(self, image: torch.Tensor) -> torch.Tensor:
        """Patch-wise erosion/dilation simulating variable pen pressure.

        The image is divided into a 4×4 grid.  Each cell is independently
        subjected to erosion (lines get thinner) or dilation (lines get thicker)
        using a small morphological kernel.

        For a white-background image where lines are dark (≈ 0):
            - Dilation of lines  = min-pool of pixel values.
            - Erosion  of lines  = max-pool of pixel values.
        """
        C, H, W = image.shape
        cell_H  = max(1, H // 4)
        cell_W  = max(1, W // 4)
        out     = image.clone()

        for row in range(4):
            for col in range(4):
                if random.random() > 0.5:
                    continue   # skip this cell

                y0 = row * cell_H
                y1 = min(y0 + cell_H, H)
                x0 = col * cell_W
                x1 = min(x0 + cell_W, W)

                patch = out[:, y0:y1, x0:x1].unsqueeze(0)   # (1, C, h, w)
                k = random.choice([3, 5])   # odd kernel → stride=1 preserves spatial size
                p = k // 2

                if random.random() < 0.5:
                    # Dilate lines (thicken): min-pool pixel values
                    processed = -F.max_pool2d(-patch, kernel_size=k, stride=1, padding=p)
                else:
                    # Erode lines (thin): max-pool pixel values
                    processed = F.max_pool2d(patch, kernel_size=k, stride=1, padding=p)

                # Clamp to valid range after pooling
                out[:, y0:y1, x0:x1] = processed.squeeze(0).clamp(0.0, 1.0)

        return out

    # ------------------------------------------------------------------
    # 2c: Positional Jitter
    # ------------------------------------------------------------------
    def _positional_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Elastic deformation simulating spatial inaccuracy of hand drawing."""
        alpha = random.uniform(10.0, 30.0)
        sigma = random.uniform(3.0, 5.0)
        return _elastic_deform(image, alpha, sigma)

    # ------------------------------------------------------------------
    # 2d: Partial Line Colour Variation
    # ------------------------------------------------------------------
    def _line_color_variation(self, image: torch.Tensor) -> torch.Tensor:
        """Convert coloured line pixels to black, simulating monochrome sketches.

        DACoN uses coloured lines to distinguish segments (e.g. highlight/shadow
        boundaries drawn in a different hue).  This augmentation trains the model
        to handle sketches where such colour coding is absent.
        """
        rgb = image[:3]

        # Coloured line pixels: dark overall yet with inter-channel variation
        is_dark         = (rgb < self.line_thresh).all(dim=0)  # (H, W)
        std_per_pixel   = rgb.std(dim=0)                        # (H, W)
        is_colored_line = is_dark & (std_per_pixel > 0.05)

        if not is_colored_line.any():
            return image

        out = image.clone()
        out[:3, is_colored_line] = 0.0   # convert coloured lines to black
        return out

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Augment a single line drawing with probability `apply_prob`.

        Args:
            image: (C, H, W) float tensor in [0, 1].

        Returns:
            Augmented (C, H, W) tensor; original tensor if not triggered.
        """
        if random.random() > self.apply_prob:
            return image

        if random.random() < self.gap_prob:
            image = self._line_gap(image)
        if random.random() < self.thick_prob:
            image = self._thickness_variation(image)
        if random.random() < self.jitter_prob:
            image = self._positional_jitter(image)
        if random.random() < self.color_prob:
            image = self._line_color_variation(image)

        return image

    def augment_batch(self, line_images: torch.Tensor) -> torch.Tensor:
        """Augment a batch tensor of line images in-place.

        Each (batch, frame) image is augmented independently.

        Args:
            line_images: (B, S, C, H, W) float tensor.

        Returns:
            Augmented (B, S, C, H, W) tensor.
        """
        B, S, C, H, W = line_images.shape
        out = line_images.clone()
        for b in range(B):
            for s in range(S):
                out[b, s] = self(line_images[b, s])
        return out
