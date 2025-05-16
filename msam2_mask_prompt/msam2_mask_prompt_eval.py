#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
msam2_mask_prompt_eval.py
---------------------
Evaluate a MedSAM-2 3-D segmentation produced from a user-defined ROI prompt.

Metrics
~~~~~~~
* Dice similarity coefficient
* Mean absolute HU error inside the predicted mask
* HU histogram-overlap score (intersection / union of 256-bin PDFs)

Visual output
~~~~~~~~~~~~~
A 3×3 grid:
    row-1 → 25-th-percentile lesion slice
    row-2 → key slice (max GT area)
    row-3 → 95-th-percentile lesion slice
    col-1 → plain CT
    col-2 → CT + GT mask overlay
    col-3 → CT + predicted mask overlay
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial.distance import jensenshannon


# ------------------------------------------------------------------------- #
# Helpers                                                                   #
# ------------------------------------------------------------------------- #
def load_nii(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img


def fuse_labels(mask: np.ndarray) -> np.ndarray:
    """Return binary mask where any non-zero label becomes 1."""
    return (mask > 0).astype(np.uint8)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 2.0 * inter / denom if denom else 1.0


def mean_abs_hu_error(ct: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Return | mean_HU(pred) - mean_HU(gt) |.
    Safer than voxel-wise subtraction because the masks can differ in size.
    """
    hu_gt   = ct[gt   > 0]
    hu_pred = ct[pred > 0]
    if hu_gt.size == 0 or hu_pred.size == 0:
        return 0.0
    return float(abs(hu_gt.mean() - hu_pred.mean()))


def hu_hist_overlap(ct: np.ndarray, gt: np.ndarray, pred: np.ndarray, bins=256) -> float:
    """Histogram intersection / union inside HU range."""
    hu_vals_gt   = ct[gt > 0].flatten()
    hu_vals_pred = ct[pred > 0].flatten()
    if hu_vals_gt.size == 0 or hu_vals_pred.size == 0:
        return 0.0
    h_gt, _   = np.histogram(hu_vals_gt, bins=bins, range=(-1024, 3071), density=True)
    h_pred, _ = np.histogram(hu_vals_pred, bins=bins, range=(-1024, 3071), density=True)
    # Intersection / Union ≈ 1 - JS divergence
    return float(1.0 - jensenshannon(h_gt, h_pred) ** 2)


def pick_axial_slices(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return (p25, key, p95) *axial* slice indices.
    
    Works regardless of whether the array is stored (Z, Y, X) or (X, Y, Z):
    we look for the axis with the largest number of non-empty slices first.
    """
    # find which axis contains the mask most sparsely populated with zeros → that is Z
    nz_per_axis = [np.count_nonzero(mask.sum(axis=i)) for i in range(3)]
    z_axis = int(np.argmax(nz_per_axis))      # axis with the fullest coverage
    
    # sum over (Y,X) to get voxel count per Z
    counts = mask.sum(axis=tuple(j for j in range(3) if j != z_axis))
    
    if counts.max() == 0:                    # completely empty
        mid = mask.shape[z_axis] // 2
        return mid, mid, mid, z_axis
    
    z_idx = np.arange(mask.shape[z_axis])
    cum  = np.cumsum(counts) / counts.sum()
    p25  = int(z_idx[np.searchsorted(cum, 0.25)])
    key  = int(z_idx[counts.argmax()])
    p95  = int(z_idx[np.searchsorted(cum, 0.95, side="right") - 1])
    
    return p25, key, p95, z_axis


def get_slice(vol, z, axis=2):
    """Helper to fetch the right 2-D plane"""
    return np.take(vol, z, axis=axis)


def overlay_slice(ax, ct_slice, mask, color):
    """Display CT slice with segmentation mask overlay."""
    ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=500)
    if mask is not None and np.any(mask):
        # Create a colored mask with transparency
        masked_data = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked_data, cmap=color, alpha=0.6, vmin=0, vmax=1, interpolation="none")
    ax.axis("off")


# ------------------------------------------------------------------------- #
# Main eval workflow                                                        #
# ------------------------------------------------------------------------- #
def evaluate(ct_path: Path, gt_path: Path, pred_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ct,   ct_img   = load_nii(ct_path)
    gt,   _        = load_nii(gt_path)
    pred, _        = load_nii(pred_path)

    # Ensure binary masks
    gt_bin   = fuse_labels(gt)
    pred_bin = fuse_labels(pred)
    
    # Debug - print mask info
    print(f"GT mask shape: {gt_bin.shape}, non-zero voxels: {np.count_nonzero(gt_bin)}")
    print(f"Pred mask shape: {pred_bin.shape}, non-zero voxels: {np.count_nonzero(pred_bin)}")
    
    # Combine gt and pred masks to find regions of interest
    combined_mask = np.logical_or(gt_bin > 0, pred_bin > 0).astype(np.uint8)
    print(f"Combined mask has {np.count_nonzero(combined_mask)} non-zero voxels")

    metrics = {
        "dice":           dice(gt_bin, pred_bin),
        "mean_abs_HUerr": mean_abs_hu_error(ct, gt_bin, pred_bin),
        "HU_hist_ovl":    hu_hist_overlap(ct, gt_bin, pred_bin),
    }

    # ---- figure ----------------------------------------------------------------
    # Find valid slices manually - check both GT and prediction
    counts_by_slice = []
    for axis in range(3):
        counts = []
        for i in range(combined_mask.shape[axis]):
            slice_idx = [slice(None)] * 3
            slice_idx[axis] = i
            counts.append(np.count_nonzero(combined_mask[tuple(slice_idx)]))
        counts_by_slice.append(counts)
    
    # Identify best axis and slices - use axis with most populated slices
    best_axis = np.argmax([max(counts) if counts else 0 for counts in counts_by_slice])
    best_axis_counts = counts_by_slice[best_axis]
    print(f"Best axis for visualization: {best_axis}")
    
    # Find key slice and percentile slices
    if max(best_axis_counts) == 0:
        # No data found, use middle slices
        mid = combined_mask.shape[best_axis] // 2
        p25, key, p95 = mid, mid, mid
    else:
        # Get slices based on voxel distribution
        nonzero_indices = [i for i, count in enumerate(best_axis_counts) if count > 0]
        counts = [best_axis_counts[i] for i in nonzero_indices]
        
        # Find key slice (maximum area)
        key_idx = np.argmax(counts)
        key = nonzero_indices[key_idx]
        
        # Calculate percentiles if we have enough slices
        if len(nonzero_indices) >= 3:
            p25_idx = len(nonzero_indices) // 4
            p95_idx = min(int(len(nonzero_indices) * 0.95), len(nonzero_indices) - 1)
            p25 = nonzero_indices[p25_idx]
            p95 = nonzero_indices[p95_idx]
        else:
            p25, p95 = key, key
            
    print(f"Selected slices - p25: {p25}, key: {key}, p95: {p95}")
    slices = [p25, key, p95]
    z_ax = best_axis
    
    # Print debugging info about selected slices
    for i, z in enumerate(slices):
        gt_slice = get_slice(gt_bin, z, axis=z_ax)
        pred_slice = get_slice(pred_bin, z, axis=z_ax)
        print(f"Slice {i} (index {z}): GT non-zero pixels: {np.count_nonzero(gt_slice)}, Pred non-zero pixels: {np.count_nonzero(pred_slice)}")
    
    # Set up colormaps for better visibility
    gt_cmap = plt.cm.Reds  # Use matplotlib colormap objects
    pred_cmap = plt.cm.Greens
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    for r, z in enumerate(slices):
        ct_slice = get_slice(ct, z, axis=z_ax)
        gt_slice = get_slice(gt_bin, z, axis=z_ax)
        pred_slice = get_slice(pred_bin, z, axis=z_ax)
        
        overlay_slice(axes[r, 0], ct_slice, None, None)  # plain CT
        overlay_slice(axes[r, 1], ct_slice, gt_slice, gt_cmap)  # GT
        overlay_slice(axes[r, 2], ct_slice, pred_slice, pred_cmap)  # pred
        axes[r, 0].set_ylabel(f"z={z} {'(empty)' if np.count_nonzero(gt_slice) == 0 else ''}", rotation=90, va="center")

    axes[0, 0].set_title("CT")
    axes[0, 1].set_title("GT overlay")
    axes[0, 2].set_title("Pred overlay")
    fig.tight_layout()
    fig.savefig(out_dir / "evaluation.png", dpi=200)
    plt.close(fig)

    # ---- write metrics ---------------------------------------------------------
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✓ Evaluation complete →", out_dir)
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}")


# ------------------------------------------------------------------------- #
# CLI                                                                       #
# ------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate MedSAM-2 3-D segmentation from ROI prompt")
    p.add_argument("--ct",   required=True, type=Path, help="Input CT volume (.nii/.nii.gz)")
    p.add_argument("--gt",   required=True, type=Path, help="Ground-truth mask")
    p.add_argument("--pred", required=True, type=Path, help="Predicted mask")
    p.add_argument("--out",  required=True, type=Path, help="Output folder for png + json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.ct, args.gt, args.pred, args.out)
