#!/usr/bin/env python
"""
Fit a K-component Gaussian Mixture Model to ground truth Car H3D values
and output bin boundaries at the 50/50 crossover points between adjacent
Gaussians.

Usage (from repo root):
    python tools/find_h3d_bins.py --config configs/monodgp.yaml --k 3 --split train
    python tools/find_h3d_bins.py --config configs/monodgp.yaml --k 5 --split train --clip_percentiles 1 99
"""

import argparse
import os
import sys

import numpy as np
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from lib.datasets.kitti.kitti_utils import get_objects_from_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/monodgp.yaml")
    parser.add_argument(
        "--k", type=int, required=True, help="Number of Gaussian components"
    )
    parser.add_argument(
        "--split", default="train", help="Which split to use (train/val)"
    )
    parser.add_argument(
        "--clip_percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Clip outliers by percentiles before GMM fit, e.g. --clip_percentiles 1 99",
    )
    return parser.parse_args()


def find_crossover(mu1, std1, w1, mu2, std2, w2):
    """Find x where w1*N(x|mu1,std1) == w2*N(x|mu2,std2), between the means."""
    lo, hi = min(mu1, mu2), max(mu1, mu2)
    xs = np.linspace(lo, hi, 10000)
    diff = w1 * norm.pdf(xs, mu1, std1) - w2 * norm.pdf(xs, mu2, std2)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return (mu1 + mu2) / 2
    return xs[sign_changes[-1]]


def main():
    args = parse_args()

    cfg_path = (
        os.path.join(ROOT_DIR, args.config)
        if not os.path.isabs(args.config)
        else args.config
    )
    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)

    root_dir = cfg["dataset"]["root_dir"]
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(ROOT_DIR, root_dir)

    split_file = os.path.join(root_dir, "ImageSets", args.split + ".txt")
    label_dir = os.path.join(root_dir, "training", "label_2")
    idx_list = [x.strip() for x in open(split_file).readlines()]

    gt_vals = []
    for idx in idx_list:
        for obj in get_objects_from_label(os.path.join(label_dir, idx + ".txt")):
            if obj.cls_type == "Car":
                gt_vals.append(obj.h)

    gt_raw = np.array(gt_vals, dtype=np.float32)
    print(f"Loaded {len(gt_raw)} Car H3D values from {args.split} split")

    # Remove non-finite values first
    gt_raw = gt_raw[np.isfinite(gt_raw)]

    # Clip outliers by percentile before GMM fitting
    p_low, p_high = args.clip_percentiles
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(f"Invalid clip percentiles: {args.clip_percentiles}")

    clip_lo, clip_hi = np.percentile(gt_raw, [p_low, p_high])
    gt = gt_raw[(gt_raw >= clip_lo) & (gt_raw <= clip_hi)]

    print(
        f"Clipped to percentiles [{p_low}, {p_high}] -> range [{clip_lo:.3f}, {clip_hi:.3f}]"
    )
    print(
        f"Kept {len(gt)} / {len(gt_raw)} values ({100.0 * len(gt) / len(gt_raw):.2f}%)"
    )

    if len(gt) < args.k:
        raise ValueError(
            f"Not enough samples after clipping ({len(gt)}) for k={args.k}"
        )

    # Fit GMM on clipped values
    gmm = GaussianMixture(
        n_components=args.k, covariance_type="full", n_init=20, random_state=42
    )
    gmm.fit(gt.reshape(-1, 1))

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    print(f"\nGMM Components (K={args.k}):")
    print(f"  {'#':>2}  {'Weight':>8}  {'Mean (m)':>9}  {'Std (m)':>8}")
    for i in range(args.k):
        print(f"  {i + 1:2d}  {weights[i]:8.3f}  {means[i]:9.3f}  {stds[i]:8.3f}")

    # Bin boundaries at 50/50 crossover points
    boundaries = []
    for i in range(args.k - 1):
        cx = find_crossover(
            means[i], stds[i], weights[i], means[i + 1], stds[i + 1], weights[i + 1]
        )
        boundaries.append(cx)

    bin_edges = [float("-inf")] + boundaries + [float("inf")]
    print(f"\nBin boundaries: {[f'{b:.3f}' for b in boundaries]}")
    print("\nBins (evaluated on clipped data):")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (gt >= lo) & (gt < hi)
        n = mask.sum()
        if n > 0:
            print(
                f"  Bin {i + 1}: [{lo:>8.3f}, {hi:>8.3f})  "
                f"n={n:>5}  mean={gt[mask].mean():.3f}  std={gt[mask].std():.3f}"
            )
        else:
            print(f"  Bin {i + 1}: [{lo:>8.3f}, {hi:>8.3f})  n={n:>5}")

    # ---- Plot ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    bins_hist = np.linspace(gt.min() - 0.1, gt.max() + 0.1, 60)
    ax.hist(
        gt,
        bins=bins_hist,
        density=True,
        alpha=0.4,
        color="#4080D0",
        edgecolor="white",
        linewidth=0.5,
        label=f"Clipped GT (n={len(gt)})",
    )

    x = np.linspace(gt.min() - 0.3, gt.max() + 0.3, 500)
    overall = np.zeros_like(x)
    cmap = plt.cm.Set1
    for i in range(args.k):
        yi = weights[i] * norm.pdf(x, means[i], stds[i])
        overall += yi
        ax.plot(
            x,
            yi,
            color=cmap(i),
            linewidth=1.5,
            linestyle="--",
            label=f"K{i + 1}: μ={means[i]:.2f}, σ={stds[i]:.2f}, w={weights[i]:.2f}",
        )
        ax.fill_between(x, yi, alpha=0.07, color=cmap(i))

    ax.plot(x, overall, color="black", linewidth=2, label="GMM fit")

    for b in boundaries:
        ax.axvline(b, color="#D04040", linewidth=1.5, linestyle=":")
        ax.text(
            b,
            ax.get_ylim()[1] * 0.95,
            f"{b:.3f}m",
            ha="center",
            va="top",
            fontsize=8,
            color="#D04040",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    ax.set_xlabel("H3D — 3D height (m)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Car H3D: GMM with K={args.k} components ({args.split}, clipped)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out_path = os.path.join(ROOT_DIR, f"h3d_bins_k{args.k}_{args.split}_clipped.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
