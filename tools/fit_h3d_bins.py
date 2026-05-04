#!/usr/bin/env python
"""
Fit height bins from KITTI train split using a Gaussian Mixture Model.

Bin boundaries are placed at the crossover points between adjacent GMM components.

Usage:
    python tools/fit_h3d_bins.py --config configs/monodgp.yaml --k 5
"""

import argparse
import os
import sys

import numpy as np
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from lib.datasets.kitti.kitti_utils import get_objects_from_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/monodgp.yaml")
    parser.add_argument("--k", type=int, required=True, help="Number of GMM components")
    parser.add_argument(
        "--split", default="train", help="KITTI split to use (default: train)"
    )
    parser.add_argument("--output", default="h3d_bins.pdf")
    return parser.parse_args()


def load_kitti_heights(root_dir, split):
    split_file = os.path.join(root_dir, "ImageSets", split + ".txt")
    label_dir = os.path.join(root_dir, "training", "label_2")
    assert os.path.exists(split_file), f"Split file not found: {split_file}"
    assert os.path.exists(label_dir), f"Label dir not found: {label_dir}"
    heights = []
    for idx in [x.strip() for x in open(split_file).readlines()]:
        for obj in get_objects_from_label(os.path.join(label_dir, idx + ".txt")):
            if obj.cls_type == "Car":
                heights.append(obj.h)
    return np.array(heights)


def find_crossover(mu1, std1, w1, mu2, std2, w2):
    lo, hi = min(mu1, mu2), max(mu1, mu2)
    xs = np.linspace(lo, hi, 10000)
    diff = w1 * norm.pdf(xs, mu1, std1) - w2 * norm.pdf(xs, mu2, std2)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    return xs[sign_changes[-1]] if len(sign_changes) else (mu1 + mu2) / 2


def fit_gmm(heights, k):
    gmm = GaussianMixture(
        n_components=k, covariance_type="full", n_init=20, random_state=42
    )
    gmm.fit(heights.reshape(-1, 1))
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    order = np.argsort(means)
    return means[order], stds[order], weights[order]


def main():
    args = parse_args()

    cfg_path = (
        os.path.join(ROOT_DIR, args.config)
        if not os.path.isabs(args.config)
        else args.config
    )
    cfg = yaml.load(open(cfg_path), Loader=yaml.Loader)
    root_dir = cfg["dataset"]["root_dir"]
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(ROOT_DIR, root_dir)

    print(f"Loading KITTI {args.split} split ...")
    heights = load_kitti_heights(root_dir, args.split)
    outliers = (heights > 2).sum()
    heights = heights[heights <= 2]
    print(
        f"  {len(heights)} cars  mean={heights.mean():.3f}  std={heights.std():.3f}"
        f"  (removed {outliers} outliers >2m)"
    )

    means, stds, weights = fit_gmm(heights, args.k)

    print(f"\nGMM components (k={args.k}):")
    print(f"  {'#':>2}  {'Weight':>8}  {'Mean (m)':>9}  {'Std (m)':>8}")
    for i in range(args.k):
        print(f"  {i + 1:2d}  {weights[i]:8.3f}  {means[i]:9.3f}  {stds[i]:8.3f}")

    boundaries = [
        find_crossover(
            means[i], stds[i], weights[i], means[i + 1], stds[i + 1], weights[i + 1]
        )
        for i in range(args.k - 1)
    ]

    bin_edges = [float("-inf")] + boundaries + [float("inf")]

    print(f"\nBin boundaries: {[f'{b:.4f}' for b in boundaries]}")
    print("\nBins:")
    print(f"  {'Bin':>4}  {'Range':>22}  {'N':>6}  {'Mean':>8}  {'Std':>8}")
    centers = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (heights >= lo) & (heights < hi)
        n = mask.sum()
        m = heights[mask].mean()
        s = heights[mask].std()
        centers.append(m)
        lo_s = f"{lo:.4f}" if lo != float("-inf") else "    -inf"
        hi_s = f"{hi:.4f}" if hi != float("inf") else "    +inf"
        print(f"  {i:>4}  [{lo_s}, {hi_s})  {n:>6}  {m:>8.4f}  {s:>8.4f}")

    print("\nPaste these into lib/models/monodgp/height_bins.py:")
    print(
        f"HEIGHT_BIN_EDGES   = np.array([{', '.join(f'{b:.4f}' for b in boundaries)}])"
    )
    print(f"HEIGHT_BIN_CENTERS = np.array([{', '.join(f'{c:.4f}' for c in centers)}])")

    # ── Plot ──────────────────────────────────────────────────────────────
    # Okabe-Ito colorblind-safe palette
    cb_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
                 "#D55E00", "#CC79A7", "#000000"]
    x = np.linspace(heights.min() - 0.1, heights.max() + 0.1, 500)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bins_hist = np.linspace(heights.min() - 0.05, heights.max() + 0.05, 60)
    ax.hist(
        heights,
        bins=bins_hist,
        density=True,
        alpha=0.35,
        color="#56B4E9",
        edgecolor="none",
        label=f"KITTI {args.split} (n={len(heights)})",
    )

    overall = np.zeros_like(x)
    for i in range(args.k):
        yi = weights[i] * norm.pdf(x, means[i], stds[i])
        overall += yi
        ax.plot(
            x,
            yi,
            color=cb_colors[i % len(cb_colors)],
            linewidth=1.5,
            linestyle="--",
            label=f"K{i + 1}: μ={means[i]:.2f}, w={weights[i]:.2f}",
        )
        ax.fill_between(x, yi, alpha=0.07, color=cb_colors[i % len(cb_colors)])
    ax.plot(x, overall, color="#000000", linewidth=2, label="GMM fit")

    for i, b in enumerate(boundaries):
        ax.axvline(b, color="#000000", linewidth=1.5, linestyle="-",
                   label="Bin boundaries" if i == 0 else None)

    for i, mu in enumerate(means):
        ax.axvline(mu, color="#D55E00", linewidth=0.8, linestyle=":",
                   label="GMM means" if i == 0 else None)

    ax.set_xlabel("H3D — 3D height (m)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Car H3D: GMM k={args.k}  (KITTI {args.split})", fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(ROOT_DIR, args.output)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
