#!/usr/bin/env python
"""
Sweep k=3..6 GMM fits on KITTI Car H3D heights and select best k by BIC.

Outputs
-------
  gmm_k_search.pdf   – subplot per k, plus BIC/AIC summary panel
  gmm_k_search.txt   – full text report with bin edges/centers for each k

Usage
-----
    python tools/gmm_k_search.py --config configs/monodgp.yaml
    python tools/gmm_k_search.py --config configs/monodgp.yaml --cls Pedestrian
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
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/monodgp.yaml")
    p.add_argument("--label_dir", default=None,
                   help="Direct path to label_2 dir (overrides config root_dir)")
    p.add_argument("--cls", default="Car", help="KITTI class to analyse (Car, Pedestrian, Cyclist)")
    p.add_argument("--k_min", type=int, default=3)
    p.add_argument("--k_max", type=int, default=6)
    p.add_argument("--clip_percentiles", type=float, nargs=2, default=(1.0, 99.0))
    p.add_argument("--n_init", type=int, default=30, help="GMM restarts for stability")
    p.add_argument("--out_dir", default=".")
    return p.parse_args()


def load_heights(label_dir, cls_type):
    heights = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue
        for obj in get_objects_from_label(os.path.join(label_dir, fname)):
            if obj.cls_type == cls_type:
                heights.append(obj.h)
    return np.array(heights, dtype=np.float64)


def find_crossover(mu1, s1, w1, mu2, s2, w2):
    lo, hi = min(mu1, mu2), max(mu1, mu2)
    xs   = np.linspace(lo, hi, 10_000)
    diff = w1 * norm.pdf(xs, mu1, s1) - w2 * norm.pdf(xs, mu2, s2)
    idx  = np.where(np.diff(np.sign(diff)))[0]
    return xs[idx[-1]] if len(idx) else (mu1 + mu2) / 2


def fit_gmm(data, k, n_init):
    gmm = GaussianMixture(
        n_components=k, covariance_type="full",
        n_init=n_init, random_state=42, max_iter=300
    )
    gmm.fit(data.reshape(-1, 1))
    means   = gmm.means_.flatten()
    stds    = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    order   = np.argsort(means)
    return gmm, means[order], stds[order], weights[order]


def bin_stats(data, boundaries):
    edges = [float("-inf")] + list(boundaries) + [float("inf")]
    centers, counts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (data >= lo) & (data < hi)
        centers.append(data[mask].mean() if mask.sum() else (lo + hi) / 2)
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(counts)


CB = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
      "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def plot_single(ax, data, k, means, stds, weights, boundaries, bic, aic):
    x = np.linspace(data.min() - 0.1, data.max() + 0.1, 600)
    ax.hist(data, bins=60, density=True, alpha=0.30,
            color="#56B4E9", edgecolor="none", label=f"n={len(data)}")
    overall = np.zeros_like(x)
    for i in range(k):
        yi = weights[i] * norm.pdf(x, means[i], stds[i])
        overall += yi
        ax.plot(x, yi, color=CB[i % len(CB)], lw=1.4, ls="--",
                label=f"K{i+1}: μ={means[i]:.3f}, w={weights[i]:.2f}")
        ax.fill_between(x, yi, alpha=0.07, color=CB[i % len(CB)])
    ax.plot(x, overall, color="black", lw=2, label="GMM")
    for b in boundaries:
        ax.axvline(b, color="#D04040", lw=1.2, ls=":")
    ax.set_title(f"k={k}  BIC={bic:.1f}  AIC={aic:.1f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("H3D (m)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def main():
    args = parse_args()

    if args.label_dir:
        label_dir = args.label_dir
    else:
        cfg_path = (os.path.join(ROOT_DIR, args.config)
                    if not os.path.isabs(args.config) else args.config)
        cfg      = yaml.load(open(cfg_path), Loader=yaml.Loader)
        root_dir = cfg["dataset"]["root_dir"]
        if not os.path.isabs(root_dir):
            root_dir = os.path.join(ROOT_DIR, root_dir)
        label_dir = os.path.join(root_dir, "training", "label_2")

    print(f"Loading {args.cls} heights from {label_dir} ...")
    heights_raw = load_heights(label_dir, args.cls)
    heights_raw = heights_raw[np.isfinite(heights_raw)]
    lo_p, hi_p  = np.percentile(heights_raw, args.clip_percentiles)
    heights     = heights_raw[(heights_raw >= lo_p) & (heights_raw <= hi_p)]
    print(f"  raw={len(heights_raw)}  after clip=[{lo_p:.3f},{hi_p:.3f}]: {len(heights)}")
    print(f"  mean={heights.mean():.3f}  std={heights.std():.3f}\n")

    ks   = list(range(args.k_min, args.k_max + 1))
    results = {}

    lines = [
        f"GMM k-sweep on KITTI {args.cls} heights",
        f"label_dir: {label_dir}",
        f"clip percentiles: {args.clip_percentiles}  n={len(heights)}",
        "=" * 70,
    ]

    for k in ks:
        gmm, means, stds, weights = fit_gmm(heights, k, args.n_init)
        bic = gmm.bic(heights.reshape(-1, 1))
        aic = gmm.aic(heights.reshape(-1, 1))
        ll  = gmm.score(heights.reshape(-1, 1)) * len(heights)  # total log-likelihood

        boundaries     = [find_crossover(means[i], stds[i], weights[i],
                                         means[i+1], stds[i+1], weights[i+1])
                          for i in range(k - 1)]
        centers, counts = bin_stats(heights, boundaries)

        results[k] = dict(gmm=gmm, means=means, stds=stds, weights=weights,
                          boundaries=boundaries, centers=centers, counts=counts,
                          bic=bic, aic=aic, ll=ll)

        block = [
            f"\n--- k={k} ---",
            f"  BIC={bic:.2f}  AIC={aic:.2f}  log-likelihood={ll:.2f}",
            f"  Components:  {'#':>2}  {'Weight':>7}  {'Mean':>8}  {'Std':>8}",
        ]
        for i in range(k):
            block.append(f"               {i+1:2d}  {weights[i]:7.3f}  {means[i]:8.4f}  {stds[i]:8.4f}")
        block.append(f"  Bin edges  : {[f'{b:.4f}' for b in boundaries]}")
        block.append(f"  Bin centers: {[f'{c:.4f}' for c in centers]}")
        block.append(f"  Bin counts : {list(counts)}")
        block.append("")
        block.append("  Paste into lib/models/monodgp/height_bins.py:")
        block.append(f"    NUM_HEIGHT_BINS    = {k}")
        block.append(f"    HEIGHT_BIN_EDGES   = np.array([{', '.join(f'{b:.4f}' for b in boundaries)}])")
        block.append(f"    HEIGHT_BIN_CENTERS = np.array([{', '.join(f'{c:.4f}' for c in centers)}])")
        lines.extend(block)

        print("\n".join(block))

    # ── Best k by BIC ──────────────────────────────────────────────────────
    best_k_bic = min(ks, key=lambda k: results[k]["bic"])
    best_k_aic = min(ks, key=lambda k: results[k]["aic"])
    summary = [
        "\n" + "=" * 70,
        "SUMMARY",
        f"  {'k':>4}  {'BIC':>12}  {'AIC':>12}  {'log-lik':>12}",
    ]
    for k in ks:
        r   = results[k]
        tag = " <-- best BIC" if k == best_k_bic else ("" if k != best_k_aic else " <-- best AIC")
        summary.append(f"  {k:4d}  {r['bic']:12.2f}  {r['aic']:12.2f}  {r['ll']:12.2f}{tag}")
    summary += [
        f"\nBest k by BIC : {best_k_bic}",
        f"Best k by AIC : {best_k_aic}",
    ]
    lines.extend(summary)
    print("\n".join(summary))

    # ── Plot ───────────────────────────────────────────────────────────────
    n_k      = len(ks)
    fig, axes = plt.subplots(2, n_k, figsize=(4.5 * n_k, 9),
                              gridspec_kw={"height_ratios": [3, 1]})

    for col, k in enumerate(ks):
        r = results[k]
        plot_single(axes[0, col], heights, k,
                    r["means"], r["stds"], r["weights"],
                    r["boundaries"], r["bic"], r["aic"])

    # BIC / AIC comparison bar chart
    bics = [results[k]["bic"] for k in ks]
    aics = [results[k]["aic"] for k in ks]
    bic_min = min(bics)
    delta_bic = [b - bic_min for b in bics]

    ax_sum = axes[1, :]
    for ax in ax_sum[1:]:
        ax.set_visible(False)
    ax_sum = ax_sum[0]
    ax_sum.set_position([0.05, 0.04, 0.90, 0.22])

    x_pos  = np.arange(n_k)
    width  = 0.35
    bars_b = ax_sum.bar(x_pos - width/2, bics, width, label="BIC", color="#0072B2", alpha=0.8)
    bars_a = ax_sum.bar(x_pos + width/2, aics, width, label="AIC", color="#D55E00", alpha=0.8)
    ax_sum.set_xticks(x_pos)
    ax_sum.set_xticklabels([f"k={k}" for k in ks])
    ax_sum.set_ylabel("Score (lower = better)")
    ax_sum.set_title("BIC / AIC comparison", fontweight="bold")
    ax_sum.legend()
    ax_sum.axvline(ks.index(best_k_bic) - width/2, color="#0072B2", lw=1.5, ls="--", alpha=0.5)

    fig.suptitle(f"GMM k-sweep — KITTI {args.cls} H3D\n"
                 f"Best BIC: k={best_k_bic}  |  Best AIC: k={best_k_aic}",
                 fontsize=12, fontweight="bold", y=1.01)

    out_dir = os.path.join(ROOT_DIR, args.out_dir)
    pdf_path = os.path.join(out_dir, "gmm_k_search.pdf")
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {pdf_path}")

    txt_path = os.path.join(out_dir, "gmm_k_search.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report saved to {txt_path}")


if __name__ == "__main__":
    main()
