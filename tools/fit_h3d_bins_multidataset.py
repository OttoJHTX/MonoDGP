#!/usr/bin/env python
"""
Fit height bins from car H3D across multiple datasets (KITTI, nuScenes, Waymo)
to avoid data leakage from using only KITTI val annotations.

Usage:
    # KITTI train only (always available):
    python tools/fit_h3d_bins_multidataset.py --config configs/monodgp.yaml --k 5

    # With nuScenes:
    python tools/fit_h3d_bins_multidataset.py --config configs/monodgp.yaml --k 5 \
        --nuscenes_root /path/to/nuscenes

    # With Waymo (KITTI-format labels):
    python tools/fit_h3d_bins_multidataset.py --config configs/monodgp.yaml --k 5 \
        --waymo_label_dir /path/to/waymo/kitti_labels

    # All three:
    python tools/fit_h3d_bins_multidataset.py --config configs/monodgp.yaml --k 5 \
        --nuscenes_root /path/to/nuscenes \
        --waymo_label_dir /path/to/waymo/kitti_labels

nuScenes annotations only need the JSON files — no images required.
  Expected structure: <nuscenes_root>/v1.0-trainval/sample_annotation.json
                      <nuscenes_root>/v1.0-trainval/category.json

Waymo expects KITTI-format label .txt files in a flat directory
(as produced by OpenPCDet or similar converters).
"""
import os
import sys
import json
import glob
import yaml
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from lib.datasets.kitti.kitti_utils import get_objects_from_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/monodgp.yaml')
    parser.add_argument('--k', type=int, required=True, help='Number of GMM components')
    parser.add_argument('--kitti_split', default='train',
                        help='Which KITTI split to use (default: train, not val)')
    parser.add_argument('--nuscenes_root', default=None,
                        help='Root directory of nuScenes dataset')
    parser.add_argument('--nuscenes_version', default='v1.0-trainval',
                        help='nuScenes version folder (default: v1.0-trainval)')
    parser.add_argument('--waymo_label_dir', default=None,
                        help='Directory of Waymo labels in KITTI format')
    parser.add_argument('--output', default='h3d_bins_multidataset.pdf')
    return parser.parse_args()


# ---- KITTI ----

def load_kitti_heights(root_dir, split):
    split_file = os.path.join(root_dir, 'ImageSets', split + '.txt')
    label_dir = os.path.join(root_dir, 'training', 'label_2')
    assert os.path.exists(split_file), f"Split file not found: {split_file}"
    assert os.path.exists(label_dir), f"Label dir not found: {label_dir}"

    idx_list = [x.strip() for x in open(split_file).readlines()]
    heights = []
    for idx in idx_list:
        for obj in get_objects_from_label(os.path.join(label_dir, idx + '.txt')):
            if obj.cls_type == 'Car':
                heights.append(obj.h)
    return np.array(heights)


# ---- nuScenes ----

def load_nuscenes_heights(nuscenes_root, version='v1.0-trainval'):
    """
    Load car heights from nuScenes annotation JSONs.
    Only the two JSON files are needed — no images or sensor data.

    nuScenes size field: [width, length, height] (all in metres).
    Car categories: 'vehicle.car'.
    """
    ann_file = os.path.join(nuscenes_root, version, 'sample_annotation.json')
    cat_file = os.path.join(nuscenes_root, version, 'category.json')

    if not os.path.exists(ann_file):
        raise FileNotFoundError(
            f"nuScenes annotation file not found: {ann_file}\n"
            f"Make sure --nuscenes_root points to the dataset root and "
            f"--nuscenes_version matches the folder name.")

    with open(cat_file) as f:
        categories = json.load(f)
    cat_map = {c['token']: c['name'] for c in categories}

    with open(ann_file) as f:
        annotations = json.load(f)

    heights = []
    for ann in annotations:
        cat_name = cat_map.get(ann['category_token'], '')
        if cat_name == 'vehicle.car':
            # nuScenes size: [width, length, height]
            heights.append(ann['size'][2])

    return np.array(heights)


# ---- Waymo (KITTI-format labels) ----

def load_waymo_heights(label_dir):
    """
    Load car heights from Waymo labels converted to KITTI format.
    Any .txt file in label_dir is treated as a KITTI label file.
    """
    heights = []
    for fpath in glob.glob(os.path.join(label_dir, '*.txt')):
        for obj in get_objects_from_label(fpath):
            if obj.cls_type == 'Car':
                heights.append(obj.h)
    return np.array(heights)


# ---- GMM fitting ----

def find_crossover(mu1, std1, w1, mu2, std2, w2):
    lo, hi = min(mu1, mu2), max(mu1, mu2)
    xs = np.linspace(lo, hi, 10000)
    diff = w1 * norm.pdf(xs, mu1, std1) - w2 * norm.pdf(xs, mu2, std2)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    return xs[sign_changes[-1]] if len(sign_changes) else (mu1 + mu2) / 2


def fit_gmm(heights, k):
    gmm = GaussianMixture(n_components=k, covariance_type='full', n_init=20, random_state=42)
    gmm.fit(heights.reshape(-1, 1))
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    order = np.argsort(means)
    return means[order], stds[order], weights[order]


def main():
    args = parse_args()

    cfg_path = os.path.join(ROOT_DIR, args.config) if not os.path.isabs(args.config) else args.config
    cfg = yaml.load(open(cfg_path), Loader=yaml.Loader)
    root_dir = cfg['dataset']['root_dir']
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(ROOT_DIR, root_dir)

    # ---- Load heights ----
    dataset_heights = {}

    print(f"Loading KITTI {args.kitti_split} split ...")
    kitti_h = load_kitti_heights(root_dir, args.kitti_split)
    dataset_heights['KITTI'] = kitti_h
    print(f"  {len(kitti_h)} cars  mean={kitti_h.mean():.3f}  std={kitti_h.std():.3f}")

    if args.nuscenes_root:
        print(f"Loading nuScenes ({args.nuscenes_version}) ...")
        ns_h = load_nuscenes_heights(args.nuscenes_root, args.nuscenes_version)
        dataset_heights['nuScenes'] = ns_h
        print(f"  {len(ns_h)} cars  mean={ns_h.mean():.3f}  std={ns_h.std():.3f}")

    if args.waymo_label_dir:
        print(f"Loading Waymo (KITTI-format labels) ...")
        wy_h = load_waymo_heights(args.waymo_label_dir)
        dataset_heights['Waymo'] = wy_h
        print(f"  {len(wy_h)} cars  mean={wy_h.mean():.3f}  std={wy_h.std():.3f}")

    combined = np.concatenate(list(dataset_heights.values()))
    print(f"\nCombined: {len(combined)} cars  mean={combined.mean():.3f}  std={combined.std():.3f}")

    # ---- Fit GMM ----
    means, stds, weights = fit_gmm(combined, args.k)

    print(f"\nGMM Components (K={args.k}, fit on combined data):")
    print(f"  {'#':>2}  {'Weight':>8}  {'Mean (m)':>9}  {'Std (m)':>8}")
    for i in range(args.k):
        print(f"  {i+1:2d}  {weights[i]:8.3f}  {means[i]:9.3f}  {stds[i]:8.3f}")

    boundaries = [find_crossover(means[i], stds[i], weights[i],
                                 means[i+1], stds[i+1], weights[i+1])
                  for i in range(args.k - 1)]

    bin_edges = [float('-inf')] + boundaries + [float('inf')]
    print(f"\nBin boundaries: {[f'{b:.4f}' for b in boundaries]}")
    print(f"\nBins (evaluated on combined data):")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (combined >= lo) & (combined < hi)
        n = mask.sum()
        print(f"  Bin {i+1}: [{lo:>9.4f}, {hi:>9.4f})  "
              f"n={n:>6}  mean={combined[mask].mean():.4f}  std={combined[mask].std():.4f}")

    print(f"\nPaste these into lib/models/monodgp/height_bins.py:")
    centers = [f"{combined[(combined >= bin_edges[i]) & (combined < bin_edges[i+1])].mean():.4f}"
               for i in range(len(bin_edges) - 1)]
    print(f"HEIGHT_BIN_EDGES   = np.array([{', '.join(f'{b:.4f}' for b in boundaries)}])")
    print(f"HEIGHT_BIN_CENTERS = np.array([{', '.join(centers)}])")

    # ---- Plot ----
    colors = {'KITTI': '#4080D0', 'nuScenes': '#D04040', 'Waymo': '#40B040'}
    cmap = plt.cm.Set1
    x = np.linspace(combined.min() - 0.3, combined.max() + 0.3, 500)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={'width_ratios': [3, 1]})

    ax = axes[0]
    bins_hist = np.linspace(combined.min() - 0.1, combined.max() + 0.1, 70)
    total_n = len(combined)
    for name, h in dataset_heights.items():
        ax.hist(h, bins=bins_hist, density=True,
                alpha=0.35, color=colors.get(name, 'gray'),
                label=f'{name} (n={len(h)})', edgecolor='none')

    overall = np.zeros_like(x)
    for i in range(args.k):
        yi = weights[i] * norm.pdf(x, means[i], stds[i])
        overall += yi
        ax.plot(x, yi, color=cmap(i), linewidth=1.5, linestyle='--',
                label=f'K{i+1}: mu={means[i]:.2f}, w={weights[i]:.2f}')
        ax.fill_between(x, yi, alpha=0.07, color=cmap(i))

    ax.plot(x, overall, color='black', linewidth=2, label='GMM fit (combined)')

    for b in boundaries:
        ax.axvline(b, color='#D04040', linewidth=1.5, linestyle=':')
        ax.text(b, ax.get_ylim()[1] * 0.97, f'{b:.3f}m',
                ha='center', va='top', fontsize=8, color='#D04040',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_xlabel('H3D - 3D height (m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    datasets_str = ' + '.join(dataset_heights.keys())
    ax.set_title(f'Car H3D: GMM K={args.k} fit on {datasets_str}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

    # Per-dataset mean markers
    ax2 = axes[1]
    ax2.set_title('Per-dataset mean & std', fontsize=11, fontweight='bold')
    ys = range(len(dataset_heights))
    for y, (name, h) in zip(ys, dataset_heights.items()):
        ax2.errorbar(h.mean(), y, xerr=h.std(), fmt='o',
                     color=colors.get(name, 'gray'), capsize=5, markersize=7)
        ax2.text(h.mean(), y + 0.15, f'{h.mean():.3f}+/-{h.std():.3f}',
                 ha='center', fontsize=8)
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels(list(dataset_heights.keys()))
    ax2.set_xlabel('H3D (m)', fontsize=10)
    ax2.set_xlim(combined.min() - 0.3, combined.max() + 0.3)

    fig.tight_layout()
    out_path = os.path.join(ROOT_DIR, args.output)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")


if __name__ == '__main__':
    main()
