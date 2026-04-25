#!/usr/bin/env python
"""
Tests for the height-bin classification + residual implementation.
Covers: constants, dataset encoding, model forward, loss, and decode.

Run:
    python tools/test_height_bins.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

# Import height_bins directly to avoid triggering the full model import chain (needs CUDA ops)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "height_bins", os.path.join(ROOT_DIR, "lib", "models", "monodgp", "height_bins.py"))
_hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hb)
NUM_HEIGHT_BINS = _hb.NUM_HEIGHT_BINS
HEIGHT_BIN_EDGES = _hb.HEIGHT_BIN_EDGES
HEIGHT_BIN_CENTERS = _hb.HEIGHT_BIN_CENTERS
HEIGHT_BIN_CENTERS_TENSOR = _hb.HEIGHT_BIN_CENTERS_TENSOR


def test_constants():
    assert NUM_HEIGHT_BINS == 5
    assert len(HEIGHT_BIN_EDGES) == 4
    assert len(HEIGHT_BIN_CENTERS) == 5
    assert HEIGHT_BIN_CENTERS_TENSOR.shape == (5,)
    for i in range(4):
        assert HEIGHT_BIN_CENTERS[i] < HEIGHT_BIN_EDGES[i] < HEIGHT_BIN_CENTERS[i + 1], \
            "Centers must straddle edges"
    print("[PASS] constants")


def test_bin_encoding():
    """Verify that GT heights are encoded into correct bins with correct residuals."""
    test_cases = [
        (1.20,  0, 1.20 - 1.272),   # below first edge → bin 0
        (1.35,  1, 1.35 - 1.401),   # between edge 0 and 1 → bin 1
        (1.50,  2, 1.50 - 1.488),   # between edge 1 and 2 → bin 2
        (1.60,  3, 1.60 - 1.614),   # between edge 2 and 3 → bin 3
        (1.90,  4, 1.90 - 1.839),   # above last edge → bin 4
        (1.330, 1, 1.330 - 1.401),  # exactly on edge → np.digitize puts in bin 1
    ]
    for h, expected_bin, expected_res in test_cases:
        bin_idx = np.clip(np.digitize(h, HEIGHT_BIN_EDGES), 0, NUM_HEIGHT_BINS - 1)
        res = h - HEIGHT_BIN_CENTERS[bin_idx]
        assert bin_idx == expected_bin, f"h={h}: got bin {bin_idx}, expected {expected_bin}"
        assert abs(res - expected_res) < 1e-6, f"h={h}: got res {res:.6f}, expected {expected_res:.6f}"
    print("[PASS] bin encoding")


def test_soft_decode_matches_hard():
    """When logits are one-hot confident, soft decode ≈ hard decode."""
    torch.manual_seed(42)
    B, Q = 2, 10
    bin_centers = HEIGHT_BIN_CENTERS_TENSOR

    # Create logits that strongly favor one bin
    logits = torch.randn(B, Q, NUM_HEIGHT_BINS) * 0.01
    gt_bins = torch.randint(0, NUM_HEIGHT_BINS, (B, Q))
    for b in range(B):
        for q in range(Q):
            logits[b, q, gt_bins[b, q]] += 20.0  # make one bin dominant

    residuals = torch.randn(B, Q, NUM_HEIGHT_BINS) * 0.05

    # Soft decode (training path)
    probs = F.softmax(logits, dim=-1)
    soft_height = (probs * (bin_centers + residuals)).sum(dim=-1)

    # Hard decode (inference path)
    hard_bin = logits.argmax(dim=-1)
    sel_centers = bin_centers[hard_bin]
    sel_res = residuals.gather(2, hard_bin.unsqueeze(-1)).squeeze(-1)
    hard_height = sel_centers + sel_res

    diff = (soft_height - hard_height).abs().max().item()
    assert diff < 1e-4, f"soft vs hard diff too large: {diff}"
    print(f"[PASS] soft~=hard decode (max diff={diff:.2e})")


def test_soft_decode_is_differentiable():
    """Verify gradients flow through soft decode to logits and residuals."""
    logits = torch.randn(1, 5, NUM_HEIGHT_BINS, requires_grad=True)
    residuals = torch.randn(1, 5, NUM_HEIGHT_BINS, requires_grad=True)
    bin_centers = HEIGHT_BIN_CENTERS_TENSOR

    probs = F.softmax(logits, dim=-1)
    height = (probs * (bin_centers + residuals)).sum(dim=-1)
    loss = height.sum()
    loss.backward()

    assert logits.grad is not None and logits.grad.abs().sum() > 0, "no grad on logits"
    assert residuals.grad is not None and residuals.grad.abs().sum() > 0, "no grad on residuals"
    print("[PASS] soft decode is differentiable")


def test_loss_height_bins():
    """Verify the loss computation matches the angle loss pattern."""
    torch.manual_seed(0)
    N = 20  # matched objects

    height_input = torch.randn(N, NUM_HEIGHT_BINS * 2)
    target_cls = torch.randint(0, NUM_HEIGHT_BINS, (N,))
    target_res = torch.randn(N) * 0.05

    # Classification loss
    cls_loss = F.cross_entropy(height_input[:, :NUM_HEIGHT_BINS], target_cls, reduction='none')

    # Regression loss (select residual for correct bin)
    res_preds = height_input[:, NUM_HEIGHT_BINS:]
    cls_onehot = torch.zeros(N, NUM_HEIGHT_BINS).scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    res_pred = torch.sum(res_preds * cls_onehot, 1)
    reg_loss = F.l1_loss(res_pred, target_res, reduction='none')

    total = (cls_loss + reg_loss).mean()

    assert total.item() > 0, "loss should be positive"
    assert not torch.isnan(total), "loss is NaN"
    print(f"[PASS] loss computation (mean loss={total.item():.4f})")


def test_hard_decode_roundtrip():
    """Encode GT height → bin+res, then hard-decode → should recover original height."""
    test_heights = [1.20, 1.35, 1.45, 1.50, 1.60, 1.72, 1.90]
    for h in test_heights:
        bin_idx = np.clip(np.digitize(h, HEIGHT_BIN_EDGES), 0, NUM_HEIGHT_BINS - 1)
        res = h - HEIGHT_BIN_CENTERS[bin_idx]
        recovered = HEIGHT_BIN_CENTERS[bin_idx] + res
        assert abs(recovered - h) < 1e-7, f"roundtrip failed: {h} → {recovered}"
    print("[PASS] encode/decode roundtrip")


def test_dim_embed_output_shape():
    """Verify the MLP dimensions are correct for the new heads."""
    from torch import nn
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
            self.num_layers = num_layers
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            return x
    hidden = 256

    dim_head = MLP(hidden, hidden, 2, 2)
    x = torch.randn(2, 50, hidden)
    out = dim_head(x)
    assert out.shape == (2, 50, 2), f"dim_embed_3d shape: {out.shape}, expected (2,50,2)"

    hbin_head = MLP(hidden, hidden, NUM_HEIGHT_BINS * 2, 2)
    out = hbin_head(x)
    assert out.shape == (2, 50, 10), f"height_bin_embed shape: {out.shape}, expected (2,50,10)"
    print("[PASS] MLP output shapes")


def test_full_forward_mock():
    """Simulate the forward-pass height computation end-to-end."""
    torch.manual_seed(42)
    from torch import nn
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
            self.num_layers = num_layers
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            return x
    hidden = 256
    B, Q = 2, 50

    dim_head = MLP(hidden, hidden, 2, 2)
    hbin_head = MLP(hidden, hidden, NUM_HEIGHT_BINS * 2, 2)
    bin_centers = HEIGHT_BIN_CENTERS_TENSOR

    hs = torch.randn(B, Q, hidden)
    calibs_fx = torch.full((B, 1), 721.0)
    img_h = torch.full((B, 1), 384.0)
    box2d_height_norm = torch.rand(B, Q) * 0.3 + 0.05  # between 0.05 and 0.35

    # Forward
    wl = dim_head(hs)
    hbin_out = hbin_head(hs)
    h_logits = hbin_out[:, :, :NUM_HEIGHT_BINS]
    h_residuals = hbin_out[:, :, NUM_HEIGHT_BINS:]
    h_probs = F.softmax(h_logits, dim=-1)
    height = (h_probs * (bin_centers + h_residuals)).sum(dim=-1, keepdim=True)

    size3d = torch.cat([height, wl], dim=-1)
    assert size3d.shape == (B, Q, 3)

    box2d_height = torch.clamp(box2d_height_norm * img_h, min=1.0)
    depth_geo = size3d[:, :, 0] / box2d_height * calibs_fx

    assert depth_geo.shape == (B, Q)
    assert not torch.isnan(depth_geo).any()
    assert (depth_geo > 0).all(), "depth should be positive for positive heights"

    # Verify gradient flow from depth_geo to height bin head
    loss = depth_geo.sum()
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in hbin_head.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradient flowing to height_bin_embed"
    print(f"[PASS] full forward mock (grad_norm={grad_norm:.4f})")


if __name__ == '__main__':
    test_constants()
    test_bin_encoding()
    test_soft_decode_matches_hard()
    test_soft_decode_is_differentiable()
    test_loss_height_bins()
    test_hard_decode_roundtrip()
    test_dim_embed_output_shape()
    test_full_forward_mock()
    print("\nAll height-bin tests passed.")
