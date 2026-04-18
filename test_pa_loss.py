"""Quick smoke test for the projection-alignment loss."""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import math
import numpy as np


def test_circular_angle():
    """Verify circular mean handles the wrap-around boundary correctly."""
    angle_per_class = 2 * math.pi / 12.0
    bin_centers = torch.arange(12, dtype=torch.float32) * angle_per_class

    # Case 1: True alpha near 0, softmax splits weight between bin 0 (0°) and bin 11 (330°)
    angle_cls = torch.zeros(1, 12)
    angle_cls[0, 0] = 5.0
    angle_cls[0, 11] = 4.0
    angle_res = torch.zeros(1, 12)

    weights = torch.softmax(angle_cls, dim=1)

    # OLD (buggy): linear average
    old_alpha = (weights * bin_centers.unsqueeze(0)).sum(dim=1)

    # NEW (fixed): circular mean
    theta_per_bin = bin_centers.unsqueeze(0) + angle_res
    sin_a = (weights * torch.sin(theta_per_bin)).sum(dim=1)
    cos_a = (weights * torch.cos(theta_per_bin)).sum(dim=1)
    new_alpha = torch.atan2(sin_a, cos_a)

    old_deg = old_alpha.item() * 180 / math.pi
    new_deg = new_alpha.item() * 180 / math.pi

    print(f"  Wrap-around test (true alpha ~ 0 deg):")
    print(f"    Old (linear):   {old_deg:7.1f} deg  <- WRONG (should be near 0)")
    print(f"    New (circular): {new_deg:7.1f} deg  <- CORRECT")
    assert abs(new_deg) < 30, f"Circular mean should be near 0, got {new_deg}"

    # Case 2: True alpha near 90°, no wrap issue — both should agree
    angle_cls2 = torch.zeros(1, 12)
    angle_cls2[0, 3] = 10.0
    weights2 = torch.softmax(angle_cls2, dim=1)
    theta2 = bin_centers.unsqueeze(0) + angle_res
    old2 = (weights2 * bin_centers.unsqueeze(0)).sum(dim=1).item() * 180 / math.pi
    new2 = torch.atan2(
        (weights2 * torch.sin(theta2)).sum(dim=1),
        (weights2 * torch.cos(theta2)).sum(dim=1)
    ).item() * 180 / math.pi
    print(f"  No-wrap test (true alpha ~ 90 deg):")
    print(f"    Old (linear):   {old2:7.1f} deg")
    print(f"    New (circular): {new2:7.1f} deg")
    assert abs(old2 - new2) < 1.0, "Should agree when no wrap-around"

    print("  PASSED\n")


def test_loss_runs():
    """Run the PA loss computation standalone with dummy data and check gradients."""
    device = torch.device('cpu')

    B, Q = 1, 50
    pred_boxes = torch.sigmoid(torch.randn(B, Q, 6, device=device)).requires_grad_(True)
    pred_dims = (torch.abs(torch.randn(B, Q, 3, device=device)) + 0.5).requires_grad_(True)
    pred_depth_raw = (torch.abs(torch.randn(B, Q, 2, device=device)) + 5.0).requires_grad_(True)
    pred_angle = torch.randn(B, Q, 24, device=device).requires_grad_(True)

    calibs = torch.zeros(B, 3, 4, device=device)
    calibs[0, 0, 0] = 721.5; calibs[0, 1, 1] = 721.5
    calibs[0, 0, 2] = 609.6; calibs[0, 1, 2] = 172.9; calibs[0, 2, 2] = 1.0
    img_sizes = torch.tensor([[1242.0, 375.0]], device=device)

    batch_idx = torch.tensor([0, 0, 0])
    src_idx = torch.tensor([0, 5, 10])
    idx = (batch_idx, src_idx)

    # --- Reproduce the loss computation (mirrors current monodgp.py) ---
    boxes = pred_boxes[idx]
    cx, cy = boxes[:, 0], boxes[:, 1]
    l, r, t, b = boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5]
    box2d_x1, box2d_y1 = cx - l, cy - t
    box2d_x2, box2d_y2 = cx + r, cy + b

    dims = pred_dims[idx]
    depth = pred_depth_raw[idx][:, 0]
    angle_raw = pred_angle[idx]

    # Circular angle decoding
    angle_cls, angle_res = angle_raw[:, :12], angle_raw[:, 12:]
    weights = torch.softmax(angle_cls, dim=1)
    angle_per_class = 2 * math.pi / 12.0
    bin_centers = torch.arange(12, dtype=torch.float32) * angle_per_class
    theta_per_bin = bin_centers.unsqueeze(0) + angle_res
    sin_a = (weights * torch.sin(theta_per_bin)).sum(dim=1)
    cos_a = (weights * torch.cos(theta_per_bin)).sum(dim=1)
    alpha = torch.atan2(sin_a, cos_a)

    fu = calibs[batch_idx, 0, 0]
    fv = calibs[batch_idx, 1, 1]
    cu = calibs[batch_idx, 0, 2]
    cv = calibs[batch_idx, 1, 2]
    img_w = img_sizes[batch_idx, 0]
    img_h = img_sizes[batch_idx, 1]

    cx_px, cy_px = cx * img_w, cy * img_h
    Z = depth
    X = (cx_px - cu) * Z / fu
    Y = (cy_px - cv) * Z / fv
    ry = alpha + torch.atan2(cx_px - cu, fu)

    h, w, ll = dims[:, 0], dims[:, 1], dims[:, 2]
    cos_ry, sin_ry = torch.cos(ry), torch.sin(ry)

    # Bottom-center y-corners (KITTI convention)
    x_corners = torch.stack([ll/2, ll/2, -ll/2, -ll/2, ll/2, ll/2, -ll/2, -ll/2], dim=1)
    y_corners = torch.stack([torch.zeros_like(h), torch.zeros_like(h),
                             torch.zeros_like(h), torch.zeros_like(h),
                             -h, -h, -h, -h], dim=1)
    z_corners = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1)

    x_rot = cos_ry.unsqueeze(1) * x_corners + sin_ry.unsqueeze(1) * z_corners
    z_rot = -sin_ry.unsqueeze(1) * x_corners + cos_ry.unsqueeze(1) * z_corners
    x_cam = x_rot + X.unsqueeze(1)
    y_cam = y_corners + Y.unsqueeze(1)
    z_cam = torch.clamp(z_rot + Z.unsqueeze(1), min=1.0)

    u_proj = fu.unsqueeze(1) * x_cam / z_cam + cu.unsqueeze(1)
    v_proj = fv.unsqueeze(1) * y_cam / z_cam + cv.unsqueeze(1)

    proj_x1 = u_proj.min(dim=1).values / img_w
    proj_x2 = u_proj.max(dim=1).values / img_w
    proj_y1 = v_proj.min(dim=1).values / img_h
    proj_y2 = v_proj.max(dim=1).values / img_h

    loss_pa = ((box2d_x1 - proj_x1)**2 + (box2d_x2 - proj_x2)**2 +
               (box2d_y1 - proj_y1)**2 + (box2d_y2 - proj_y2)**2)
    loss = loss_pa.sum() / 3.0

    print(f"  loss_proj_align = {loss.item():.6f}")
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() >= 0, f"Loss should be non-negative: {loss.item()}"

    # Check gradients flow to all prediction heads
    loss.backward()
    for name, tensor in [('pred_boxes', pred_boxes), ('pred_dims', pred_dims),
                          ('pred_depth', pred_depth_raw), ('pred_angle', pred_angle)]:
        assert tensor.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"Non-finite gradient for {name}"
        print(f"    {name:15s} grad norm: {tensor.grad.norm().item():.6f}")

    print("  PASSED\n")


if __name__ == '__main__':
    print("Test 1: Circular angle decoding")
    test_circular_angle()

    print("Test 2: Loss computation + gradient flow")
    test_loss_runs()

    print("All tests passed!")
