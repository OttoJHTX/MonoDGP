"""
Unit tests for loss_projection_alignment on the projection-alignment branch.

Tests:
  1. loss is a non-negative scalar
  2. gradients flow to all predicted quantities (pred_boxes, pred_3d_dim, pred_depth, pred_angle)
  3. loss is L1 (not L2): doubling the 2D box offset doubles the loss
  4. geometry sanity: a box far right has higher alignment error than the same box centred

Run with:
  python tools/test_proj_align_loss.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from unittest.mock import MagicMock
from lib.models.monodgp.monodgp import SetCriterion


def make_criterion():
    criterion = SetCriterion.__new__(SetCriterion)
    criterion.num_classes = 3
    criterion.weight_dict = {}
    criterion.losses = []
    criterion.inter_losses = []
    return criterion


def make_outputs(box2d_cx=0.5, device='cpu'):
    """
    Single batch, single matched query.
    Camera: fu=fv=700, cu=640, cv=180, image 1280x360.
    Object: Z=10m, dims H=1.5 W=2.0 L=4.0, ry~0 (uniform angle logits).
    """
    B, Q = 1, 1

    pred_boxes_val = torch.tensor([[box2d_cx, 0.5, 0.1, 0.1, 0.1, 0.1]], device=device)  # [1, 6]
    pred_boxes = pred_boxes_val.unsqueeze(0).requires_grad_(True)   # [1, 1, 6]

    pred_3d_dim_val = torch.tensor([[1.5, 2.0, 4.0]], device=device)
    pred_3d_dim = pred_3d_dim_val.unsqueeze(0).requires_grad_(True)  # [1, 1, 3]

    pred_depth_val = torch.tensor([[10.0, 1.0]], device=device)
    pred_depth = pred_depth_val.unsqueeze(0).requires_grad_(True)    # [1, 1, 2]

    pred_angle = torch.zeros(B, Q, 24, device=device, requires_grad=True)

    calibs = torch.zeros(B, 3, 4, device=device)
    calibs[0, 0, 0] = 700.0
    calibs[0, 1, 1] = 700.0
    calibs[0, 0, 2] = 640.0
    calibs[0, 1, 2] = 180.0

    img_sizes = torch.tensor([[1280.0, 360.0]], device=device)

    return {
        'pred_boxes': pred_boxes,
        'pred_3d_dim': pred_3d_dim,
        'pred_depth': pred_depth,
        'pred_angle': pred_angle,
        'calibs': calibs,
        'img_sizes': img_sizes,
    }


def make_indices():
    return [(torch.tensor([0]), torch.tensor([0]))]


def compute_loss(box2d_cx=0.5):
    criterion = make_criterion()
    outputs = make_outputs(box2d_cx=box2d_cx)
    indices = make_indices()
    result = criterion.loss_projection_alignment(outputs, targets=[], indices=indices, num_boxes=1.0)
    return result['loss_proj_align'], outputs


def test_non_negative():
    loss, _ = compute_loss()
    assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"
    print(f"  PASS test_non_negative: loss={loss.item():.6f}")


def test_gradient_flow():
    loss, outputs = compute_loss()
    loss.backward()

    for name in ('pred_boxes', 'pred_3d_dim', 'pred_depth', 'pred_angle'):
        grad = outputs[name].grad
        assert grad is not None, f"No gradient for {name}"
        assert grad.abs().sum().item() > 0, f"Zero gradient for {name}"

    print("  PASS test_gradient_flow: gradients exist for pred_boxes, pred_3d_dim, pred_depth, pred_angle")


def test_l1_not_l2():
    """L1: doubling the offset doubles the loss. L2 would quadruple it."""
    base_cx = 0.5
    offset = 0.1

    loss1, _ = compute_loss(box2d_cx=base_cx + offset)
    loss2, _ = compute_loss(box2d_cx=base_cx + 2 * offset)

    ratio = loss2.item() / loss1.item()
    assert abs(ratio - 2.0) < 0.05, (
        f"Expected L1 scaling (ratio~2.0), got {ratio:.4f}. "
        f"L2 would give ratio~4.0."
    )
    print(f"  PASS test_l1_not_l2: loss(2x offset)/loss(1x offset) = {ratio:.4f} (expected ~2.0)")


def test_geometry_sanity():
    """
    A box whose predicted 2D centre is far from the projected 3D centre
    should have higher loss than one that is close.
    """
    # projected 3D centre lands near cx~0.5 (centre of image)
    loss_near, _ = compute_loss(box2d_cx=0.5)
    loss_far, _  = compute_loss(box2d_cx=0.05)  # 2D box shifted far left

    assert loss_far.item() > loss_near.item(), (
        f"Box far from projection should have higher loss. "
        f"near={loss_near.item():.4f}, far={loss_far.item():.4f}"
    )
    print(f"  PASS test_geometry_sanity: loss_far={loss_far.item():.4f} > loss_near={loss_near.item():.4f}")


if __name__ == '__main__':
    print("Running projection-alignment loss tests...\n")
    test_non_negative()
    test_gradient_flow()
    test_l1_not_l2()
    test_geometry_sanity()
    print("\nAll tests passed.")
