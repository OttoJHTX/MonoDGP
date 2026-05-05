"""
Unit tests for loss_projection_alignment on the projection-alignment branch.

Tests:
  1. loss is a non-negative scalar
  2. gradients flow to all predicted quantities (pred_boxes, pred_3d_dim, pred_depth, pred_angle)
  3. loss is L1 (not L2): gradient w.r.t. box position is constant regardless of error size
     (for L2, gradient ∝ error magnitude)
  4. geometry sanity: a box far from the projection has higher loss than one close to it

Run with:
  python tools/test_proj_align_loss.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
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
    # Leaf tensors so .grad accumulates after backward()
    pred_boxes = torch.tensor(
        [[[box2d_cx, 0.5, 0.1, 0.1, 0.1, 0.1]]],
        dtype=torch.float32, device=device, requires_grad=True)   # [1, 1, 6]

    pred_3d_dim = torch.tensor(
        [[[1.5, 2.0, 4.0]]],
        dtype=torch.float32, device=device, requires_grad=True)   # [1, 1, 3]

    pred_depth = torch.tensor(
        [[[10.0, 1.0]]],
        dtype=torch.float32, device=device, requires_grad=True)   # [1, 1, 2]

    pred_angle = torch.zeros(1, 1, 24, dtype=torch.float32, device=device, requires_grad=True)

    calibs = torch.zeros(1, 3, 4, device=device)
    calibs[0, 0, 0] = 700.0  # fu
    calibs[0, 1, 1] = 700.0  # fv
    calibs[0, 0, 2] = 640.0  # cu
    calibs[0, 1, 2] = 180.0  # cv

    img_sizes = torch.tensor([[1280.0, 360.0]], device=device)

    return {
        'pred_boxes': pred_boxes,
        'pred_3d_dim': pred_3d_dim,
        'pred_depth': pred_depth,
        'pred_angle': pred_angle,
        'calibs': calibs,
        'img_sizes': img_sizes,
    }


def compute_loss(box2d_cx=0.5):
    criterion = make_criterion()
    outputs = make_outputs(box2d_cx=box2d_cx)
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    result = criterion.loss_projection_alignment(outputs, targets=[], indices=indices, num_boxes=1.0)
    return result['loss_proj_align'], outputs


# ---------------------------------------------------------------------------

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
    """
    For L1: d|proj - box| / d(cx) = ±1, constant regardless of how far box is from projection.
    For L2: d(proj - box)^2 / d(cx) = -2*(proj - box), grows linearly with error.

    We check the gradient w.r.t. cx at a small offset and a large offset — they should be equal.
    """
    def cx_gradient(box2d_cx):
        loss, outputs = compute_loss(box2d_cx=box2d_cx)
        loss.backward()
        return outputs['pred_boxes'].grad[0, 0, 0].abs().item()  # |d_loss / d_cx|

    grad_small_err = cx_gradient(box2d_cx=0.5 + 0.05)
    grad_large_err = cx_gradient(box2d_cx=0.5 + 0.30)

    # L1: both should be equal (constant sign, not magnitude-dependent)
    # L2: grad_large / grad_small would be 0.30/0.05 = 6
    assert abs(grad_small_err - grad_large_err) < 0.01, (
        f"L1 gradient should be constant w.r.t. cx offset, "
        f"got {grad_small_err:.4f} (small error) vs {grad_large_err:.4f} (large error). "
        f"L2 would give a ~6x difference."
    )
    print(f"  PASS test_l1_not_l2: |d_loss/d_cx| = {grad_small_err:.4f} (small) and "
          f"{grad_large_err:.4f} (large) — constant confirms L1")


def test_geometry_sanity():
    """
    A box whose predicted 2D centre is far from the projected 3D centre
    should have higher loss than one that is close.
    Projected 3D centre lands near cx~0.5 (image centre); offset by 0.4 should be worse.
    """
    loss_near, _ = compute_loss(box2d_cx=0.5)
    loss_far,  _ = compute_loss(box2d_cx=0.1)

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
