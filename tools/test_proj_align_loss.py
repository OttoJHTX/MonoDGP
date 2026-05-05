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
    pred_boxes[..., 2] = l (2D left half-width) appears ONLY in box2d_x1 = cx - l.
    It does NOT feed into the 3D projection (which uses pred_3d_dim for dimensions).
    So d_loss/d_l is the pure gradient of one L1 term.

    L1: |d_loss/d_l| = 1.0 regardless of how large the error is.
    L2: |d_loss/d_l| = 2 * |proj_x1 - box2d_x1|, grows with error.

    We pick two l values that place box2d_x1 on opposite sides of proj_x1
    (~0.378 for the default camera/depth setup) to confirm both give gradient 1.0.
    """
    def l_gradient(l_val):
        criterion = make_criterion()
        outputs = make_outputs()  # 3D params fixed; only pred_boxes changes
        pred_boxes = torch.tensor(
            [[[0.5, 0.5, l_val, 0.1, 0.1, 0.1]]],
            dtype=torch.float32, requires_grad=True)
        outputs['pred_boxes'] = pred_boxes
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        result = criterion.loss_projection_alignment(outputs, targets=[], indices=indices, num_boxes=1.0)
        result['loss_proj_align'].backward()
        return pred_boxes.grad[0, 0, 2].abs().item()  # |d_loss / d_l|

    # l=0.05 → box2d_x1=0.45  (right of proj_x1≈0.378, small error)
    # l=0.40 → box2d_x1=0.10  (left  of proj_x1≈0.378, large error, opposite sign)
    grad_small_err = l_gradient(0.05)
    grad_large_err = l_gradient(0.40)

    assert abs(grad_small_err - 1.0) < 0.01, \
        f"L1 gradient w.r.t. l should be 1.0, got {grad_small_err:.4f}"
    assert abs(grad_large_err - 1.0) < 0.01, \
        f"L1 gradient w.r.t. l should be 1.0, got {grad_large_err:.4f}"
    print(f"  PASS test_l1_not_l2: |d_loss/d_l| = {grad_small_err:.4f} (small error) and "
          f"{grad_large_err:.4f} (large error) — both 1.0 confirms L1 "
          f"(L2 would give ~0.14 and ~0.56)")


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
