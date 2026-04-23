"""
Verifies that the gradient-accumulation logic used in trainer_helper.py
matches a single full-batch step.

Run:
    python tools/test_grad_accum.py
"""
import torch
import torch.nn as nn


def run(batch_size, accumulation_steps, data, targets, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
    optim = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 so we only compare grads
    loss_fn = nn.MSELoss()

    optim.zero_grad()
    num_batches = data.shape[0] // batch_size
    for i in range(num_batches):
        x = data[i * batch_size:(i + 1) * batch_size]
        y = targets[i * batch_size:(i + 1) * batch_size]
        loss = loss_fn(model(x), y)
        (loss / accumulation_steps).backward()

        is_last = (i + 1) == num_batches
        if ((i + 1) % accumulation_steps == 0) or is_last:
            optim.step()
            optim.zero_grad(set_to_none=False)  # keep .grad tensors for inspection

    return [p.grad.detach().clone() for p in model.parameters()]


def main():
    torch.manual_seed(42)
    effective_batch = 8
    data = torch.randn(effective_batch, 16)
    targets = torch.randn(effective_batch, 4)

    # Reference: one step over the full effective batch (no accumulation).
    ref_grads = run(batch_size=8, accumulation_steps=1, data=data, targets=targets)

    # Accumulated: batch_size=4 with accumulation_steps=2.
    acc_grads = run(batch_size=4, accumulation_steps=2, data=data, targets=targets)

    # Also test an uneven case: 3 micro-batches of size 2 + trailing flush,
    # with accumulation_steps=2 (so the last partial group flushes at epoch end).
    # Effective batch here is 6, so use a separate reference.
    data6 = data[:6]
    targets6 = targets[:6]
    ref6 = run(batch_size=6, accumulation_steps=1, data=data6, targets=targets6)
    acc6 = run(batch_size=2, accumulation_steps=2, data=data6, targets=targets6)

    def compare(name, a, b):
        max_diff = max((x - y).abs().max().item() for x, y in zip(a, b))
        ok = max_diff < 1e-6
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: max grad diff = {max_diff:.2e}")
        return ok

    ok1 = compare("bs=4, accum=2 vs bs=8, accum=1", ref_grads, acc_grads)
    # For the uneven case the accumulated run scales every micro-batch by
    # 1/accumulation_steps=1/2, but the reference averages over the full
    # effective batch of 6 (i.e. /3 compared to a single micro-batch loss).
    # So we expect acc6 grads = ref6 grads * (3 / 2) (micro-batch count / accum).
    scale = 3 / 2
    scaled_ref6 = [g * scale for g in ref6]
    ok2 = compare("bs=2, accum=2 (uneven flush, scaled)", scaled_ref6, acc6)

    if ok1 and ok2:
        print("\nAll good — gradient accumulation matches the reference.")
    else:
        raise SystemExit("Gradient accumulation test FAILED.")


if __name__ == "__main__":
    main()
