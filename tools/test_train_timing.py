"""
Verifies that Trainer.train() logs a total-training-time line and that
the reported duration is close to real wall-clock time.

Run:
    python tools/test_train_timing.py
"""
import os
import sys
import time
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from lib.helpers.trainer_helper import Trainer


class _ListLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


class _NoopScheduler:
    def step(self):
        pass


def main():
    # Bypass __init__ — we only want to exercise .train().
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = {
        'max_epoch': 3,
        'save_frequency': 999,  # skip checkpoint/eval branch
        'save_all': False,
        'save_path': 'outputs/',
    }
    trainer.epoch = 0
    trainer.best_result = 0
    trainer.best_epoch = 0
    trainer.warmup_lr_scheduler = None
    trainer.lr_scheduler = _NoopScheduler()
    trainer.logger = _ListLogger()
    trainer.output_dir = '/tmp/unused'
    trainer.tester = None
    trainer.model = None
    trainer.optimizer = None

    sleep_per_epoch = 0.2

    def fake_train_one_epoch(epoch):
        time.sleep(sleep_per_epoch)

    trainer.train_one_epoch = fake_train_one_epoch

    wall_start = time.time()
    trainer.train()
    wall_elapsed = time.time() - wall_start

    timing_lines = [m for m in trainer.logger.messages if m.startswith("Total training time:")]
    if len(timing_lines) != 1:
        raise SystemExit(
            "FAIL: expected exactly one 'Total training time:' line, got {}:\n{}".format(
                len(timing_lines), trainer.logger.messages))

    line = timing_lines[0]
    print("Logged line:", line)

    # Parse the "(XX.Xs over N epoch(s))" segment.
    import re
    match = re.search(r"\(([\d.]+)s over (\d+) epoch\(s\)\)", line)
    if not match:
        raise SystemExit("FAIL: could not parse timing line: " + line)

    reported_seconds = float(match.group(1))
    reported_epochs = int(match.group(2))

    expected = sleep_per_epoch * trainer.cfg['max_epoch']
    if reported_epochs != trainer.cfg['max_epoch']:
        raise SystemExit(
            "FAIL: epoch count mismatch: reported={}, expected={}".format(
                reported_epochs, trainer.cfg['max_epoch']))

    # Reported time should be within ~0.5s of actual wall time (which itself
    # should be ~= expected). Loose bounds to survive CI jitter.
    if not (expected - 0.1 <= reported_seconds <= wall_elapsed + 0.5):
        raise SystemExit(
            "FAIL: duration {:.3f}s outside plausible range [{:.3f}, {:.3f}]".format(
                reported_seconds, expected - 0.1, wall_elapsed + 0.5))

    print("[PASS] timing log present, epochs={}, reported={:.3f}s, wall={:.3f}s".format(
        reported_epochs, reported_seconds, wall_elapsed))


if __name__ == "__main__":
    main()
