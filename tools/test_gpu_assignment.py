"""
Quick test: verify that gpu_ids in config routes the model to the correct GPU.
Usage: python tools/test_gpu_assignment.py --config configs/monodgp.yaml
"""
import os
import sys
import torch
import yaml
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.helpers.model_helper import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))
expected_device = f"cuda:{gpu_ids[0]}"

model, _ = build_model(cfg['model'])
device = torch.device(expected_device if torch.cuda.is_available() else "cpu")
model = model.to(device)

actual_device = next(model.parameters()).device
assert str(actual_device) == expected_device, \
    f"FAIL: model is on {actual_device}, expected {expected_device}"

print(f"PASS: model is on {actual_device} (gpu_ids={gpu_ids})")
print(f"      torch.cuda.get_device_name({gpu_ids[0]}): {torch.cuda.get_device_name(gpu_ids[0])}")
