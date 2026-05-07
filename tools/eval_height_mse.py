"""
Evaluate 3D height prediction MSE against GT on the val set.

Usage:
  python tools/eval_height_mse.py --config configs/monodgp.yaml --checkpoint path/to/checkpoint.pth
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.save_helper import load_checkpoint


class _Logger:
    def info(self, msg): print(msg)


def prepare_targets(targets, batch_size):
    mask = targets['mask_2d']
    key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
    targets_list = []
    for bz in range(batch_size):
        d = {}
        for key, val in targets.items():
            if key in key_list:
                d[key] = val[bz][mask[bz]]
            if key == 'depth_map':
                d[key] = val[bz]
            if key == 'obj_region':
                d[key] = val[bz]
        targets_list.append(d)
    return targets_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model and loss (criterion contains the matcher)
    model, criterion = build_model(cfg['model'])
    load_checkpoint(model=model, optimizer=None,
                    filename=args.checkpoint, map_location=device, logger=_Logger())
    model.to(device).eval()
    criterion.to(device)

    # val loader only
    _, val_loader = build_dataloader(cfg['dataset'])

    all_pred_h = []
    all_gt_h   = []

    torch.set_grad_enabled(False)
    for inputs, calibs, targets, info in tqdm.tqdm(val_loader, desc='eval'):
        inputs = inputs.to(device)
        calibs = calibs.to(device)
        for k in targets:
            targets[k] = targets[k].to(device)

        img_sizes = targets['img_size']
        targets_list = prepare_targets(targets, inputs.shape[0])

        outputs = model(inputs, calibs, targets_list, img_sizes, dn_args=0)

        # use the criterion's matcher to get matched pred→GT pairs
        outputs_main = {k: v for k, v in outputs.items()
                        if k not in ('aux_outputs', 'inter_outputs')}
        group_num = cfg['model'].get('group_num', 11)
        indices = criterion.matcher(outputs_main, targets_list, group_num=group_num)

        pred_dim = outputs['pred_3d_dim']  # [B, Q, 3]  -> [H, W, L]
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            pred_h = pred_dim[b][src_idx][:, 0].cpu().numpy()
            gt_h   = targets_list[b]['size_3d'][tgt_idx][:, 0].cpu().numpy()
            all_pred_h.append(pred_h)
            all_gt_h.append(gt_h)

    pred_h = np.concatenate(all_pred_h)
    gt_h   = np.concatenate(all_gt_h)

    mse = np.mean((pred_h - gt_h) ** 2)
    mae = np.mean(np.abs(pred_h - gt_h))
    print(f"\nMatched objects : {len(pred_h)}")
    print(f"Height MSE      : {mse:.4f} m²")
    print(f"Height MAE      : {mae:.4f} m")
    print(f"GT  height mean : {gt_h.mean():.4f} m  std: {gt_h.std():.4f} m")
    print(f"Pred height mean: {pred_h.mean():.4f} m  std: {pred_h.std():.4f} m")


if __name__ == '__main__':
    main()
