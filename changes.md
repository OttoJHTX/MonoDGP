# Changes: height-residual-estimation

## Height Bin + Residual Head (`lib/models/monodgp/monodgp.py`, `lib/models/monodgp/height_bins.py`)
- Replaced direct absolute height prediction with a bin+residual scheme. Height is encoded as a bin index into 5 GMM-derived bins plus a within-bin residual.
- Added `height_bin_embed` MLP (`MLP(256, 256, NUM_HEIGHT_BINS*2, 2)`), outputting 5 bin classification logits and 5 per-bin residuals. During training, height is decoded as a soft weighted sum over bins; during inference a hard argmax is used.
- `dim_embed_3d` output reduced from 3 to 2 (W and L only), with H now handled entirely by `height_bin_embed`.
- Added `lib/models/monodgp/height_bins.py` defining the 5 bin edges and centers, derived from GMM fitting on KITTI Car heights (k=5, selected by BIC/AIC).

## Height Bin Loss (`lib/models/monodgp/monodgp.py`, `configs/monodgp.yaml`)
- Added `loss_height_bins` combining a cross-entropy classification loss on the bin index and an L1 regression loss on the within-bin residual (supervised only at the ground truth bin).
- Added `height_bin_loss_coef: 1` to `configs/monodgp.yaml`.

## Dataset (`lib/datasets/kitti/kitti_dataset.py`)
- Added `height_bin` and `height_res` target fields, encoding each object's 3D height as a bin index and residual offset from the bin center.

## Inference Decoding (`lib/helpers/decode_helper.py`)
- Updated `extract_dets_from_outputs` to hard-decode height from the predicted bin logits and residuals using argmax, replacing the predicted H in `size_3d`.

## Gradient Accumulation (`lib/helpers/trainer_helper.py`, `configs/monodgp.yaml`)
- Added gradient accumulation support in the trainer (`accumulation_steps: 2` in config).
- Reduced batch size from 8 to 4, keeping effective batch size at 8.
- Added total training time logging at the end of each run.
