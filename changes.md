# Changes: auxiliary_height_loss

## Auxiliary Height Loss (`lib/models/monodgp/monodgp.py`, `configs/monodgp.yaml`)
- Added `loss_height`, an explicit L1 loss on the predicted 3D height (H) from `dim_embed_3d`. This directly penalises height errors, since H is critical for geometric depth estimation via `Z = f * H / h_bbox`.
- Applied only at the final decoder layer (excluded from auxiliary decoder losses).
- Added `height_loss_coef: 1` to `configs/monodgp.yaml`.

## Gradient Accumulation (`lib/helpers/trainer_helper.py`, `configs/monodgp.yaml`)
- Added gradient accumulation support in the trainer (`accumulation_steps: 2` in config).
- Reduced batch size from 8 to 4, keeping effective batch size at 8.
- Added total training time logging at the end of each run.
