# Changes: projection-alignment

## Projection Alignment Loss (`lib/models/monodgp/monodgp.py`, `configs/monodgp.yaml`)
- Added `loss_projection_alignment`, which projects predicted 3D bounding boxes into 2D using camera intrinsics and penalises the L1 difference between the projected 2D box and the predicted 2D box. This enforces geometric consistency between the 2D and 3D predictions.
- Passes `calibs` and `img_sizes` through the model output dict to make camera parameters available in the loss.
- Applied only at the final decoder layer (excluded from auxiliary decoder losses).
- Added `proj_align_loss_coef` to `configs/monodgp.yaml`.

## Gradient Accumulation (`lib/helpers/trainer_helper.py`, `configs/monodgp.yaml`)
- Added gradient accumulation support in the trainer (`accumulation_steps: 2` in config).
- Reduced batch size from 8 to 4, keeping effective batch size at 8.
- Added total training time logging at the end of each run.
