# Changes: improve_3d_head

## 3D Dimension Head Capacity (`lib/models/monodgp/monodgp.py`)
- Increased `dim_embed_3d` MLP depth from 2 to 4 layers (`MLP(256, 256, 3, 4)`), giving the 3D dimension prediction head more representational capacity.

## Gradient Accumulation (`lib/helpers/trainer_helper.py`, `configs/monodgp.yaml`)
- Added gradient accumulation support in the trainer (`accumulation_steps: 2` in config).
- Reduced batch size from 8 to 4, keeping effective batch size at 8.
- Added total training time logging at the end of each run.
