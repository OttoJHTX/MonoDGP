GPU_IDS=$(python -c "import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print(cfg['trainer']['gpu_ids'])" $1)
CUDA_VISIBLE_DEVICES=$GPU_IDS python tools/train_val.py --config $@
