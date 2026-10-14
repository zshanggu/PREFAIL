CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --real-dir data --seed 0 \
  --save-dir saved_model --output-dir results --wandb-disabled
