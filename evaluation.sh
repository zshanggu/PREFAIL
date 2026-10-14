MODEL_PATH=${1:-saved_model/best_model_prefail_weighted.pth}

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py \
  --model-path "$MODEL_PATH" --split test --real-dir data \
  --output-dir results/test --wandb-disabled
