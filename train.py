"""Train the PREFAIL risk-prediction model.

Each epoch:
1. Refresh: run every training episode window by window with the current model (eval mode, clean
   images, no gradients) and record the past-risk input each window gets, exactly as at evaluation.
2. Train on shuffled batches of windows from all episodes with those past-risk inputs. Each window is
   trained on the risk `lookahead` windows later, with past-risk dropout and color jitter.
Validation runs every `eval_every` epochs; the checkpoint with the best weighted score (averaged over
the last `score_avg_window` validations) is kept, or the final epoch with --save-last.
"""
import argparse
import os
import random
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from common import (color_jitter_windows, combined_classification_mse_loss, compute_train_stats,
                    drop_past_risk, init_past_risk_buffer, inject_past_risk, update_past_risk_buffer)
from dataset import create_dataset, parse_data_sources
from evaluate import evaluate_model_by_episode
from model import FUSIONS, build_model, load_model

REFRESH_GROUP = 64  # episodes run side by side in the past-risk refresh pass


def stack(windows, key, device):
    return torch.stack([w[key] for w in windows]).to(device)


def refresh_past_risk(model, dataset, episode_ids, window_size, device):
    """Past-risk input (shared_state with the model's own previous predictions) for every
    (episode, window index). Episodes run side by side for speed; nothing is trained here."""
    past_risks = {}
    model.eval()
    with torch.no_grad():
        for g in range(0, len(episode_ids), REFRESH_GROUP):
            group = episode_ids[g:g + REFRESH_GROUP]
            episodes = [dataset.get_episode(ep) for ep in group]
            buffers = [init_past_risk_buffer(window_size) for _ in group]
            for t in range(max(len(windows) for windows in episodes)):
                active = [i for i, windows in enumerate(episodes) if t < len(windows)]
                batch = [episodes[i][t]['window_data'] for i in active]
                shared = [inject_past_risk(w['shared_state'], buffers[i]) for i, w in zip(active, batch)]
                probs = model(stack(batch, 'states_cam1', device), stack(batch, 'states_cam2', device),
                              torch.stack(shared).to(device),
                              stack(batch, 'rgb_cam1', device), stack(batch, 'rgb_cam2', device))
                for i, shared_state, prob in zip(active, shared, probs.view(-1).tolist()):
                    past_risks[(group[i], t)] = shared_state
                    update_past_risk_buffer(buffers[i], prob)
    model.train()
    return past_risks


def train(model, train_dataset, val_dataset, model_name, save_dir='saved_model', val_output_dir='results/val',
          num_epochs=30, warmup_epochs=3, stop_epoch=15, initial_lr=3e-4, min_lr=1e-6, pretrained_lr_scale=0.1,
          batch_size=32, grad_clip=1.0, window_size=5, risk_threshold=0.5, classification_weight=0.0,
          pos_weight=None, lookahead=2, past_risk_dropout=0.5, eval_every=5, score_avg_window=3,
          smoothing_type='none', smoothing_window=5, smoothing_alpha=0.3, save_last=False):
    """Train the model and return the selected checkpoint loaded for evaluation.

    stop_epoch: stop after this many epochs while keeping the num_epochs learning-rate schedule (None = all).
    pos_weight: BCE positive-class weight; None = n_neg / n_pos over the training windows.
    save_last: keep the final-epoch model instead of the best validation checkpoint (for training on all data).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    train_episodes = train_dataset.get_episode_ids()
    if not train_episodes:
        raise ValueError("Training dataset contains no episodes.")
    if val_dataset is None or not val_dataset.get_episode_ids():
        raise ValueError("Validation dataset contains no episodes.")
    print(f"Training with {len(train_episodes)} episodes, validating with {len(val_dataset.get_episode_ids())}")

    # Training-set statistics: state standardization (stored in the model) and BCE pos_weight
    stats = compute_train_stats(train_dataset, risk_threshold=risk_threshold)
    model.set_input_stats(stats['cam1_mean'], stats['cam1_std'], stats['cam2_mean'], stats['cam2_std'],
                          stats['robot_mean'], stats['robot_std'])
    if pos_weight is None:
        pos_weight = stats['pos_weight']
    print(f"Training windows: {stats['n_pos']} positive / {stats['n_neg']} negative, BCE pos_weight = {pos_weight:.2f}")
    model = model.to(device)
    pos_weight = torch.tensor([pos_weight], device=device)

    # ImageNet-pretrained image trunk trains at lr * pretrained_lr_scale
    pretrained_ids = {id(p) for p in model.pretrained_parameters()}
    optimizer = optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) not in pretrained_ids], 'lr_scale': 1.0},
        {'params': list(model.pretrained_parameters()), 'lr_scale': pretrained_lr_scale},
    ], lr=initial_lr, weight_decay=0.01)

    def get_lr(progress):
        """Linear warmup then cosine decay; progress is measured in (fractional) epochs."""
        if progress < warmup_epochs:
            return min_lr + (initial_lr - min_lr) * (progress / warmup_epochs)
        decay = min(1.0, (progress - warmup_epochs) / max(num_epochs - warmup_epochs, 1e-8))
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * decay))

    def save(path):
        torch.save(model.state_dict(), path)
        wandb.save(path)

    best_score, best_mse, recent_scores = -float('inf'), float('inf'), []
    last_epoch = min(stop_epoch, num_epochs) - 1 if stop_epoch else num_epochs - 1
    epoch_pbar = tqdm(range(last_epoch + 1), desc="Training Progress")
    for epoch in epoch_pbar:
        past_risks = refresh_past_risk(model, train_dataset, train_episodes, window_size, device)
        keys = list(past_risks)
        random.shuffle(keys)
        batches = [keys[b:b + batch_size] for b in range(0, len(keys), batch_size)]

        epoch_loss = epoch_mse = 0.0
        train_pbar = tqdm(batches, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
        for step, batch_keys in enumerate(train_pbar):
            episodes = {ep: train_dataset.get_episode(ep) for ep in {ep for ep, _ in batch_keys}}
            batch = [episodes[ep][t]['window_data'] for ep, t in batch_keys]
            # target: risk `lookahead` windows later in the same episode (last window near the end)
            targets = torch.stack([episodes[ep][min(t + lookahead, len(episodes[ep]) - 1)]['window_data']['risk']
                                   for ep, t in batch_keys]).to(device)
            shared_state = torch.stack([past_risks[k] for k in batch_keys]).to(device)

            lr = get_lr(epoch + step / len(batches))
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_scale']
            optimizer.zero_grad()
            logits = model(stack(batch, 'states_cam1', device), stack(batch, 'states_cam2', device),
                           drop_past_risk(shared_state, past_risk_dropout),
                           color_jitter_windows(stack(batch, 'rgb_cam1', device)),
                           color_jitter_windows(stack(batch, 'rgb_cam2', device)), return_logits=True)
            loss = combined_classification_mse_loss(logits, targets, risk_threshold=risk_threshold,
                                                    pos_weight=pos_weight, classification_weight=classification_weight)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += nn.functional.mse_loss(torch.sigmoid(logits.detach()), targets.view_as(logits)).item()
            train_pbar.set_postfix({'batch_loss': f"{loss.item():.4f}", 'lr': f"{lr:.6f}"})

        wandb.log({'epoch': epoch + 1, 'learning_rate': lr,
                   'train_loss': epoch_loss / len(batches), 'train_mse': epoch_mse / len(batches)})

        if (epoch + 1) % eval_every == 0 or epoch == last_epoch:
            results = evaluate_model_by_episode(model, val_dataset, f"{model_name}_val", output_dir=val_output_dir,
                                                risk_threshold=risk_threshold, smoothing_type=smoothing_type,
                                                smoothing_window=smoothing_window, smoothing_alpha=smoothing_alpha,
                                                window_size=window_size)
            model.train()
            recent_scores.append(results['score'])
            avg_score = float(np.mean(recent_scores[-score_avg_window:]))
            wandb.log({'epoch': epoch + 1, 'val_mse': results['mse'], 'val_weighted_score': results['score'],
                       'val_weighted_score_avg': avg_score,
                       **{f'val_{k}': results[k] for k in ['precision', 'recall', 'f1_score',
                                                           'TP', 'TN', 'FP1', 'FP2', 'FN']},
                       'val_average_execute_rate': results['average_execute_rate']})

            save(os.path.join(save_dir, f'intermediate_model_{model_name}_{epoch}.pth'))
            if avg_score > best_score:
                best_score = avg_score
                save(os.path.join(save_dir, f'best_model_{model_name}_weighted.pth'))
                print(f"New best model (weighted score, mean of last {score_avg_window}: {avg_score:.4f})")
            if results['mse'] < best_mse:
                best_mse = results['mse']
                save(os.path.join(save_dir, f'best_model_{model_name}_mse.pth'))
            epoch_pbar.set_postfix({'val_mse': f"{results['mse']:.4f}", 'val_score_avg': f"{avg_score:.4f}"})

    if save_last:
        path = os.path.join(save_dir, f'last_model_{model_name}.pth')
        torch.save(model.state_dict(), path)
        print(f"Saved final-epoch model to {path}")
        return model.eval()

    path = os.path.join(save_dir, f'best_model_{model_name}_weighted.pth')
    print(f"Loading best model (weighted score) from {path}")
    return load_model(path, window_size=window_size)


def main():
    parser = argparse.ArgumentParser(description='Train the PREFAIL risk prediction model')
    # data
    parser.add_argument('--real-dir', type=str, default='data', help='Real data directory (with train/val/test)')
    parser.add_argument('--sim-dir', type=str, default='history_data/simulation', help='Simulated data directory')
    parser.add_argument('--train-sources', type=str, default='real', help='e.g. "real", "sim", "sim+real"')
    parser.add_argument('--val-sources', type=str, default='real', help='e.g. "real", "sim", "sim+real"')
    parser.add_argument('--test-sources', type=str, default='real', help='e.g. "real", "sim", "sim+real"')
    parser.add_argument('--cache-dir', type=str, default='./dataset_cache', help='Dataset cache directory')
    parser.add_argument('--clear-cache', action='store_true', help='Delete the dataset cache first')
    parser.add_argument('--chunk-size', type=int, default=20, help='Episodes per cache chunk')
    parser.add_argument('--window-size', type=int, default=5, help='Frames per window')
    parser.add_argument('--stride', type=int, default=1, help='Window stride')
    # model
    parser.add_argument('--fusion', type=str, default='transformer', choices=FUSIONS,
                        help='Fusion of the encoder outputs: transformer (default) or a 1D ResNet over the frames')
    # optimization
    parser.add_argument('--batch-size', type=int, default=32, help='Windows per training batch')
    parser.add_argument('--num-epochs', type=int, default=30, help='Length of the learning-rate schedule')
    parser.add_argument('--stop-epoch', type=int, default=15,
                        help='Stop after this many epochs, keeping the --num-epochs schedule (0 = run all)')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Linear learning-rate warmup epochs')
    parser.add_argument('--initial-lr', type=float, default=3e-4, help='Peak learning rate (AdamW)')
    parser.add_argument('--pretrained-lr-scale', type=float, default=0.1,
                        help='Learning-rate multiplier for the ImageNet-pretrained image encoder')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Max gradient norm (0 = off)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # objective
    parser.add_argument('--risk-threshold', type=float, default=0.5, help='Risk threshold for failure')
    parser.add_argument('--classification-weight', type=float, default=0.0,
                        help='Loss = w * BCE(risk > threshold) + (1 - w) * MSE(risk)')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='BCE positive-class weight (default: n_neg / n_pos over training windows)')
    parser.add_argument('--lookahead', type=int, default=2, help='Train each window on the risk this many windows later')
    parser.add_argument('--past-risk-dropout', type=float, default=0.5,
                        help='Probability of zeroing the past-risk input of a training sample')
    # validation / output
    parser.add_argument('--eval-every', type=int, default=5, help='Validate and checkpoint every N epochs')
    parser.add_argument('--smoothing-type', type=str, default='none',
                        choices=['none', 'moving_average', 'exponential', 'median'], help='Prediction smoothing')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Window for moving average/median smoothing')
    parser.add_argument('--smoothing-alpha', type=float, default=0.3, help='Exponential smoothing factor (0, 1]')
    parser.add_argument('--save-last', action='store_true',
                        help='Keep the final-epoch model (training on all data, no held-out selection)')
    parser.add_argument('--save-dir', type=str, default='saved_model', help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory for evaluation outputs')
    parser.add_argument('--wandb-disabled', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.clear_cache and os.path.exists(args.cache_dir):
        import shutil
        print(f"Clearing cache directory: {args.cache_dir}")
        shutil.rmtree(args.cache_dir)

    base_dirs = {'real': args.real_dir, 'sim': args.sim_dir}
    datasets = {split: create_dataset(base_dirs, split, parse_data_sources(sources), window_size=args.window_size,
                                      stride=args.stride, cache_dir=args.cache_dir, chunk_size=args.chunk_size)
                for split, sources in [('train', args.train_sources), ('val', args.val_sources),
                                       ('test', args.test_sources)]}
    if datasets['train'] is None or datasets['val'] is None:
        sys.exit("No training or validation data found.")

    model_name = 'prefail' if args.fusion == 'transformer' else f'prefail_{args.fusion}'
    wandb.init(project="robot-trajectory-prediction",
               name=f"{model_name}_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
               config=vars(args), mode="disabled" if args.wandb_disabled else None)

    model = train(build_model(window_size=args.window_size, fusion=args.fusion), datasets['train'], datasets['val'], model_name,
                  save_dir=args.save_dir, val_output_dir=os.path.join(args.output_dir, 'val'),
                  num_epochs=args.num_epochs, warmup_epochs=args.warmup_epochs, stop_epoch=args.stop_epoch or None,
                  initial_lr=args.initial_lr, pretrained_lr_scale=args.pretrained_lr_scale,
                  batch_size=args.batch_size, grad_clip=args.grad_clip, window_size=args.window_size,
                  risk_threshold=args.risk_threshold, classification_weight=args.classification_weight,
                  pos_weight=args.pos_weight, lookahead=args.lookahead, past_risk_dropout=args.past_risk_dropout,
                  eval_every=args.eval_every, smoothing_type=args.smoothing_type,
                  smoothing_window=args.smoothing_window, smoothing_alpha=args.smoothing_alpha,
                  save_last=args.save_last)

    if datasets['test'] is not None:
        results = evaluate_model_by_episode(model, datasets['test'], model_name,
                                            output_dir=os.path.join(args.output_dir, 'test'),
                                            risk_threshold=args.risk_threshold, smoothing_type=args.smoothing_type,
                                            smoothing_window=args.smoothing_window,
                                            smoothing_alpha=args.smoothing_alpha, window_size=args.window_size)
        print(f"Test MSE: {results['mse']:.4f}")
        wandb.log({"test_mse": results['mse']})


if __name__ == "__main__":
    main()
