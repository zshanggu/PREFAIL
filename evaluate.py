"""Episode-level evaluation of early failure prediction.

Each episode is run window by window, feeding the model its own previous predictions as past risk.
An episode is classified by when the prediction first exceeds the risk threshold compared with
when the ground truth does:
    TP  failure predicted no later than the true failure      TN  no failure, none predicted
    FP1 failure predicted, none happened (false alarm)       FN  failure missed
    FP2 failure predicted after the true failure (late)
Execution rate (TP only) = predicted failure time / true failure time; higher means a later warning.
Score = mean of precision, recall, accuracy and average execution rate.
"""
import argparse
import os
import sys
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from common import init_past_risk_buffer, inject_past_risk, update_past_risk_buffer
from dataset import create_dataset, parse_data_sources
from model import load_model

RESULT_CLASSES = ['TP', 'TN', 'FN', 'FP1', 'FP2']


class PredictionSmoother:
    """Online smoothing of per-window predictions: 'none', 'moving_average', 'exponential' or 'median'."""

    def __init__(self, smoothing_type='none', window_size=5, alpha=0.3):
        if smoothing_type not in ('none', 'moving_average', 'exponential', 'median'):
            raise ValueError(f"Unknown smoothing type: {smoothing_type}")
        self.smoothing_type = smoothing_type
        self.window_size = window_size
        self.alpha = alpha
        self.reset()

    def reset(self):
        """Reset the smoother state for a new episode."""
        self.buffer = deque(maxlen=self.window_size)
        self.ema_value = None

    def smooth(self, prediction):
        if self.smoothing_type == 'none':
            return prediction
        if self.smoothing_type == 'exponential':
            self.ema_value = prediction if self.ema_value is None else \
                self.alpha * prediction + (1 - self.alpha) * self.ema_value
            return self.ema_value
        self.buffer.append(prediction)
        return float(np.mean(self.buffer) if self.smoothing_type == 'moving_average' else np.median(self.buffer))


def execution_rate_stats(rates):
    """Mean, median, mode (0.1-wide bins) and std of the execution rates."""
    if not rates:
        return {'mean': 0, 'median': 0, 'mode': 0, 'std': 0}
    rates = np.array(rates)
    hist, edges = np.histogram(rates, bins=np.arange(0, rates.max() + 0.2, 0.1))
    mode_idx = np.argmax(hist)
    return {'mean': rates.mean(), 'median': np.median(rates),
            'mode': (edges[mode_idx] + edges[mode_idx + 1]) / 2, 'std': rates.std()}


def classify_episode(gt_fail_time, pred_fail_time):
    """Episode result and execution rate (TP only) from the first-crossing window indices (None = never)."""
    if gt_fail_time is None:
        return ('TN', None) if pred_fail_time is None else ('FP1', None)
    if pred_fail_time is None:
        return 'FN', None
    if pred_fail_time <= gt_fail_time:
        return 'TP', (pred_fail_time / gt_fail_time if gt_fail_time > 0 else 0.0)
    return 'FP2', None


def save_episode_plot(ep, result, risk_threshold, window_size, output_dir, smoothing_enabled):
    """Plot predictions vs ground truth into <output_dir>/<result>/<episode>.png."""
    plt.rcParams.update({k: 25 for k in ['font.size', 'axes.titlesize', 'axes.labelsize', 'xtick.labelsize',
                                          'ytick.labelsize', 'legend.fontsize', 'figure.titlesize']})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    ax1.plot(ep['time_steps'], ep['predictions'], 'b-', label='Smoothed Predictions' if smoothing_enabled else 'Predictions')
    ax1.plot(ep['time_steps'], ep['ground_truth'], 'r-', label='Ground Truth')
    ax1.axhline(y=risk_threshold, color='g', linestyle='--', label=f'Risk Threshold ({risk_threshold})')
    # a window's index is its first frame; its prediction is plotted at its last frame
    if ep['gt_fail_time'] is not None:
        ax1.axvline(x=ep['gt_fail_time'] + window_size - 1, color='r', linestyle='--', label='GT Failure')
    if ep['pred_fail_time'] is not None:
        ax1.axvline(x=ep['pred_fail_time'] + window_size - 1, color='b', linestyle='--', label='Predicted Failure')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Risk Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if smoothing_enabled:
        ax2.plot(ep['time_steps'], ep['raw_predictions'], 'lightblue', alpha=0.7, label='Raw Predictions')
        ax2.plot(ep['time_steps'], ep['predictions'], 'b-', label='Smoothed Predictions')
        ax2.axhline(y=risk_threshold, color='g', linestyle='--', label=f'Risk Threshold ({risk_threshold})')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Risk Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Raw vs Smoothed Predictions')
    else:
        ax2.set_visible(False)

    plt.tight_layout()
    folder = os.path.join(output_dir, result)
    os.makedirs(folder, exist_ok=True)
    suffix = '_smoothed' if smoothing_enabled else ''
    plt.savefig(os.path.join(folder, f"{ep['episode_filename']}{suffix}.png"))
    plt.close()


def save_episode_npy(ep, result, execution_rate, output_dir, smoothing_enabled):
    """Save per-episode predictions to <output_dir>/npy_results/<result>/<episode>.npy."""
    folder = os.path.join(output_dir, 'npy_results', result)
    os.makedirs(folder, exist_ok=True)
    data = {
        'episode_idx': ep['episode_idx'],
        'episode_filename': ep['episode_filename'],
        'time_steps': np.array(ep['time_steps']),
        'predictions': np.array(ep['predictions']),
        'ground_truth': np.array(ep['ground_truth']),
        'classification_result': result,
        'gt_fail': ep['gt_fail_time'] is not None,
        'pred_fail': ep['pred_fail_time'] is not None,
        'gt_fail_time': ep['gt_fail_time'],
        'pred_fail_time': ep['pred_fail_time'],
        'smoothing_enabled': smoothing_enabled,
    }
    if smoothing_enabled:
        data['raw_predictions'] = np.array(ep['raw_predictions'])
    if execution_rate is not None:
        data['execution_rate'] = execution_rate
    suffix = '_smoothed' if smoothing_enabled else ''
    np.save(os.path.join(folder, f"{ep['episode_filename']}{suffix}.npy"), data)


def predict_episode(model, windows, window_size, smoother, device):
    """Run one episode window by window with autoregressive past risk.
    Returns raw predictions, smoothed predictions and per-window MSE against the current risk."""
    buffer = init_past_risk_buffer(window_size)
    smoother.reset()
    raw, smoothed, mse = [], [], []
    with torch.no_grad():
        for window in windows:
            w = window['window_data']
            prob = model(w['states_cam1'].unsqueeze(0).to(device),
                         w['states_cam2'].unsqueeze(0).to(device),
                         inject_past_risk(w['shared_state'], buffer).unsqueeze(0).to(device),
                         w['rgb_cam1'].unsqueeze(0).to(device),
                         w['rgb_cam2'].unsqueeze(0).to(device))
            value = prob.item()
            update_past_risk_buffer(buffer, value)  # raw (unsmoothed) prediction, as in training
            raw.append(value)
            smoothed.append(smoother.smooth(value))
            mse.append(nn.functional.mse_loss(prob.view(-1).cpu(), w['risk'].view(-1)).item())
    return raw, smoothed, mse


def evaluate_model_by_episode(model, dataset, model_name, output_dir="results", risk_threshold=0.5,
                              smoothing_type='none', smoothing_window=5, smoothing_alpha=0.3, window_size=5):
    """Evaluate every episode of the dataset; saves per-episode plots and .npy files under output_dir
    and returns the metrics (see module docstring)."""
    device = next(model.parameters()).device
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    smoother = PredictionSmoother(smoothing_type, smoothing_window, smoothing_alpha)
    smoothing_enabled = smoothing_type != 'none'

    counts = {k: 0 for k in RESULT_CLASSES}
    execution_rates, all_mse = [], []
    for episode_idx in tqdm(dataset.get_episode_ids(), desc=f"Evaluating episodes with {model_name}"):
        windows = dataset.get_episode(episode_idx)
        raw, smoothed, mse = predict_episode(model, windows, window_size, smoother, device)
        ground_truth = [w['window_data']['risk'].item() for w in windows]
        # first window where ground truth / prediction exceeds the threshold (same ">" as the loss)
        gt_fail_time = next((w['window_idx'] for w, g in zip(windows, ground_truth) if g > risk_threshold), None)
        pred_fail_time = next((w['window_idx'] for w, p in zip(windows, smoothed) if p > risk_threshold), None)
        result, execution_rate = classify_episode(gt_fail_time, pred_fail_time)
        counts[result] += 1
        if execution_rate is not None:
            execution_rates.append(execution_rate)
        all_mse.extend(mse)

        ep = {'episode_idx': episode_idx, 'episode_filename': dataset.get_episode_filename(episode_idx),
              'time_steps': [w['time_step'] for w in windows], 'predictions': smoothed, 'raw_predictions': raw,
              'ground_truth': ground_truth, 'gt_fail_time': gt_fail_time, 'pred_fail_time': pred_fail_time}
        save_episode_npy(ep, result, execution_rate, output_dir, smoothing_enabled)
        save_episode_plot(ep, result, risk_threshold, window_size, output_dir, smoothing_enabled)

    TP, TN, FP1, FP2, FN = (counts[k] for k in ['TP', 'TN', 'FP1', 'FP2', 'FN'])
    n = sum(counts.values())
    precision = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / n if n > 0 else 0
    rates = execution_rate_stats(execution_rates)
    score = 0.25 * (precision + recall + accuracy + rates['mean'])

    log_data = {
        "precision": precision, "recall": recall, "accuracy": accuracy, "f1_score": f1_score,
        "true_positives": TP, "true_negatives": TN, "false_positives_fp1": FP1,
        "false_positives_fp2": FP2, "false_negatives": FN,
        "average_execute_rate": rates['mean'], "median_execute_rate": rates['median'],
        "mode_execute_rate": rates['mode'], "std_execute_rate": rates['std'], "score": score,
    }
    if smoothing_enabled:
        log_data.update({"smoothing_type": smoothing_type, "smoothing_window": smoothing_window,
                         "smoothing_alpha": smoothing_alpha})
    wandb.log(log_data)

    print(f"\nEarly Failure Prediction Metrics:")
    if smoothing_enabled:
        print(f"Smoothing: {smoothing_type} (window={smoothing_window}, alpha={smoothing_alpha})")
    print(f"True Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives FP1 (False Alarm): {FP1}")
    print(f"False Positives FP2 (Late Prediction): {FP2}")
    print(f"False Negatives (FN): {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Average Execution Rate: {rates['mean']:.4f}")
    print(f"Score: {score:.4f}")

    return {
        'mse': float(np.mean(all_mse)), 'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'f1_score': f1_score, 'TP': TP, 'TN': TN, 'FP1': FP1, 'FP2': FP2, 'FN': FN,
        'average_execute_rate': rates['mean'], 'median_execute_rate': rates['median'],
        'mode_execute_rate': rates['mode'], 'std_execute_rate': rates['std'], 'score': score,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained risk prediction model')
    parser.add_argument('--model-path', type=str, required=True, help='Trained model checkpoint (.pth)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--sources', type=str, default='real', help='Data sources, e.g. "real", "sim", "sim+real"')
    parser.add_argument('--real-dir', type=str, default='data', help='Real data directory (with train/val/test)')
    parser.add_argument('--sim-dir', type=str, default='history_data/simulation', help='Simulated data directory')
    parser.add_argument('--cache-dir', type=str, default='./dataset_cache', help='Dataset cache directory')
    parser.add_argument('--chunk-size', type=int, default=20, help='Episodes per cache chunk')
    parser.add_argument('--window-size', type=int, default=5, help='Frames per window')
    parser.add_argument('--stride', type=int, default=1, help='Window stride')
    parser.add_argument('--risk-threshold', type=float, default=0.5, help='Risk threshold for failure')
    parser.add_argument('--smoothing-type', type=str, default='none',
                        choices=['none', 'moving_average', 'exponential', 'median'], help='Prediction smoothing')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Window for moving average/median smoothing')
    parser.add_argument('--smoothing-alpha', type=float, default=0.3, help='Exponential smoothing factor (0, 1]')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory for plots and .npy results')
    parser.add_argument('--wandb-disabled', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    if not 0 < args.smoothing_alpha <= 1:
        raise ValueError("Smoothing alpha must be in range (0, 1]")
    if args.smoothing_window < 1:
        raise ValueError("Smoothing window must be >= 1")

    dataset = create_dataset({'real': args.real_dir, 'sim': args.sim_dir}, args.split,
                             parse_data_sources(args.sources), window_size=args.window_size,
                             stride=args.stride, cache_dir=args.cache_dir, chunk_size=args.chunk_size)
    if dataset is None:
        sys.exit("No evaluation data found.")

    wandb.init(project="robot-trajectory-prediction",
               name=f"prefail_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
               config=vars(args), mode="disabled" if args.wandb_disabled else None)
    model = load_model(args.model_path, window_size=args.window_size)
    results = evaluate_model_by_episode(model, dataset, "prefail", output_dir=args.output_dir,
                                        risk_threshold=args.risk_threshold, smoothing_type=args.smoothing_type,
                                        smoothing_window=args.smoothing_window,
                                        smoothing_alpha=args.smoothing_alpha, window_size=args.window_size)
    print(f"{args.split} MSE: {results['mse']:.4f}")
    wandb.log({f"{args.split}_mse": results['mse']})


if __name__ == "__main__":
    main()
