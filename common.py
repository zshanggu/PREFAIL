"""Training helpers: loss, autoregressive past-risk input, augmentation and training-set statistics."""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from model import N_ROBOT_STATE


def combined_classification_mse_loss(logits, targets, risk_threshold=0.5, pos_weight=None,
                                     classification_weight=0.5, reduction='mean'):
    """BCE on (targets > risk_threshold) weighted by classification_weight, plus MSE on the continuous
    risk weighted by 1 - classification_weight.

    logits: model outputs before sigmoid, (N, 1); targets: risk in [0, 1], (N, 1).
    """
    targets = targets.view_as(logits)
    binary_targets = (targets > risk_threshold).float()
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)(logits, binary_targets)
    mse_loss = nn.MSELoss(reduction=reduction)(torch.sigmoid(logits), targets)
    return classification_weight * classification_loss + (1.0 - classification_weight) * mse_loss


# ---- Autoregressive past-risk input ----
# shared_state rows N_ROBOT_STATE: hold the past (window_size - 1) risk values. Training and evaluation
# both fill them with the model's own previous predictions, never ground truth.
PAST_RISK_ROW = N_ROBOT_STATE


def init_past_risk_buffer(window_size):
    """Per-episode buffer of the model's previous predictions; zeros = no risk at episode start."""
    n_past = window_size - 1
    return deque([0.0] * n_past, maxlen=n_past)


def inject_past_risk(shared_state, buffer):
    """Return a copy of shared_state (19, 1) with its past-risk rows replaced by the buffer."""
    shared = shared_state.clone()
    for i, val in enumerate(buffer):
        shared[PAST_RISK_ROW + i, 0] = float(val)
    return shared


def update_past_risk_buffer(buffer, raw_pred):
    """Append the raw (unsmoothed) prediction for the window just processed."""
    buffer.append(float(raw_pred))


def drop_past_risk(shared_state, p):
    """Past-risk dropout: zero the past-risk rows of each sample with probability p.

    shared_state: (batch, 19, 1). Training only; forces the model to detect risk from the sensors
    instead of extrapolating its own previous predictions.
    """
    if p <= 0:
        return shared_state
    keep = (torch.rand(shared_state.shape[0], 1, 1, device=shared_state.device) >= p).to(shared_state.dtype)
    out = shared_state.clone()
    out[:, PAST_RISK_ROW:] = out[:, PAST_RISK_ROW:] * keep
    return out


# ---- Training-time image augmentation ----
def color_jitter_windows(rgb, brightness=0.2, contrast=0.2, saturation=0.2):
    """Color jitter with one random setting per window, shared by all its frames.

    rgb: (batch, frames, 3, H, W) in [0, 1]. Applied during training only; cached images are clean.
    """
    out = []
    for window in rgb:
        window = TF.adjust_brightness(window, float(np.random.uniform(1 - brightness, 1 + brightness)))
        window = TF.adjust_contrast(window, float(np.random.uniform(1 - contrast, 1 + contrast)))
        window = TF.adjust_saturation(window, float(np.random.uniform(1 - saturation, 1 + saturation)))
        out.append(window.clamp(0, 1))
    return torch.stack(out)


# ---- Training-set statistics ----
def compute_train_stats(train_dataset, risk_threshold=0.5):
    """Per-channel mean/std of the state inputs over all training windows, and the BCE
    positive-class weight (n_neg / n_pos)."""
    cam1, cam2, robot, risks = [], [], [], []
    for episode_id in train_dataset.get_episode_ids():
        for window in train_dataset.get_episode(episode_id):
            w = window['window_data']
            cam1.append(w['states_cam1'].T)
            cam2.append(w['states_cam2'].T)
            robot.append(w['shared_state'][:N_ROBOT_STATE, 0])
            risks.append(w['risk'].reshape(-1))

    def mean_std(rows):
        x = torch.cat(rows).float() if rows[0].dim() == 2 else torch.stack(rows).float()
        return x.mean(0), x.std(0).clamp_min(1e-6)

    risks = torch.cat(risks)
    n_pos = int((risks > risk_threshold).sum())
    n_neg = int(risks.numel() - n_pos)
    cam1_mean, cam1_std = mean_std(cam1)
    cam2_mean, cam2_std = mean_std(cam2)
    robot_mean, robot_std = mean_std(robot)
    return {
        'cam1_mean': cam1_mean, 'cam1_std': cam1_std,
        'cam2_mean': cam2_mean, 'cam2_std': cam2_std,
        'robot_mean': robot_mean, 'robot_std': robot_std,
        'pos_weight': n_neg / max(n_pos, 1), 'n_pos': n_pos, 'n_neg': n_neg,
    }
