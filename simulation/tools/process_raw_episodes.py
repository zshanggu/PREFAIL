#!/usr/bin/env python3
"""Convert raw episodes (from src/simulate_backtracking.py, or real-world recordings with the same fields)
into the training format used by PREFAIL.

Per episode:
1. Risk label: 'risk' is built from 'failure_phase_value'. The first frame with value 0.5 (failure phase
   start) and the first with 1.0 (failure reached) anchor an interpolated ramp from 0 to 1
   (--interpolation); frames after the failure stay at 1.0. 'risk_delta' is the frame-to-frame change.
2. Features, from the object / panel contours of both cameras (2D rigid transform per frame, theta, x, y):
   state_to_init_relative = [panel - object motion since the first frame, top camera (3) and front camera (3),
   end-effector position (3), joint angles (6), joint velocities (6)]  (21 values, used by the model);
   also state_to_last* (relative to the previous frame), state_to_init and *_accumulated variants.
3. Frame 0 (no motion yet) is dropped. Episodes without a failure keep their first --max-safe-steps frames;
   episodes whose failure phase starts before frame --min-failure-start are skipped.

Joint speeds and quaternions come straight from MuJoCo and need no correction.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from sim_utils import (hermite_spline_interpolation, linear_interpolation, piecewise_bezier_interpolation,  # noqa: E402
                       process_consecutive_frames, resample_data, spline_interpolation)


def label_risk(frames):
    """Set 'risk' from 'failure_phase_value'; returns (failure start index, failure reached index), -1 if absent."""
    start = reach = -1
    for i, frame in enumerate(frames):
        value = frame['failure_phase_value']
        frame['risk'] = value
        if value == 0.5 and start == -1:
            start = i
        if value == 1.0 and reach == -1:
            reach = i
    return start, reach


def interpolate_risk(frames, start, reach, interpolation):
    """Replace the step label with a smooth 0 → 1 ramp anchored at the failure start (0.5) and reach (1.0)."""
    if interpolation == "linear":
        for i, value in enumerate(linear_interpolation(reach, start), start=start):
            frames[i]['risk'] = np.asarray([value], dtype=np.float32)
        return
    if interpolation == "spline":
        for i in range(start):
            frames[i]['risk'] = np.asarray([0.0], dtype=np.float32)
        values = spline_interpolation(0, start, reach)
    elif interpolation == "bezier":
        values = piecewise_bezier_interpolation(0, start, reach)
    else:  # hermite
        values = hermite_spline_interpolation(0, start, reach)
    for i in range(reach + 1):
        frames[i]['risk'] = np.asarray([values[i]], dtype=np.float32)
    for i in range(reach + 1, len(frames)):
        frames[i]['risk'] = np.asarray([1.0], dtype=np.float32)


def add_features(frames, n_frames):
    """Add the state_* feature vectors to frames 1..n_frames-1. Returns False if a contour is missing."""
    contour_keys = ['object_top_contour', 'gripper_top_contour', 'object_front_contour', 'gripper_front_contour']
    accumulated = None  # running sums of [top_to_last, front_to_last, top_to_init, front_to_init]
    for i in range(1, n_frames):
        cur, prev, first = frames[i], frames[i - 1], frames[0]
        if any(np.asarray(f[k]).shape[0] == 0 for k in contour_keys for f in (cur, prev)):
            print(f"  Warning: empty contour at frame {i}, stopping feature extraction for this episode")
            return False
        # theta, x, y transform of each contour relative to the previous / first frame
        to_last = [np.asarray(process_consecutive_frames(prev[k], cur[k]), dtype=np.float32) for k in contour_keys]
        to_init = [np.asarray(process_consecutive_frames(first[k], cur[k]), dtype=np.float32) for k in contour_keys]
        robot = [cur['end_effector_pos'], cur['robot_joint_angles'], cur['robot_joint_velocities']]
        # panel (gripper) motion minus object motion, per camera: the object's slip relative to the panel
        rel_last = [to_last[1] - to_last[0], to_last[3] - to_last[2]]
        rel_init = [to_init[1] - to_init[0], to_init[3] - to_init[2]]
        accumulated = [r.copy() for r in rel_last + rel_init] if accumulated is None else \
            [a + r for a, r in zip(accumulated, rel_last + rel_init)]

        features = {
            'state_to_last': np.concatenate(to_last + robot),
            'state_to_last_relative': np.concatenate(rel_last + robot),
            'state_to_init': np.concatenate(to_init + robot),
            'state_to_init_relative': np.concatenate(rel_init + robot),
            'state_to_last_relative_accumulated': np.concatenate(accumulated[:2] + robot),
            'state_to_init_relative_accumulated': np.concatenate(accumulated[2:] + robot),
        }
        if features['state_to_last'].shape[0] != 15 + 12:
            raise ValueError(f"Unexpected feature size {features['state_to_last'].shape[0]} at frame {i}")
        for key, value in features.items():
            cur[key] = np.asarray(value, dtype=np.float32)
    return True


def process_episode(path, interpolation, min_failure_start, max_safe_steps):
    """Processed frames of one raw episode, or None if the episode is skipped."""
    frames = np.load(path, allow_pickle=True)
    start, reach = label_risk(frames)
    if start != -1 and start < min_failure_start:
        print(f"  Skipped: failure phase starts at frame {start} < {min_failure_start}")
        return None
    n_frames = min(max_safe_steps, len(frames)) if start == -1 else len(frames)
    if start == -1:
        print(f"  No failure: keeping the first {n_frames} frames")

    add_features(frames, n_frames)
    for i in range(n_frames):  # projected frames are not used for training
        frames[i].pop('projected_top_frame_rgb', None)
        frames[i].pop('projected_front_frame_rgb', None)

    if start != -1 and reach == -1:
        print("  Failure phase starts but never reaches 1.0: keeping the raw risk label")
    elif start != -1:
        interpolate_risk(frames, start, reach, interpolation)

    for i in range(n_frames):
        risk = np.asarray(frames[i]['risk']).flatten()[0]
        prev = np.asarray(frames[i - 1]['risk']).flatten()[0] if i > 0 else risk
        frames[i]['risk_delta'] = np.asarray([risk - prev], dtype=np.float32)

    episode = resample_data(frames[1:n_frames], cut=True, scale=1)  # frame 0 has no motion features
    missing = sum('state_to_last' not in f or 'risk' not in f for f in episode)
    if missing or len(episode) == 0:
        print(f"  Not saved: {missing} frames without features (of {len(episode)})")
        return None
    return episode


def main():
    parser = argparse.ArgumentParser(description='Convert raw episodes into the PREFAIL training format')
    parser.add_argument('--input-dir', required=True, help='Directory with raw episode .npy files')
    parser.add_argument('--output-dir', required=True, help='Directory for processed episodes')
    parser.add_argument('--interpolation', default='hermite', choices=['linear', 'spline', 'bezier', 'hermite'],
                        help='Risk ramp between failure start and failure (default: hermite)')
    parser.add_argument('--min-failure-start', type=int, default=5,
                        help='Skip episodes whose failure phase starts before this frame')
    parser.add_argument('--max-safe-steps', type=int, default=150, help='Frames kept from episodes without failure')
    parser.add_argument('--overwrite', action='store_true', help='Reprocess episodes that already have an output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.npy'))
    saved = 0
    for name in files:
        out_path = os.path.join(args.output_dir, name)
        if not args.overwrite and os.path.exists(out_path):
            print(f"{name}: already processed, skipping")
            continue
        print(name)
        episode = process_episode(os.path.join(args.input_dir, name), args.interpolation,
                                  args.min_failure_start, args.max_safe_steps)
        if episode is not None:
            np.save(out_path, episode)
            saved += 1
    print(f"Saved {saved} processed episodes to {args.output_dir}")


if __name__ == "__main__":
    main()
