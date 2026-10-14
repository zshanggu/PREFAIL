"""Episode datasets: each episode file (.npy, a list of frames) becomes a list of sliding windows.

Processed windows are cached on disk in chunks of episodes (pickle files) so the raw episodes are
only processed once; chunks are loaded on demand and kept in memory.
"""
import hashlib
import os
import pickle
from typing import List

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# Bump when cached window contents change, so stale caches are not reused
CACHE_VERSION = "v2_clean_images_clipped_risk"
IMAGE_SIZE = 64

_image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # uint8 (H, W, C) → float (C, H, W) in [0, 1]
])


def parse_data_sources(source_string: str) -> List[str]:
    """Parse a data source string like 'sim+real' into a list of sources."""
    return [s.strip() for s in source_string.split('+')]


def process_window(frames):
    """Turn a window of raw frames into model inputs.

    Each frame's 'state_to_init_relative' holds [RT_cam1(3), RT_cam2(3), EE_pos(3), joint_angles(6),
    joint_speeds(6)]. Returns:
        states_cam1, states_cam2: (3, T)  relative motion per camera over the window
        shared_state:             (19, 1) current-frame EE + joint angles + joint speeds, then the
                                  ground-truth risk of the previous T-1 frames (replaced by the model's
                                  own predictions during training and evaluation)
        rgb_cam1, rgb_cam2:       (T, 3, 64, 64) top / front camera frames in [0, 1]
        risk:                     (1,) risk of the last frame (the target)
    """
    states = np.stack([frame['state_to_init_relative'] for frame in frames])  # (T, 21)
    risk = np.clip([np.asarray(frame['risk']).flatten()[0] for frame in frames], 0.0, 1.0)
    shared_state = np.concatenate([states[-1, 6:21], risk[:-1]])[:, None]

    def images(key):
        return torch.stack([_image_transform(frame[key]) for frame in frames])

    return {
        'states_cam1': torch.FloatTensor(states[:, 0:3]).T,
        'states_cam2': torch.FloatTensor(states[:, 3:6]).T,
        'shared_state': torch.FloatTensor(shared_state),
        'rgb_cam1': images('full_top_frame_rgb'),
        'rgb_cam2': images('full_front_frame_rgb'),
        'risk': torch.FloatTensor([risk[-1]]),
    }


def process_episode(frames, window_size, stride):
    """All windows of one episode; window_idx is the first frame, time_step the last."""
    return [{'window_data': process_window(frames[start:start + window_size]),
             'window_idx': start,
             'time_step': start + window_size - 1}
            for start in range(0, len(frames) - window_size + 1, stride)]


class CachedEpisodeDataset:
    """Windows of all episodes in one directory, cached on disk in chunks of chunk_size episodes."""

    def __init__(self, data_dir, dataset_name, window_size=5, stride=1, cache_dir="./dataset_cache",
                 chunk_size=50, max_cached_chunks=100):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.max_cached_chunks = max_cached_chunks
        self.chunk_cache = {}
        self.episode_to_chunk_map = {}  # episode id → (chunk file, id within chunk)

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = os.path.join(cache_dir, f"{self._cache_key()}_intermediate")
        mapping_file = os.path.join(self.cache_dir, "episode_mapping.pkl")
        if os.path.exists(mapping_file):
            print(f"Loading cached windows from {self.cache_dir}")
            with open(mapping_file, 'rb') as f:
                self.episode_to_chunk_map = pickle.load(f)['episode_to_chunk_map']
        else:
            self._build_cache(mapping_file)
        print(f"{dataset_name}: {len(self.episode_to_chunk_map)} episodes")

    def _cache_key(self):
        # The literal "False" and "multimodal" fields keep the key identical to earlier versions of
        # this code, so existing caches stay valid. Episode file times detect data changes.
        params = f"{self.window_size}_{self.stride}_False_{self.dataset_name}_multimodal_{CACHE_VERSION}"
        times = sorted(str(os.path.getmtime(os.path.join(self.data_dir, f)))
                       for f in os.listdir(self.data_dir) if f.endswith('.npy'))
        params += "_" + "_".join(times)
        return f"{self.dataset_name}_{hashlib.md5(params.encode()).hexdigest()}"

    def _build_cache(self, mapping_file):
        files = sorted(f for f in os.listdir(self.data_dir) if f.endswith('.npy'))
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Caching {len(files)} episodes from {self.data_dir} in chunks of {self.chunk_size}")
        episode_id = 0
        for chunk_idx, start in enumerate(range(0, len(files), self.chunk_size)):
            chunk_file = os.path.join(self.cache_dir, f"chunk_{chunk_idx:04d}.pkl")
            if os.path.exists(chunk_file):  # left by an interrupted build
                with open(chunk_file, 'rb') as f:
                    chunk = pickle.load(f)
            else:
                chunk = {'episodes': {}, 'episode_lengths': {}, 'episode_filenames': {}}
                chunk_files = files[start:start + self.chunk_size]
                for local_id, name in enumerate(tqdm(chunk_files, desc=f"Chunk {chunk_idx}")):
                    try:
                        frames = np.load(os.path.join(self.data_dir, name), allow_pickle=True)
                    except Exception as e:
                        print(f"Skipping unreadable file {name}: {e}")
                        continue
                    if len(frames) < self.window_size:
                        continue
                    chunk['episodes'][local_id] = process_episode(frames, self.window_size, self.stride)
                    chunk['episode_lengths'][local_id] = len(chunk['episodes'][local_id])
                    chunk['episode_filenames'][local_id] = os.path.splitext(name)[0]
                with open(chunk_file, 'wb') as f:
                    pickle.dump(chunk, f)
            for local_id in chunk['episodes']:
                self.episode_to_chunk_map[episode_id] = (chunk_file, local_id)
                episode_id += 1
        with open(mapping_file, 'wb') as f:
            pickle.dump({'episode_to_chunk_map': self.episode_to_chunk_map,
                         'total_episodes': len(self.episode_to_chunk_map)}, f)

    def _load_chunk(self, chunk_file):
        if chunk_file not in self.chunk_cache:
            if len(self.chunk_cache) >= self.max_cached_chunks:
                del self.chunk_cache[next(iter(self.chunk_cache))]
            with open(chunk_file, 'rb') as f:
                self.chunk_cache[chunk_file] = pickle.load(f)
        return self.chunk_cache[chunk_file]

    def _chunk_entry(self, episode_idx, key):
        chunk_file, local_id = self.episode_to_chunk_map[episode_idx]
        return self._load_chunk(chunk_file)[key][local_id]

    def get_episode_ids(self):
        return list(self.episode_to_chunk_map.keys())

    def get_episode(self, episode_idx):
        """List of windows: {'window_data': model inputs, 'window_idx': first frame, 'time_step': last frame}."""
        return self._chunk_entry(episode_idx, 'episodes')

    def get_episode_length(self, episode_idx):
        return self._chunk_entry(episode_idx, 'episode_lengths')

    def get_episode_filename(self, episode_idx):
        return self._chunk_entry(episode_idx, 'episode_filenames')


class MultiSourceEpisodeDataset:
    """Concatenation of several CachedEpisodeDatasets (e.g. sim + real) behind the same interface."""

    def __init__(self, datasets: List[CachedEpisodeDataset]):
        self.datasets = datasets
        self.window_size = datasets[0].window_size
        self.episode_map = [(i, ep) for i, ds in enumerate(datasets) for ep in ds.get_episode_ids()]
        print(f"Combined {len(datasets)} datasets with {len(self.episode_map)} episodes")

    def get_episode_ids(self):
        return list(range(len(self.episode_map)))

    def get_episode(self, idx):
        i, ep = self.episode_map[idx]
        return self.datasets[i].get_episode(ep)

    def get_episode_length(self, idx):
        i, ep = self.episode_map[idx]
        return self.datasets[i].get_episode_length(ep)

    def get_episode_filename(self, idx):
        i, ep = self.episode_map[idx]
        return f"{self.datasets[i].dataset_name}_{self.datasets[i].get_episode_filename(ep)}"


def create_dataset(base_dirs: dict, split: str, sources: List[str], window_size=5, stride=1,
                   cache_dir="./dataset_cache", chunk_size=50):
    """Dataset for one split ('train' / 'val' / 'test') from <base_dirs[source]>/<split> of each source.

    Returns None if none of the source directories exist.
    """
    datasets = []
    for source in sources:
        if source not in base_dirs:
            print(f"Warning: unknown data source '{source}', skipping")
            continue
        data_dir = os.path.join(base_dirs[source], split)
        if not os.path.isdir(data_dir):
            print(f"Warning: {data_dir} does not exist, skipping")
            continue
        datasets.append(CachedEpisodeDataset(data_dir, f"{source}_{split}", window_size=window_size,
                                             stride=stride, cache_dir=cache_dir, chunk_size=chunk_size))
    if not datasets:
        return None
    return datasets[0] if len(datasets) == 1 else MultiSourceEpisodeDataset(datasets)
