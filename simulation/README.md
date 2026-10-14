# Simulation: historical backtracking data generation

Generates labeled failure-prediction episodes in MuJoCo. A UR5e carries an object on a panel; when the object
fails (slides off), `src/simulate_backtracking.py` binary-searches for the latest e-stop time that still avoids
the failure and labels the `failure_phase_value` of the frames in between.

## Layout

```
simulation/
├── run_simulation.sh                 # Entry point: simulates every .obj of one object type
├── src/
│   ├── simulate_backtracking.py      # Simulation + binary-search backtracking labeling
│   ├── mujoco_base.py                # MuJoCo/GLFW base class
│   └── sim_utils.py                  # Contour processing, failure phase, interpolation, config reading
├── tools/
│   ├── update_scene_xml.py           # Writes model/ur5e/scene_<type>_tmp.xml with the current object mesh
│   ├── update_sim_config.py          # Writes configs/simulation_<type>_tmp.json for the current object
│   ├── list_processed_objects.py     # Lists objects that already have finished episodes
│   ├── cleanup_unprocessed.py        # Removes partial raw outputs of unfinished objects
│   └── process_raw_episodes.py       # Raw episodes → training format (risk ramp, relative-motion features)
├── configs/
│   ├── simulation.json               # Simulation / camera / randomization parameters
│   ├── objects/<type>.json           # Per-object mass, size, friction, pose for each object type
│   └── trajectories/traj.txt           # UR5e joint waypoints
├── model/ur5e/
│   ├── scene.xml                     # Scene: robot + panel + object
│   ├── LICENSE
│   └── assets/
│       ├── robot/                    # UR5e and panel meshes
│       └── objects/<type>/           # Object meshes: boxes, sem, mujoco, ddg, core-*
└── data/                             # Simulation output (git-ignored)
```

## Object meshes

Everything in `model/ur5e/assets/objects/` except `boxes/` (~410 MB in total) is git-ignored. Copy the
object sets in before running; `run_simulation.sh` reads `.obj` files from `objects/<OBJ_TYPE>/`.

## Usage

```bash
pip install -r requirements.txt
./run_simulation.sh
```

Set `OBJ_TYPE` (e.g. `sem`, `mujoco`, `core-mug`, `boxes`) at the top of `run_simulation.sh`.

Output, all under `data/`:
- `data/train_raw/raw_data/<type>/`: raw episodes (from `save_path` + `dataset_type` in the config)
- `data/processed/<type>/`: finished episodes; objects found here are skipped on the next run

### Converting raw episodes to the training format

Raw episodes hold the failure label (`failure_phase_value`) and contours, but not the model inputs:

```bash
python3 tools/process_raw_episodes.py --input-dir data/train_raw/raw_data/sem --output-dir data/processed_sem
```

- `process_raw_episodes.py` builds the `risk` ramp (0 → 1 between the failure-phase start and the failure;
  `--interpolation hermite` by default) and `state_to_init_relative` (panel-minus-object motion per camera from
  the contours, plus end-effector position, joint angles and joint speeds). Episodes without a failure keep their
  first 150 frames; episodes whose failure phase starts before frame 5 are skipped.
- Joint speeds and end-effector quaternions are taken directly from MuJoCo (`qvel`, `xquat`), so they need no
  correction (the real-world recordings had their joint speeds recomputed from joint angles at 30 fps).

Then split the processed episodes into `train/`, `val/` and `test/` folders and train with
`--sim-dir <folder> --train-sources sim` (or `sim+real`), see the top-level README.

### Single object

After `run_simulation.sh` has written the `_tmp` files:

```bash
python3.10 src/simulate_backtracking.py --config configs/simulation_sem_tmp.json --obj_type sem
```
