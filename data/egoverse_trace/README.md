# EgoVerse trace runs

Egocentric clips from the `trace` lab of [EgoVerse](https://egoverse.ai/) ([paper](https://arxiv.org/abs/2604.07607)),
each with the original frames and per-frame 3D hand skeletons (21 joints per hand), fingertips, wrists, camera poses
and a MANO fit, for testing retargeting from human hands to robot hands. Each run is one continuous take (no cuts),
600 frames at 30 fps.

| run folder | episode | task label | task description | frames | length | annotations in window |
|---|---|---|---|---|---|---|
| [`appliance_repair/`](appliance_repair/README.md) | `2026-07-02-02-41-51-338984` | `appliance_repair` | remove fan assembly from motherboard | 600 | 20.0 s | 7 |
| [`stitching/`](stitching/README.md) | `2026-06-01-05-40-20-542686` | `stitching` | align and sew blue fabric with sewing machine | 600 | 20.0 s | 10 |
| [`sewing/`](sewing/README.md) | `2026-07-02-09-08-54-842528` | `sewing` | sew fabric | 600 | 20.0 s | 4 |

Each run's README lists its task label, task description and every step annotation in the window, with quality checks
and its licence and attribution.

**Naming.** Each run folder is named after the episode's EgoVerse task label (`task` in the episode metadata). If two
runs ever share a task label, the folder is `<task>_<episode hash>`.

## Common format

Every run folder has the same files:

| file | what it is |
|---|---|
| `frames/0000.jpg` ... `frames/0599.jpg` | the original EgoVerse JPEG frames, byte for byte; frame `i` is row `i` of every array |
| `video.mp4` | the same frames as H.264 (CRF 18), for viewing |
| `overlay.mp4` | the video with both 21-joint skeletons drawn on it (left hand blue, right hand green), CRF 18 |
| `<task>_world3d_only.mp4` | floating 3D view in the world frame (MANO meshes over the skeletons, head camera pyramid), 1280 x 960 |
| `data.npz` | per-frame skeleton, fingertip, wrist and camera arrays |
| `mano_fit.npz` | MANO hand model fitted to the world-frame skeletons |
| `meta.json` | clip details, conventions and the dataset's step annotations |
| `README.md`, `LICENSE` | run description, licence (CC BY-SA 4.0) and attribution |

The keys, shapes and units of `data.npz` and `mano_fit.npz` are identical across runs and documented in each run's
README. Conventions: axis 1 of every `(T, 2, ...)` array is the hand (0 = left, 1 = right); joints are 0 wrist, then
thumb, index, middle, ring, pinky (4 each, base to tip); world positions are in metres in the episode's own world frame
(+z appears to point up); the camera frame is OpenCV (x right, y down, z forward); `head_pose_world` is camera-to-world.

## Loading

Install the pinned packages from the repo root with `pip install -r requirements.txt`, then:

```python
import numpy as np
for run in ["appliance_repair", "stitching", "sewing"]:
    d = np.load(f"data/egoverse_trace/{run}/data.npz")
    tips = d["fingertips_world"]                                 # (600, 2, 5, 3) metres
    speed = np.linalg.norm(d["wrist_vel_world"][:, 1], axis=-1)  # right wrist speed, m/s
    print(run, tips.shape, float(np.nanmedian(speed)))
```

## Licence

All data here is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), from EgoVerse; see each
run's `LICENSE` and its "Licence and attribution" section for the source episode and the changes we made.
