# H2O run `take_out_cappuccino`: hand skeletons, fingertips and wrists

**The data is not included in this repository** (H2O's terms do not allow passing it on). Run `data/h2o/reproduce_h2o.py`
with your own H2O login to download and build it here; see [`data/h2o/README.md`](../README.md). This README describes what
the script writes into this folder.

One 11.7-second egocentric clip with per-frame 3D hand skeletons, for testing retargeting from human
hands to robot hands. Same file layout, array keys and conventions as the EgoVerse runs in
[`data/egoverse/trace`](../../egoverse/trace/README.md).

| | |
|---|---|
| source | [H2O](https://h2odataset.ethz.ch/) (Kwon et al., ICCV 2021), subject 1, scene `k1`, sequence `0`, egocentric camera `cam4` |
| object | `cappuccino` |
| task name | `take_out_cappuccino` |
| window | the whole sequence, frames 0-350 (351 frames) |
| rate, length | 30 fps, 11.7 s |
| image | 1280 x 720 |

## Task name and annotations

H2O labels every frame of a sequence with one action from its 36-action vocabulary (verb + object, such as
`grab cappuccino`) or background. We name the run after its main action: the longest action segment that is not a
plain `place ...` or `grab ...`, here **`take out cappuccino`**, written `take_out_cappuccino`.

All action segments of the sequence, in clip frames (start inclusive, end exclusive; frames not listed are background).
`meta.json` holds the same list under `annotations`.

| # | H2O action label | frames |
|---|---|---|
| 1 | place cappuccino | 16-68 |
| 2 | grab cappuccino | 68-116 |
| 3 | take out cappuccino | 116-209 |
| 4 | put in cappuccino | 256-321 |

**Label caveat.** The hand and camera poses are H2O's shipped labels: MANO hands fitted to multi-view RGB-D recordings in a
lab (one desk with calibration markers), not motion capture.

## Files

| file | what it is |
|---|---|
| `frames/0000.png` ... `frames/0350.png` | the original H2O RGB frames (PNG), byte for byte; frame `i` is row `i` of every array |
| `video.mp4` | the same frames as H.264 (CRF 18, yuv420p), for viewing |
| `overlay.mp4` | the video with both 21-joint skeletons drawn on it (left hand blue, right hand green), CRF 18 |
| `take_out_cappuccino_world3d_only.mp4` | floating 3D view in the world frame: MANO hand meshes over the skeletons, the head camera as a pyramid, a fixed virtual camera, 1280 x 960, CRF 18 (needs `--mano`) |
| `data.npz` | per-frame skeleton, fingertip, wrist and camera arrays (below) |
| `mano_fit.npz` | H2O's MANO hand annotation, moved into the world frame (below; needs `--mano`) |
| `meta.json` | clip details, conventions, the world transform and H2O's action labels as annotations |

## `data.npz`

T = 351. Axis 1 of every `(T, 2, ...)` array is the hand: 0 = left, 1 = right. float32 unless noted. Missing values
are NaN, never zeros.

| key | shape | units | frame | meaning |
|---|---|---|---|---|
| `timestamps` | (T,) | s | – | time from the first frame |
| `frame_index` | (T,) int32 | – | – | frame number in the H2O sequence |
| `skeleton_world` | (T, 2, 21, 3) | m | world | 21 joints per hand |
| `skeleton_cam` | (T, 2, 21, 3) | m | camera | the same joints in the head camera's frame (H2O's `hand_pose`, unchanged) |
| `skeleton_px` | (T, 2, 21, 2) | px | image | the joints projected with `K` |
| `fingertips_world` | (T, 2, 5, 3) | m | world | thumb, index, middle, ring, pinky tips (joints 4, 8, 12, 16, 20) |
| `fingertips_cam` | (T, 2, 5, 3) | m | camera | the same, camera frame |
| `wrist_pos_world` | (T, 2, 3) | m | world | wrist position (joint 0) |
| `wrist_pos_cam` | (T, 2, 3) | m | camera | the same, camera frame |
| `wrist_quat_world` | (T, 2, 4) | – | world | wrist orientation `[qw, qx, qy, qz]`: the rotation of H2O's MANO root (global orient), moved into the world frame |
| `wrist_vel_world` | (T, 2, 3) | m/s | world | central difference of the wrist position; NaN at the first and last frame |
| `fingertip_vel_world` | (T, 2, 5, 3) | m/s | world | central difference of each fingertip; NaN at the first and last frame |
| `valid` | (T, 2) bool | – | – | hand labelled |
| `in_view` | (T, 2) bool | – | – | labelled, wrist more than 5 cm in front of the camera and inside the image |
| `head_pose_world` | (T, 4, 4) | m | world | head camera pose, camera-to-world |
| `K` | (3, 3) | px | – | pinhole intrinsics |
| `image_size` | (2,) int32 | px | – | `[width, height]` |

**Joint order** (21 per hand): 0 wrist; thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20, each from base to
tip.

**Frames.** World: H2O's world frame (from its calibrated cameras) rotated so that +z is the table normal (up), x is the
mean camera viewing direction along the table, and the table is at z = 0; `meta.json` `world_from_h2o_world` is that
4x4. Unlike EgoVerse's world, "up" here is measured (a plane fit to the table in the depth maps). Camera: OpenCV (x
right, y down, z forward). With `head_pose_world = [R t; 0 0 0 1]`:

$$
x_{\text{cam}} = R^\top (x_{\text{world}} - t)
$$

**Wrist orientation.** EgoVerse ships a tracker wrist frame; H2O does not, so `wrist_quat_world` is MANO's root
rotation. Compare orientations across datasets only after aligning the two wrist frames.

## `mano_fit.npz`

H2O's own MANO annotation (not a new fit), moved from the camera frame into the world frame above, one shape per hand.
For `<s>` = `left`, `right`:

| key | shape | units | meaning |
|---|---|---|---|
| `<s>_betas` | (10,) | – | MANO shape |
| `<s>_global_orient` | (T, 3) | rad | wrist rotation, axis-angle, world frame |
| `<s>_transl` | (T, 3) | m | MANO translation, world frame |
| `<s>_hand_pose` | (T, 45) | rad | 15 finger joints, axis-angle, absolute (`flat_hand_mean=True` in `smplx.MANO`) |
| `<s>_joints21_world` | (T, 21, 3) | m | the MANO hand's 21 joints in the order above (MANO's 16 joints + fingertip vertices 744, 320, 443, 554, 671) |
| `<s>_err` | (T, 21) | mm | distance from each MANO joint to the labelled joint |

The 16 MANO joints match H2O's joints; the fingertips differ by a few millimetres because H2O uses slightly different
tip vertices. **Left hand:** H2O built its left hands with `MANO_LEFT.pkl` exactly as shipped, without the usual fix that
flips the sign of the left model's shape directions (x component); use the left model unchanged to reproduce these
joints. The MANO model files are not included; get them from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (free
research licence).

## Loading

From the repository root, after running the script:

```python
import numpy as np
d = np.load("data/h2o/take_out_cappuccino/data.npz")
tips = d["fingertips_world"]                                 # (351, 2, 5, 3) metres
wrist = d["wrist_pos_world"][:, 1]                           # right wrist, (351, 3)
speed = np.linalg.norm(d["wrist_vel_world"][:, 1], axis=-1)  # m/s
```

## Terms of use

The data files of this run (everything the script writes here; not this README) come from the H2O dataset and stay
under H2O's terms of use ([h2odataset.ethz.ch](https://h2odataset.ethz.ch/), accepted at registration). They are **not**
openly licensed and are not part of the ego_retarget repository. Key terms, quoted from that page:

- the dataset "shall only be downloaded if you agree to these terms";
- it "is to be used only for the academic purposes" and "will not be used for commercial purposes";
- it "will not be transferred to any third party": do not commit, push or send the generated files to anyone who has
  not accepted H2O's terms themselves (`data/h2o/.gitignore` keeps them out of git);
- "any publication based on, or containing, the DATASET shall include a reference to the Data set";
- you "shall further not carry out any procedures with the DATASET (linking, comparison, processing) with which any
  identity of a person could be derived";
- "Upon first request by ETH Zurich the right to use the DATASET will end immediately. You will return or destroy all
  DATASET and confirm the deletion of the DATASET to ETH Zurich."

Cite: Taein Kwon, Bugra Tekin, Jan Stuehmer, Federica Bogo, Marc Pollefeys. *H2O: Two Hands Manipulating Objects for
First Person Interaction Recognition.* ICCV 2021. Project page: http://www.taeinkwon.com/projects/h2o
