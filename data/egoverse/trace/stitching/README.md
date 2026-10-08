# EgoVerse trace run: stitching

One 20-second egocentric clip with per-frame 3D hand skeletons, for testing retargeting from human hands to robot hands.
Part of the [EgoVerse trace set](../README.md).

| | |
|---|---|
| source | [EgoVerse](https://egoverse.ai/) ([paper](https://arxiv.org/abs/2604.07607)), lab `trace` (own head camera, not Aria) |
| episode | `2026-06-01-05-40-20-542686` (`processed_v3/trace/2026-06-01-05-40-20-542686.zarr`, 739 frames) |
| task label | `stitching` |
| task description | align and sew blue fabric with sewing machine |
| window | episode frames 69-668 (600 frames) |
| rate, length | 30 fps, 20.0 s |
| image | 1280 x 720 |

Both hands are labelled and in view in all 600 frames.

## Task label and annotations

EgoVerse gives the whole episode one task label, `stitching` ("align and sew blue fabric with sewing machine"), and annotates it with
step segments. These are the 10 segments that overlap this window. Episode frames are the dataset's own
numbering (end exclusive); clip frames are rows of the arrays here (frame 0 = episode frame 69), clipped to 0-600.

| # | annotation | episode frames | clip frames |
|---|---|---|---|
| 1 | fold blue fabric piece on sewing table with both hands AND align blue fabric piece on sewing table with both hands | 30-90 | 0-21 |
| 2 | position blue fabric piece under sewing machine needle with both hands AND align thread with right hand AND lower sewing machine presser foot with right hand | 90-175 | 21-106 |
| 3 | thread sewing machine with right hand AND lower sewing machine presser foot with right hand | 175-210 | 106-141 |
| 4 | slide blue fabric piece with left hand AND shift blue fabric piece on blue fabric piece with right hand | 210-244 | 141-175 |
| 5 | position blue fabric pieces with both hands | 244-302 | 175-233 |
| 6 | align blue fabric pieces under sewing machine needle with both hands | 302-360 | 233-291 |
| 7 | align blue fabric pieces on sewing table with both hands AND sew blue fabric pieces with sewing machine with both hands | 360-466 | 291-397 |
| 8 | sew blue fabric pieces with sewing machine with both hands | 466-555 | 397-486 |
| 9 | sew blue fabric pieces with sewing machine with both hands | 555-633 | 486-564 |
| 10 | sew blue fabric pieces with sewing machine with both hands | 633-739 | 564-600 |

The same list, unclipped, is in `meta.json` under `annotations`.

**Label caveat.** The hand and head poses are EgoVerse's shipped labels: estimates from the lab's own hand and head
trackers, not motion capture.

**Quality checks we ran** (no ground truth exists, so these are consistency checks):

- MANO fits the labelled joints to a mean of 0.42 mm (left) and 0.47 mm (right).
- Bone lengths are steady: the median over bones of each bone's frame-to-frame spread is 0.08 mm (left) and 0.08 mm (right).
- No head motion leaks into the world-frame hands: fast wrist wobble is 5.53 mm in the world frame vs 6.1 mm in the
  camera frame (left) and 11.82 vs 12.0 mm (right), with head-to-wrist correlation -0.04 and -0.05.
- Continuity: continuous: no cuts in the window or the episode; the annotation "sew blue fabric pieces with sewing machine with both hands" repeats at episode frames 555 and 633 with no image change at either boundary, so it is one take split by the annotator.

## Files

| file | what it is |
|---|---|
| `frames/0000.jpg` ... `frames/0599.jpg` | the original EgoVerse JPEG frames, byte for byte; frame `i` is row `i` of every array |
| `video.mp4` | the same frames as H.264 (CRF 18, yuv420p), for viewing |
| `overlay.mp4` | the video with both 21-joint skeletons drawn on it (left hand blue, right hand green), CRF 18 |
| `stitching_world3d_only.mp4` | floating 3D view in the world frame: MANO hand meshes over the skeletons, the head camera as a pyramid, a fixed virtual camera, 1280 x 960, CRF 18 |
| `data.npz` | per-frame skeleton, fingertip, wrist and camera arrays (below) |
| `mano_fit.npz` | MANO hand model fitted to the world-frame skeletons (below) |
| `meta.json` | clip details, conventions and the dataset's step annotations for this window |
| `LICENSE` | CC BY-SA 4.0 licence text |

## `data.npz`

T = 600. Axis 1 of every `(T, 2, ...)` array is the hand: 0 = left, 1 = right. float32 unless noted. Missing values
would be NaN, never zeros (none are missing in this clip).

| key | shape | units | frame | meaning |
|---|---|---|---|---|
| `timestamps` | (T,) | s | – | time from the first frame |
| `frame_index` | (T,) int32 | – | – | frame number in the full EgoVerse episode |
| `skeleton_world` | (T, 2, 21, 3) | m | world | 21 joints per hand |
| `skeleton_cam` | (T, 2, 21, 3) | m | camera | the same joints in the head camera's frame |
| `skeleton_px` | (T, 2, 21, 2) | px | image | the joints projected with `K` |
| `fingertips_world` | (T, 2, 5, 3) | m | world | thumb, index, middle, ring, pinky tips (joints 4, 8, 12, 16, 20) |
| `fingertips_cam` | (T, 2, 5, 3) | m | camera | the same, camera frame |
| `wrist_pos_world` | (T, 2, 3) | m | world | wrist position (joint 0) |
| `wrist_pos_cam` | (T, 2, 3) | m | camera | the same, camera frame |
| `wrist_quat_world` | (T, 2, 4) | – | world | wrist orientation `[qw, qx, qy, qz]`, wrist-to-world |
| `wrist_vel_world` | (T, 2, 3) | m/s | world | central difference of the wrist position; NaN at the first and last frame |
| `fingertip_vel_world` | (T, 2, 5, 3) | m/s | world | central difference of each fingertip; NaN at the first and last frame |
| `valid` | (T, 2) bool | – | – | hand labelled |
| `in_view` | (T, 2) bool | – | – | labelled, wrist more than 5 cm in front of the camera and inside the image |
| `head_pose_world` | (T, 4, 4) | m | world | head camera pose, camera-to-world |
| `K` | (3, 3) | px | – | pinhole intrinsics |
| `image_size` | (2,) int32 | px | – | `[width, height]` |

**Joint order** (21 per hand): 0 wrist; thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20, each from base to
tip.

**Frames.** World: the episode's own world frame as EgoVerse ships it. Its +z axis appears to point up (the head sits
above the wrists along +z and looks along -z), but EgoVerse does not document this. Camera: OpenCV (x right, y down,
z forward). With `head_pose_world = [R t; 0 0 0 1]`:

$$
x_{\text{cam}} = R^\top (x_{\text{world}} - t)
$$

## `mano_fit.npz`

MANO v1.2 fitted to `skeleton_world`, one shape per hand for the clip. For `<s>` = `left`, `right`:

| key | shape | units | meaning |
|---|---|---|---|
| `<s>_betas` | (10,) | – | MANO shape |
| `<s>_global_orient` | (T, 3) | rad | wrist rotation, axis-angle, world frame |
| `<s>_transl` | (T, 3) | m | MANO translation, world frame |
| `<s>_hand_pose` | (T, 45) | rad | 15 finger joints, axis-angle, absolute (`flat_hand_mean=True` in `smplx.MANO`) |
| `<s>_joints21_world` | (T, 21, 3) | m | the fitted hand's 21 joints in the order above (MANO's 16 joints + fingertip vertices 744, 320, 443, 554, 671) |
| `<s>_err` | (T, 21) | mm | distance from each fitted joint to the label |

Mean fit error: left 0.42 mm, right 0.47 mm. The MANO model files are not included; get them from
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (free research licence).

## Loading

Install the pinned packages from the repo root with `pip install -r requirements.txt`, then:

```python
import numpy as np
d = np.load("data/egoverse/trace/stitching/data.npz")
tips = d["fingertips_world"]                                 # (600, 2, 5, 3) metres
wrist = d["wrist_pos_world"][:, 1]                           # right wrist, (600, 3)
speed = np.linalg.norm(d["wrist_vel_world"][:, 1], axis=-1)  # m/s
```

## Licence and attribution

The data in this folder is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/);
the full text is in [`LICENSE`](LICENSE).

**Source.** EgoVerse ([egoverse.ai](https://egoverse.ai/), [paper](https://arxiv.org/abs/2604.07607),
[code](https://github.com/GaTech-RL2/EgoVerse)), recorded by the `trace` lab, episode `2026-06-01-05-40-20-542686`,
released by EgoVerse under CC BY-SA 4.0 as listed on the episode's page in the EgoVerse browser:
[partners.mecka.ai/egoverse](https://partners.mecka.ai/egoverse?lab=trace&search=2026-06-01-05-40-20-542686#explorer)
(checked 2026-10-08).

**Changes made from the source.** We cut a 600-frame window (episode frames 69-668); copied its JPEG frames unchanged;
encoded `video.mp4` from them; drew the shipped hand labels onto the frames (`overlay.mp4`); converted the shipped hand,
wrist and head poses into the arrays of `data.npz` (camera-frame copies, pixel projections, fingertip subsets,
velocities); fitted MANO to the skeletons (`mano_fit.npz`); and rendered the world-frame 3D view
(`stitching_world3d_only.mp4`). All of these derived files are shared under the same CC BY-SA 4.0 licence.

The MANO model itself is not part of this folder and has its own licence
([mano.is.tue.mpg.de](https://mano.is.tue.mpg.de)); `mano_fit.npz` holds only fitted parameters.
