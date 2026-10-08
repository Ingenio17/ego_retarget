# EgoVerse trace clip: hand skeletons, fingertips and wrists

One 20-second egocentric clip with per-frame 3D hand skeletons, for testing retargeting from human hands to robot hands.

| | |
|---|---|
| source | [EgoVerse](https://egoverse.ai/) ([paper](https://arxiv.org/abs/2604.07607)), lab `trace` (own head camera, not Aria) |
| episode | `2026-07-02-02-41-51-338984` (`processed_v3/trace/2026-07-02-02-41-51-338984.zarr`) |
| task | `appliance_repair`: remove the fan assembly from a motherboard |
| window | episode frames 1486-2085 (600 frames) |
| rate, length | 30 fps, 20.0 s |
| image | 1280 x 720 |

Both hands are labelled and in view in all 600 frames.

**Label caveat.** The hand and head poses are EgoVerse's shipped labels: estimates from the lab's own hand and head
trackers, not motion capture. Bone lengths wobble by about 2 mm from frame to frame.

## Files

| file | what it is |
|---|---|
| `frames/0000.jpg` ... `frames/0599.jpg` | the original EgoVerse JPEG frames, byte for byte; frame `i` is row `i` of every array |
| `video.mp4` | the same frames as H.264 (CRF 18, yuv420p), for viewing |
| `overlay.mp4` | the video with both 21-joint skeletons drawn on it (left hand blue, right hand green), CRF 18 |
| `trace_world3d_only.mp4` | floating 3D view in the world frame: MANO hand meshes over the skeletons, the head camera as a pyramid, a fixed virtual camera, 1280 x 960, CRF 18 |
| `data.npz` | per-frame skeleton, fingertip, wrist and camera arrays (below) |
| `mano_fit.npz` | MANO hand model fitted to the world-frame skeletons (below) |
| `meta.json` | clip details, conventions and the dataset's action annotations for this window |

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
0.41-0.55 m above the wrists along +z and looks along -z), but EgoVerse does not document this. Camera: OpenCV (x right,
y down, z forward). With `head_pose_world = [R t; 0 0 0 1]`:

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

Mean fit error: left 1.6 mm, right 1.6 mm. The MANO model files are not included; get them from
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (free research licence).

## Loading

```python
import numpy as np
d = np.load("data/egoverse_trace/data.npz")
tips = d["fingertips_world"]                                 # (600, 2, 5, 3) metres
wrist = d["wrist_pos_world"][:, 1]                           # right wrist, (600, 3)
speed = np.linalg.norm(d["wrist_vel_world"][:, 1], axis=-1)  # m/s
```

## Licence and attribution

The data in this folder is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/);
the full text is in [`LICENSE`](LICENSE).

**Source.** EgoVerse ([egoverse.ai](https://egoverse.ai/), [paper](https://arxiv.org/abs/2604.07607),
[code](https://github.com/GaTech-RL2/EgoVerse)), recorded by the `trace` lab, episode `2026-07-02-02-41-51-338984`,
released by EgoVerse under CC BY-SA 4.0.

**Changes made from the source.** We cut a 600-frame window (episode frames 1486-2085); copied its JPEG frames unchanged;
encoded `video.mp4` from them; drew the shipped hand labels onto the frames (`overlay.mp4`); converted the shipped hand,
wrist and head poses into the arrays of `data.npz` (camera-frame copies, pixel projections, fingertip subsets,
velocities); fitted MANO to the skeletons (`mano_fit.npz`); and rendered the world-frame 3D view
(`trace_world3d_only.mp4`). All of these derived files are shared under the same CC BY-SA 4.0 licence.

The MANO model itself is not part of this folder and has its own licence
([mano.is.tue.mpg.de](https://mano.is.tue.mpg.de)); `mano_fit.npz` holds only fitted parameters.
