# ego_retarget

Retargeting human hand motion from egocentric (first-person) video to robot hands in simulation.

Planned contents:

- Sample egocentric clips with 3D hand skeletons (21 joints per hand) from H2O and Ego-Exo4D, in one shared input format.
- Tools to visualise a skeleton as a floating 3D hand next to its video.
- Retargeting from the human skeleton to the robot hands in `urgantry_sim`.

## Data

- [`data/egoverse_trace/`](data/egoverse_trace/README.md): one 20 s EgoVerse clip (appliance repair) with original
  frames, per-frame 21-joint skeletons, fingertips, wrists, camera poses and a MANO fit, in `.npz`.

## Submodules

`urgantry_sim` (https://github.com/leo01110111/urgantry_sim) is the simulator (bimanual UR7e arms with Wuji hands).
It follows that repo's `main` branch.

Clone with the submodule:

```bash
git clone --recurse-submodules https://github.com/Ingenio17/ego_retarget.git
```

Update the simulator to the latest `main`:

```bash
git submodule update --remote urgantry_sim
```
