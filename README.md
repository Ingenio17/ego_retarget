# ego_retarget

Retargeting human hand motion from egocentric (first-person) video to robot hands in simulation.

Planned contents:

- Sample egocentric clips with 3D hand skeletons (21 joints per hand) from H2O and Ego-Exo4D, in one shared input format.
- Tools to visualise a skeleton as a floating 3D hand next to its video.
- Retargeting from the human skeleton to the robot hands in `urgantry_sim`.

## Data

- [`data/egoverse_trace/`](data/egoverse_trace/README.md): EgoVerse clips from the `trace` lab, one folder per run, named
  by task label: [`appliance_repair`](data/egoverse_trace/appliance_repair/README.md),
  [`stitching`](data/egoverse_trace/stitching/README.md) and [`sewing`](data/egoverse_trace/sewing/README.md). Each is a
  continuous 20 s take with original frames, per-frame 21-joint skeletons, fingertips, wrists, camera poses and a MANO
  fit in `.npz`, plus the step annotations in its README.

## Setup

To load and view the data: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.

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

## Licence

Data under [`data/`](data/) is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each data folder has its own `LICENSE` and an
attribution section in its README. The `urgantry_sim` submodule is a separate repository with its own licence.
