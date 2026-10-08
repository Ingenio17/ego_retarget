# ego_retarget

Retargeting human hand motion from egocentric (first-person) video to robot hands in simulation.

Planned contents:

- Egocentric clips with 3D hand skeletons (21 joints per hand) in one shared input format: EgoVerse runs included as
  data, H2O runs rebuilt locally by a script (see below).
- Tools to visualise a skeleton as a floating 3D hand next to its video.
- Retargeting from the human skeleton to the robot hands in `urgantry_sim`.

## Data

- [`data/egoverse/trace/`](data/egoverse/trace/README.md): EgoVerse clips from the `trace` lab, one folder per run, named
  by task label: [`appliance_repair`](data/egoverse/trace/appliance_repair/README.md),
  [`stitching`](data/egoverse/trace/stitching/README.md) and [`sewing`](data/egoverse/trace/sewing/README.md). Each is a
  continuous 20 s take with original frames, per-frame 21-joint skeletons, fingertips, wrists, camera poses and a MANO
  fit in `.npz`, plus the step annotations in its README.

## H2O

[`data/h2o/`](data/h2o/README.md): five clips from the [H2O dataset](https://h2odataset.ethz.ch/) in the same format, one folder
per run named by its main H2O action (`apply_lotion`, `read_espresso`, `pour_milk`, `apply_spray`,
`take_out_cappuccino`). **The H2O data is not included**: H2O's terms forbid passing it on. Register for H2O yourself,
then `uv run data/h2o/reproduce_h2o.py --netrc <file> --mano <dir>` downloads and builds the runs into `data/h2o/` and checks them
against `data/h2o/MANIFEST.sha256`. [`data/h2o/README.md`](data/h2o/README.md) has the full setup; each run's README lists its task
label and H2O action segments.

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
attribution section in its README. Data built by `data/h2o/reproduce_h2o.py` stays under H2O's terms of use and must not be
committed or shared (`data/h2o/.gitignore` keeps it out of git). The `urgantry_sim` submodule is a separate repository with its own licence.
