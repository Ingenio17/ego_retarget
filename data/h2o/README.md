# H2O runs: rebuild them with your own H2O access

Five egocentric clips from the [H2O dataset](https://h2odataset.ethz.ch/) (two hands manipulating objects, ETH Zurich,
ICCV 2021), turned into the same per-run format as the EgoVerse runs in
[`data/egoverse/trace`](../egoverse/trace/README.md): original frames, per-frame 21-joint hand skeletons,
fingertips, wrists and camera poses (`data.npz`), MANO hands (`mano_fit.npz`), metadata with H2O's action labels,
videos with the skeletons drawn on, and a floating 3D view of the hands in the world frame.

**No H2O data is in this repository.** H2O's terms of use forbid passing the dataset on, so this folder holds only:

| file | what it is |
|---|---|
| `reproduce_h2o.py` | single-file script: downloads the needed part of H2O with **your** login, builds every run, checks the result |
| `requirements.txt` | every Python package the script needs, with exact versions |
| `MANIFEST.sha256` | sha256 of every file the script should produce (checksums only, no content) |
| `<run>/README.md` | what each run contains: source sequence, task name, H2O's action segments, file and array descriptions |
| `.gitignore` | keeps everything the script writes out of git |

After the script has run, each `<run>/` folder also holds the data files, and `data/h2o/index.csv` lists the runs. Do not
commit, push or send those files to anyone who has not accepted H2O's terms themselves.

## Runs

| run folder | H2O sequence | object | task label | frames | seconds | action segments | size after build |
|---|---|---|---|---|---|---|---|
| [`apply_lotion`](apply_lotion/README.md) | subject 1, `h1/3` | lotion | apply lotion | 929 | 31.0 | 8 | 1.1 GB |
| [`read_espresso`](read_espresso/README.md) | subject 1, `o1/7` | espresso | read espresso | 768 | 25.6 | 5 | 0.9 GB |
| [`pour_milk`](pour_milk/README.md) | subject 1, `o1/1` | milk | pour milk | 745 | 24.8 | 5 | 0.9 GB |
| [`apply_spray`](apply_spray/README.md) | subject 1, `o2/5` | spray | apply spray | 486 | 16.2 | 6 | 0.5 GB |
| [`take_out_cappuccino`](take_out_cappuccino/README.md) | subject 1, `k1/0` | cappuccino | take out cappuccino | 351 | 11.7 | 4 | 0.4 GB |

All runs are whole H2O sequences from the egocentric camera `cam4`, 1280 x 720 at 30 fps; 3.7 GB in total, almost all
of it the original PNG frames. They were picked as the most natural of subject 1's first 25 sequences in the archive:
both hands busy, many actions, fast hands or head motion, and the most home-like table (`k1`).

**Naming rule.** H2O labels every frame with one of 36 actions (verb + object, such as `pour milk`) or background. A run
is named after its main action: the longest action segment that is not a plain `place ...` or `grab ...` (ties: the
earlier one), with spaces written as underscores. If two requested sequences get the same name, both get
`_s1_<scene>_<seq>` appended (for example `pour_milk_s1_o1_1`). Each run's README lists the chosen label and every action
segment with its frames.

## Setup on a fresh machine

### 1. Prerequisites

- Linux x86_64 (tested on Ubuntu 24.04).
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended; it installs Python 3.11.16 by itself), or
  CPython 3.11 with `venv` and `pip`.
- Disk: about 25 GB free while the script runs (the extracted sequences are kept in `data/h2o/.cache/` until the build
  ends, then deleted); 3.7 GB at the end.
- RAM: about 4 GB.
- Optional, for `mano_fit.npz` and the 3D view: the MANO model (step 3) and a GPU with an EGL driver (pyrender renders
  offscreen through EGL; any recent NVIDIA driver works). Without them those files are skipped and everything else is
  still built.

### 2. Register for H2O (required)

1. Open [h2odataset.ethz.ch](https://h2odataset.ethz.ch/), read the terms of use, enter your first name, last name,
   email and affiliation, and click "I Agree".
2. ETH emails you a **user name and password**. They are **valid for 7 days**; register again when they expire.
3. Give them to the script in one of two ways (they are never printed or stored by the script):
   - a netrc file, readable only by you:
     ```bash
     printf 'machine h2odataset.ethz.ch login <user> password <password>\n' > ~/.h2o_netrc
     chmod 600 ~/.h2o_netrc
     ```
     and pass `--netrc ~/.h2o_netrc`;
   - or environment variables: `export H2O_USER=<user> H2O_PASSWORD=<password>`.

Key terms you accept (quoted from the registration page): the dataset "is to be used only for the academic purposes",
"will not be used for commercial purposes", "will not be transferred to any third party", and "any publication based
on, or containing, the DATASET shall include a reference to the Data set". ETH can end your right to use it at any time,
after which you must delete it.

### 3. Register for MANO (optional, for `mano_fit.npz` and the 3D view)

1. Create an account at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/register.php) and accept the MANO licence
   (free for non-commercial research).
2. On the [download page](https://mano.is.tue.mpg.de/download.php), download **Models & Code** (`mano_v1_2.zip`) and
   unzip it.
3. Pass the folder that contains `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` with `--mano`, i.e. `mano_v1_2/models`.

### 4. Install and run

Clone the repository and run from its root:

```bash
git clone https://github.com/Ingenio17/ego_retarget.git
cd ego_retarget
```

**With uv** (reads the pinned dependency list inside the script, creates the environment itself):

```bash
uv run data/h2o/reproduce_h2o.py --netrc ~/.h2o_netrc --mano /path/to/mano_v1_2/models
```

**With pip:**

```bash
python3.11 -m venv .venv
.venv/bin/pip install --no-deps -r data/h2o/requirements.txt
.venv/bin/python data/h2o/reproduce_h2o.py --netrc ~/.h2o_netrc --mano /path/to/mano_v1_2/models
```

`--no-deps` is needed because `requirements.txt` already lists every package including the indirect ones, and one of
them is pinned on purpose to a newer version than another package declares (pyrender 0.1.45 asks for PyOpenGL 3.1.0;
we use 3.1.10, which works with current EGL drivers). Without `--no-deps`, pip refuses the combination.

Leave out `--mano` to skip `mano_fit.npz` and the 3D views.

### Options

| option | default | meaning |
|---|---|---|
| `--netrc FILE` | – | netrc file with your H2O login (or set `H2O_USER` / `H2O_PASSWORD`) |
| `--mano DIR` | – | MANO model folder (`MANO_LEFT.pkl`, `MANO_RIGHT.pkl`); optional |
| `--clips S ...` | `h1/3 o1/7 o1/1 o2/5 k1/0` | subject-1 sequences as `<scene>/<seq>`; other sequences are named by the rule above and built the same way, but only the default five have checksums in `MANIFEST.sha256` |
| `--out DIR` | `data/h2o/` | where the run folders are written |
| `--cache DIR` | `<out>/.cache` | where the extracted sequences go; complete sequences already there are not downloaded again, so an interrupted run resumes |
| `--keep-cache` | off | keep the extracted sequences after a successful run |
| `--build-only` | off | no download: build from sequences already extracted in `--cache` (`<cache>/subject1/<scene>/<seq>/cam4/`) |
| `--workers N` | 80 | parallel download requests |
| `--jobs N` | 5 | runs built in parallel |

### What happens and how long it takes

1. **Download.** H2O ships each subject's egocentric data as one 37 GB `.tar.gz` that can only be read from the start.
   The script reads it in order with 80 parallel range requests, keeps only the files the build needs (RGB frames,
   hand and camera poses, MANO annotation, action and object labels, the first five depth maps) and stops after the
   last requested sequence. The default runs need the first **20 GB** (18.7 GiB). Speed depends on the H2O server:
   about **9 min** at 38 MB/s on 2026-10-08, **45 min** at 7.5 MB/s on 2026-10-07 (a single connection gives about
   0.12 MB/s).
2. **Build** (about 2 min with 5 parallel jobs): `data.npz`, `mano_fit.npz`, the frames, `video.mp4`, `overlay.mp4`,
   `meta.json` and each run's `README.md`.
3. **3D views** (about 1-2 min on a GPU).
4. **Check** every file against `MANIFEST.sha256`, then delete `data/h2o/.cache/`.

A complete run took 12.6 min on 2026-10-08 (8.7 min of it downloading). It ends with one line per file and `ALL OK`,
or `SOME FILES DIFFER` and a non-zero exit code:

| files | how they are checked |
|---|---|
| `frames/*.png` | exact: byte for byte (they are H2O's own files) |
| `data.npz`, `mano_fit.npz`, `meta.json`, `video.mp4`, `overlay.mp4`, `README.md`, `index.csv` | byte-identical on the reference machine; if the bytes differ (other CPU or ffmpeg build), checked by content against your own download: array keys, shapes and self-consistency, `skeleton_cam` equal to H2O's `hand_pose`, action segments recomputed, video PSNR against the frames |
| `<run>_world3d_only.mp4` | never byte-identical (GPU rendering differs between runs); checked by frame count, size and that every sampled wrist and fingertip was drawn where `data.npz` puts it. Skipped without `--mano` |

## Data format

Every run has the same files, `.npz` keys and conventions as the EgoVerse runs in
[`data/egoverse/trace`](../egoverse/trace/README.md), so code written for one reads the other:

```python
import numpy as np
d = np.load("data/h2o/pour_milk/data.npz")
tips = d["fingertips_world"]                                 # (T, 2, 5, 3) metres; axis 1: 0 = left, 1 = right
wrist = d["wrist_pos_world"][:, 1]                           # right wrist, (T, 3)
speed = np.linalg.norm(d["wrist_vel_world"][:, 1], axis=-1)  # m/s
```

Differences from the EgoVerse runs:

- **Frames are PNG** (H2O's original format), not JPEG.
- **Wrist orientation** (`wrist_quat_world`) is the rotation of H2O's MANO root, because H2O ships no tracker wrist
  frame; align the two wrist frames before comparing orientations across datasets.
- **World frame:** H2O's calibrated world rotated so +z is the table normal and the table is z = 0 ("up" is measured from
  the depth maps; for EgoVerse it is inferred). The 4x4 transform is in `meta.json` (`world_from_h2o_world`).
- **Labels** are MANO fits to multi-view RGB-D in a lab, not tracker estimates; the length of each run is the whole
  H2O sequence (12-31 s) instead of a 20 s window.

Each run's README has the full key tables.

## Troubleshooting

- **`H2O server refused the login (HTTP 401)`:** the H2O password has expired (7 days); register again at
  [h2odataset.ethz.ch](https://h2odataset.ethz.ch/) and update the netrc file or variables.
- **Slow download:** the H2O server's speed varies from day to day. Interrupt and rerun later: finished sequences in
  `data/h2o/.cache/` are kept and not downloaded again. Fewer parallel requests (`--workers 20`) can help on a weak link.
- **`skipped (built without --mano)`** for `mano_fit.npz` and `<run>_world3d_only.mp4`: pass `--mano` with the folder
  that contains `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`.
- **EGL / OpenGL errors in step 3:** the machine has no GPU with an EGL driver; run without `--mano`, or on a machine
  with one.
- **`bytes differ; ... yes`** lines are fine: the file was checked by content and passed. Only `MISMATCH` lines mean a
  real difference.

## Terms of use and citation

The data the script downloads and builds stays under H2O's terms of use (see step 2); it is not openly licensed. The
script, the manifest and these READMEs contain no H2O data. Cite H2O in any publication that uses it:

```bibtex
@inproceedings{kwon2021h2o,
  title     = {H2O: Two Hands Manipulating Objects for First Person Interaction Recognition},
  author    = {Kwon, Taein and Tekin, Bugra and St{\"u}hmer, Jan and Bogo, Federica and Pollefeys, Marc},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```

Project page: [taeinkwon.com/projects/h2o](http://www.taeinkwon.com/projects/h2o) · paper:
[arXiv:2104.11181](https://arxiv.org/abs/2104.11181). MANO: [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/) (own
licence, non-commercial research).
