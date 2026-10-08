# /// script
# requires-python = "==3.11.16"
# dependencies = [
#     "freetype-py==2.5.1",
#     "imageio==2.37.0",
#     "imageio-ffmpeg==0.6.0",
#     "networkx==3.6.1",
#     "numpy==2.2.6",
#     "opencv-python-headless==4.12.0.88",
#     "pillow==11.3.0",
#     "pyglet==2.1.16",
#     "pyopengl==3.1.10",
#     "pyrender==0.1.45",
#     "scipy==1.15.3",
#     "six==1.17.0",
#     "trimesh==4.8.3",
# ]
# [tool.uv]
# override-dependencies = ["pyopengl==3.1.10"]
# ///
"""Rebuild the H2O hand-skeleton runs of the ego_retarget repo (data/h2o/<run>/) from H2O with your own login.

The repository holds no H2O data (H2O's terms forbid passing it on). This script streams subject 1's egocentric
archive from the H2O server (https://h2odataset.ethz.ch/), extracts only the chosen sequences (by default h1/3 o1/7
o1/1 o2/5 k1/0) and writes, next to each run's committed README.md, data/h2o/<run>/:

  frames/0000.png ...            H2O's original cam4 RGB frames, byte for byte
  data.npz                       per-frame 21-joint skeletons, fingertips, wrists, velocities, camera poses
  mano_fit.npz                   H2O's MANO annotation moved into the world frame       (needs --mano)
  meta.json                      clip details, conventions, world transform, H2O's action labels
  video.mp4, overlay.mp4         the frames, and the frames with the skeletons drawn, H.264 CRF 18
  <run>_world3d_only.mp4         floating 3D view of the world-frame hands (pyrender)  (needs --mano)
  README.md                      (rewritten byte-identical to the committed one)
plus data/h2o/index.csv. <run> is the task name: H2O's longest action segment in the sequence that is not a plain
"place ..." or "grab ...", with spaces as underscores (e.g. apply_lotion); "_s1_<scene>_<seq>" is appended when two
requested sequences would get the same name. It then checks every file against MANIFEST.sha256 (sha256 only; the
manifest holds no H2O content). Files whose bytes differ are checked by content against your own download.

Needs (see data/h2o/README.md for the full setup):
  - An H2O login. Accept the terms of use at https://h2odataset.ethz.ch/ ; ETH emails a user name and password that
    are valid for 7 days. Put them in a netrc file (line: machine h2odataset.ethz.ch login <user> password <pw>,
    mode 600) passed with --netrc, or in the environment as H2O_USER / H2O_PASSWORD. They are never printed.
  - Optional: the MANO model (free registration at https://mano.is.tue.mpg.de), the folder holding MANO_LEFT.pkl
    and MANO_RIGHT.pkl, passed with --mano. Without it, mano_fit.npz and the 3D view are skipped.
  - For the 3D view, a GPU with EGL (pyrender, offscreen).
  - Disk: about 25 GB free while running (extracted sequences in --cache, deleted at the end); 3.7 GB at the end.
  - Time: the archive is one 37 GB .tar.gz that can only be read from the start; the default clips need its first
    20 GB (18.7 GiB): 43 min at 7.5 MB/s on 2026-10-07, about 9 min at 38 MB/s on 2026-10-08 (80 parallel range
    requests; one connection gives about 0.12 MB/s).

Run from the repository root either way (Linux x86_64):
  uv run data/h2o/reproduce_h2o.py --netrc ~/.h2o_netrc --mano /path/to/mano_v1_2/models
  python3.11 -m venv .venv && .venv/bin/pip install --no-deps -r data/h2o/requirements.txt && \
    .venv/bin/python data/h2o/reproduce_h2o.py --netrc ~/.h2o_netrc --mano /path/to/mano_v1_2/models
"""
import os

# Reproducible numerics: one BLAS thread and fixed CPU code paths, set before numpy loads its BLAS.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse, base64, csv, glob, hashlib, json, netrc, pickle, resource, shutil, subprocess, sys, tarfile, \
    threading, time, urllib.error, urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HOST = "h2odataset.ethz.ch"
URL = f"https://{HOST}/data/dataset/subject1_ego_v1_1.tar.gz"
ARCHIVE_SIZE = 37274083128
# order of the sequences inside the archive, with the GiB (2**30 bytes) read before each starts (measured 2026-10-07)
ARCHIVE_ORDER = {"h1/0": 0.0, "h1/6": 0.59, "h1/3": 1.26, "h1/7": 2.48, "h1/4": 3.54, "h1/2": 4.34, "h1/1": 5.0,
                 "h1/5": 5.86, "o1/0": 6.5, "o1/6": 7.07, "o1/3": 7.67, "o1/7": 8.53, "o1/4": 9.55, "o1/2": 10.27,
                 "o1/1": 10.95, "o1/5": 11.94, "o2/0": 12.65, "o2/6": 13.16, "o2/3": 13.87, "o2/7": 14.71,
                 "o2/4": 15.13, "o2/2": 15.8, "o2/1": 16.95, "o2/5": 17.72, "k1/0": 18.27, "k1/6": 18.68}
DEFAULT_CLIPS = ["h1/3", "o1/7", "o1/1", "o2/5", "k1/0"]  # in archive order
KEEP = {"rgb", "hand_pose", "cam_pose", "hand_pose_mano", "action_label", "obj_pose"}  # cam4 subfolders the build reads
N_DEPTH = 5  # the table-plane fit reads the first five depth maps

HANDS = ("left", "right")
TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
COL = {"left": (255, 140, 0), "right": (0, 210, 0)}  # BGR: left blue, right green
COL_RGB = {k: np.array(v[::-1]) / 255.0 for k, v in COL.items()}
FINGER_SHADE = [1.0, 0.9, 0.8, 0.7, 0.6]
SKIN = np.array([0.93, 0.76, 0.65])
FPS = 30.0
OBJECTS = ["background", "book", "espresso", "lotion", "spray", "milk", "cocoa", "chips", "cappuccino"]
ACTIONS = ["background", "grab book", "grab espresso", "grab lotion", "grab spray", "grab milk", "grab cocoa",
           "grab chips", "grab cappuccino", "place book", "place espresso", "place lotion", "place spray",
           "place milk", "place cocoa", "place chips", "place cappuccino", "open lotion", "open milk", "open chips",
           "close lotion", "close milk", "close chips", "pour milk", "take out espresso", "take out cocoa",
           "take out chips", "take out cappuccino", "put in espresso", "put in cocoa", "put in cappuccino",
           "apply lotion", "apply spray", "read book", "read espresso", "spray spray", "squeeze lotion"]
# dataset joint j <- MANO source: joint index, or ("v", vertex) for a fingertip
TO21 = [0, 13, 14, 15, ("v", 744), 1, 2, 3, ("v", 320), 4, 5, 6, ("v", 443), 10, 11, 12, ("v", 554),
        7, 8, 9, ("v", 671)]
X264 = ["-crf", "18", "-preset", "slow", "-threads", "1"]  # one thread: x264 output depends on the thread count
BIG = (1280, 960)  # size of the 3D view

# how each file is checked: "exact" must match MANIFEST.sha256 byte for byte; "tolerance" files are expected to
# match too (they do on the reference machine), and when their bytes differ (other CPU, GPU, driver or ffmpeg build)
# they are checked by content against your own download instead
CHECK = {"frames": "exact", "data.npz": "tolerance", "mano_fit.npz": "tolerance", "meta.json": "tolerance",
         "video.mp4": "tolerance", "overlay.mp4": "tolerance", "world3d_only.mp4": "tolerance",
         "README.md": "tolerance", "index.csv": "tolerance"}
HERE = Path(__file__).resolve().parent  # data/h2o/ in the repository: the default output folder


def log(*a):
    print(*a, flush=True)


def clip_name(c):
    """Suffix that keeps two runs with the same task name apart: s1_<scene>_<seq>."""
    return "s1_" + c.replace("/", "_")


GENERIC_VERBS = ("place ", "grab ")


def task_label(labels):
    """The H2O action label that names a run: the longest action segment that is not a plain 'place ...' or
    'grab ...' (ties: the earlier one); if there is none, the longest segment."""
    segs = segments(labels)
    main = [g for g in segs if not g["text"].startswith(GENERIC_VERBS)] or segs
    return max(main, key=lambda g: (g["end_idx"] - g["start_idx"], -g["start_idx"]))["text"]


def read_actions(cache, c):
    scene, seq = c.split("/")
    d = os.path.join(cache, "subject1", scene, seq, "cam4", "action_label")
    return np.array([int(np.loadtxt(f)) for f in sorted(glob.glob(os.path.join(d, "*.txt")))])


def run_names(clips, cache):
    """{<scene>/<seq>: (run folder name, task label)}; names are the task label with underscores, made unique among
    the requested clips by appending _s1_<scene>_<seq>."""
    lab = {c: task_label(read_actions(cache, c)) for c in clips}
    base = {c: lab[c].replace(" ", "_") for c in clips}
    dup = {b for b in base.values() if list(base.values()).count(b) > 1}
    return {c: (base[c] + "_" + clip_name(c) if base[c] in dup else base[c], lab[c]) for c in clips}


# ---------------------------------------------------------------- 1. download (stream the archive, keep only what is needed)
def auth_header(netrc_path):
    if netrc_path:
        a = netrc.netrc(os.path.expanduser(netrc_path)).authenticators(HOST)
        if not a:
            sys.exit(f"no 'machine {HOST}' entry in {netrc_path}")
        user, pw = a[0], a[2]
    else:
        user, pw = os.environ.get("H2O_USER"), os.environ.get("H2O_PASSWORD")
        if not (user and pw):
            sys.exit("no H2O login: pass --netrc FILE or set H2O_USER and H2O_PASSWORD "
                     "(ETH emails them after you accept the terms at https://h2odataset.ethz.ch/; valid 7 days)")
    return "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()


def seq_complete(cache, seq):
    return (Path(cache) / "subject1" / seq / ".complete").exists()


def wanted_member(parts):
    """parts = [top, scene, seq, 'cam4', sub, file] or [top, scene, seq, 'cam4', 'cam_intrinsics.txt']."""
    if len(parts) == 5 and parts[3] == "cam4" and parts[4] == "cam_intrinsics.txt":
        return True
    if len(parts) != 6 or parts[3] != "cam4":
        return False
    if parts[4] in KEEP:
        return True
    return parts[4] == "depth" and parts[5].endswith(".png") and parts[5][:-4].isdigit() and int(parts[5][:-4]) < N_DEPTH


def download(seqs, cache, get_auth, workers=80, chunk_mb=4, hedge_s=45.0):
    """Read the archive from the start with many parallel HTTP range requests, feed the bytes in order to a streaming
    tar reader, extract only the wanted files of the wanted sequences into <cache>/subject1/<scene>/<seq>/cam4 and stop
    after the last one. A sequence that is already complete in the cache is not extracted again; if every wanted
    sequence is complete nothing is downloaded. Failed range requests are retried (20 tries, backoff)."""
    want = {s for s in seqs if not seq_complete(cache, s)}
    if not want:
        log("  all sequences already in the cache; nothing to download")
        return dict(downloaded_gb=0.0, minutes=0.0, mb_s=0.0)
    auth = get_auth()  # the login is only needed when something must be downloaded
    try:
        req = urllib.request.Request(URL, method="HEAD", headers={"Authorization": auth})
        size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
    except urllib.error.HTTPError as e:
        sys.exit(f"H2O server refused the login (HTTP {e.code}); logins expire 7 days after registration")
    if size != ARCHIVE_SIZE:
        log(f"  warning: archive size {size} differs from the one this script was made with ({ARCHIVE_SIZE})")
    last_needed = (max(ARCHIVE_ORDER.get(s, 34.7) for s in want) + 0.7) * 2**30 / 1e9  # + about one sequence
    log(f"  sequences to extract: {sorted(want, key=lambda s: ARCHIVE_ORDER.get(s, 99))}; "
        f"about {min(last_needed, size / 1e9):.1f} GB of the {size / 1e9:.1f} GB archive must be read")
    CH = chunk_mb << 20
    NCH = (size + CH - 1) // CH

    def fetch(lo, hi):
        for attempt in range(20):
            try:
                r = urllib.request.Request(URL, headers={"Authorization": auth, "Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(r, timeout=30) as resp:
                    data = resp.read()
                if len(data) == hi - lo + 1:
                    return data
            except Exception as e:  # noqa: BLE001 (any network error: retry)
                print(f"  retry chunk at {lo}: {type(e).__name__}", file=sys.stderr, flush=True)
            time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"chunk at byte {lo} failed 20 times")

    chunks, cv, stop = {}, threading.Condition(), threading.Event()
    next_idx, consumed, hedges, max_ahead = [0], [0], [0], workers * 3  # bound memory: at most this many chunks held

    def worker():
        while not stop.is_set():
            with cv:
                while not stop.is_set() and (next_idx[0] >= NCH or len(chunks) >= max_ahead):
                    if next_idx[0] >= NCH:
                        return
                    cv.wait(1)
                if stop.is_set():
                    return
                i = next_idx[0]; next_idx[0] += 1
            data = fetch(i * CH, min(size, (i + 1) * CH) - 1)
            with cv:
                if i >= consumed[0]:
                    chunks[i] = data
                cv.notify_all()

    def hedge(i):
        data = fetch(i * CH, min(size, (i + 1) * CH) - 1)
        with cv:
            if i >= consumed[0] and i not in chunks:
                chunks[i] = data; hedges[0] += 1
            cv.notify_all()

    class Stream:
        """File-like object returning the archive bytes in order, blocking until each chunk arrives."""
        def __init__(self):
            self.i, self.buf, self.pos, self.got = 0, b"", 0, 0

        def read(self, n=-1):
            out = []
            while n != 0:
                if self.pos >= len(self.buf):
                    if self.i >= NCH:
                        break
                    with cv:
                        t_wait, hedged = time.time(), False
                        while self.i not in chunks:
                            cv.wait(5)
                            if not hedged and time.time() - t_wait > hedge_s:  # a stalled connection: ask again
                                threading.Thread(target=hedge, args=(self.i,), daemon=True).start(); hedged = True
                        self.buf = chunks.pop(self.i); consumed[0] = self.i + 1; cv.notify_all()
                    self.i += 1; self.pos = 0; self.got += len(self.buf)
                take = len(self.buf) - self.pos if n < 0 else min(n, len(self.buf) - self.pos)
                out.append(self.buf[self.pos:self.pos + take]); self.pos += take
                if n > 0:
                    n -= take
            return b"".join(out)

    for _ in range(workers):
        threading.Thread(target=worker, daemon=True).start()
    s, t0 = Stream(), time.time()
    done, cur, nfiles, last = set(), None, 0, time.time()

    def finish(seq):
        if seq in want:
            (Path(cache) / "subject1" / seq / ".complete").write_text("ok\n")
            done.add(seq)
            log(f"  extracted {seq}")

    with tarfile.open(fileobj=s, mode="r|gz") as tf:
        for m in tf:
            parts = m.name.split("/")  # subject1_ego/<scene>/<seq>/cam4/...
            seq = "/".join(parts[1:3]) if len(parts) >= 3 else None
            if seq and seq != cur:
                finish(cur)
                cur = seq
                if done == want:
                    break
            if seq in want and m.isfile() and wanted_member(parts):
                m.name = "/".join(["subject1"] + parts[1:])
                tf.extract(m, cache, filter="data")
                nfiles += 1
            if time.time() - last > 60:
                last = time.time()
                log(f"  {s.got / 1e9:.2f} GB read, {s.got / 1e6 / (last - t0):.1f} MB/s, at {cur}, {nfiles} files kept")
        else:
            finish(cur)
    stop.set()
    el = time.time() - t0
    miss = want - done
    if miss:
        sys.exit(f"sequences not found in the archive: {sorted(miss)}")
    st = dict(downloaded_gb=round(s.got / 1e9, 2), minutes=round(el / 60, 1), mb_s=round(s.got / 1e6 / el, 1),
              hedged_chunks=hedges[0], files_kept=nfiles)
    log(f"  read {st['downloaded_gb']} GB in {st['minutes']} min ({st['mb_s']} MB/s), kept {nfiles} files")
    return st


# ---------------------------------------------------------------- 2. build
def draw(fr, uv, inview_all, colour, k=1):
    """The skeleton drawing used for every overlay: solid bones and filled dots when all joints are in the image,
    thin dashed bones and hollow dots otherwise."""
    import cv2
    p = np.round(uv).astype(int)
    for fi, chain in enumerate(FINGERS):
        c = tuple(int(v * FINGER_SHADE[fi]) for v in colour)
        for a, b in zip(chain[:-1], chain[1:]):
            if inview_all:
                cv2.line(fr, tuple(p[a]), tuple(p[b]), c, 2 * k, cv2.LINE_AA)
            else:  # dashed: draw every other tenth of the bone
                for s in range(0, 10, 2):
                    q0 = p[a] + (p[b] - p[a]) * s / 10; q1 = p[a] + (p[b] - p[a]) * (s + 1) / 10
                    cv2.line(fr, tuple(q0.astype(int)), tuple(q1.astype(int)), c, k, cv2.LINE_AA)
    for j in range(21):
        cv2.circle(fr, tuple(p[j]), (4 if j == 0 else 3) * k, colour, -1 if inview_all else k, cv2.LINE_AA)


def overlay_frame(fr, f, valid, cam, px, W, H, caption_task, T):
    import cv2
    k = max(1, W // 640)
    for h, n in enumerate(HANDS):
        if not valid[f, h] or (cam[f, h, :, 2] <= 0.05).any():
            continue
        uv = px[f, h]
        all_in = bool(((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).all())
        draw(fr, uv, all_in, COL[n], k)
    cap, fs = f"H2O  {caption_task}  frame {f}/{T - 1}", W / 1280
    cv2.putText(fr, cap, (8, int(40 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(3, int(6 * fs)), cv2.LINE_AA)
    cv2.putText(fr, cap, (8, int(40 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(2 * fs)), cv2.LINE_AA)
    return fr


def central_diff(x, ts):
    v = np.full_like(x, np.nan)
    dt = (ts[2:] - ts[:-2]).reshape((-1,) + (1,) * (x.ndim - 1))
    v[1:-1] = (x[2:] - x[:-2]) / dt
    return v


class _Ch:
    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {"x": st})

    def arr(self):
        d = self.__dict__
        if "x" in d:
            return np.asarray(d["x"])
        return d["a"].arr().ravel()[d["idxs"]].reshape(d["preferred_shape"])


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith("chumpy"):
            return _Ch
        if mod.startswith("scipy.sparse"):
            import scipy.sparse as sp
            return getattr(sp, name)
        return super().find_class(mod, name)


def load_mano(path):
    with open(path, "rb") as f:
        d = _Unpickler(f, encoding="latin1").load()
    out = {}
    for k, v in d.items():
        v = v.arr() if isinstance(v, _Ch) else v
        out[k] = v.toarray() if hasattr(v, "toarray") else v
    return out


class Mano:
    """MANO v1.2 as shipped (H2O's convention: left shapedirs not sign-fixed). hand_pose is absolute (0 = flat hand)."""

    def __init__(self, model_dir, side):
        m = load_mano(f"{model_dir}/MANO_{side.upper()}.pkl")
        self.v_template = np.asarray(m["v_template"], np.float64)
        self.shapedirs = np.asarray(m["shapedirs"], np.float64)
        self.posedirs = np.asarray(m["posedirs"], np.float64).reshape(778 * 3, 135)
        self.J_regressor = np.asarray(m["J_regressor"], np.float64)
        self.weights = np.asarray(m["weights"], np.float64)
        self.faces = np.asarray(m["f"], np.int64)
        self.parents = [int(p) if p < 2**31 else -1 for p in np.asarray(m["kintree_table"])[0]]

    def rest(self, betas):
        v = self.v_template + np.einsum("vck,k->vc", self.shapedirs, betas)
        return v, self.J_regressor @ v

    def forward(self, betas, global_orient, hand_pose, transl, verts=False):
        """betas (10,); global_orient (T,3); hand_pose (T,45); transl (T,3) -> joints21 (T,21,3) [, vertices]."""
        from scipy.spatial.transform import Rotation as R
        T = len(global_orient)
        v0, J = self.rest(betas)
        Rm = R.from_rotvec(np.concatenate([global_orient, hand_pose], 1).reshape(-1, 3)).as_matrix().reshape(T, 16, 3, 3)
        v = v0[None] + ((Rm[:, 1:] - np.eye(3)).reshape(T, 135) @ self.posedirs.T).reshape(T, 778, 3)
        G = np.zeros((T, 16, 4, 4))
        for j in range(16):
            p = self.parents[j]
            A = np.zeros((T, 4, 4)); A[:, :3, :3] = Rm[:, j]; A[:, 3, 3] = 1
            A[:, :3, 3] = J[j] - (J[p] if p >= 0 else 0)
            G[:, j] = A if p < 0 else G[:, p] @ A
        Jp = G[:, :, :3, 3].copy()
        G[:, :, :3, 3] = Jp - np.einsum("tjab,jb->tja", G[:, :, :3, :3], J)
        W = np.einsum("vj,tjab->tvab", self.weights, G)
        vt = np.einsum("tvab,tvb->tva", W[:, :, :3, :3], v) + W[:, :, :3, 3] + transl[:, None]
        Jp = Jp + transl[:, None]
        j21 = np.stack([vt[:, s[1]] if isinstance(s, tuple) else Jp[:, s] for s in TO21], 1)
        return (j21, vt) if verts else j21


def table_plane(d, K, c2w, n_frames=N_DEPTH, iters=500, tol=0.01, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    pts = []
    for f in sorted(glob.glob(os.path.join(d, "depth", "*.png")))[:n_frames]:
        t = int(os.path.basename(f)[:-4])
        z = np.asarray(Image.open(f), np.float32)[::8, ::8] / 1000.0
        v, u = np.nonzero((z > 0.2) & (z < 1.5))
        zz = z[v, u]
        P = np.stack([(u * 8 - K[0, 2]) * zz / K[0, 0], (v * 8 - K[1, 2]) * zz / K[1, 1], zz], 1)
        pts.append(P @ c2w[t, :3, :3].T + c2w[t, :3, 3])
    P = np.concatenate(pts)
    best = (0, None)
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - a) @ n) < tol
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    Q = P[best[1]]
    n = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][-1]
    if (c2w[0, :3, 3] - Q.mean(0)) @ n < 0:
        n = -n
    return n, float(Q.mean(0) @ n)


def display_rotation(up, c2w):
    """Rows = output x, y, z: z = table normal, x = mean camera forward on the table."""
    z = up / np.linalg.norm(up)
    f = c2w[:, :3, 2].mean(0)
    x = f - z * (f @ z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def segments(labels):
    """Per-frame action ids -> [{text, start_idx, end_idx}] with end_idx exclusive; background (0) is skipped."""
    out, s = [], 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[s]:
            if labels[s] > 0:
                out.append(dict(text=ACTIONS[labels[s]], start_idx=int(s), end_idx=int(t)))
            s = t
    return out


def encode(src_frames_pattern, dst, fps):
    """Frames on disk -> H.264 CRF 18 with one encoder thread (same bytes on rerun with the same ffmpeg)."""
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-framerate", f"{fps:g}", "-i",
                    src_frames_pattern, "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-threads", "1", str(dst)], check=True)


def writer(path, fps):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=1,
                              ffmpeg_params=X264)


def load_source(d):
    """Everything the build reads from one extracted cam4 folder."""
    ld = lambda sub, ext="txt": sorted(glob.glob(os.path.join(d, sub, f"*.{ext}")))
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(d, "cam_intrinsics.txt"))
    W, H = int(W), int(H)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    hand = np.stack([np.loadtxt(f) for f in ld("hand_pose")])  # T x 128, camera frame, metres
    T = len(hand)
    valid = np.stack([hand[:, 0], hand[:, 64]], 1) > 0
    cam = np.concatenate([hand[:, 1:64], hand[:, 65:128]], 1).reshape(T, 2, 21, 3)
    cam[~valid] = np.nan
    c2w_h = np.stack([np.loadtxt(f).reshape(4, 4) for f in ld("cam_pose")])
    mano = np.stack([np.loadtxt(f) for f in ld("hand_pose_mano")])  # T x 124
    action = np.array([int(np.loadtxt(f)) for f in ld("action_label")])
    obj = np.stack([np.loadtxt(f) for f in ld("obj_pose")])
    obj_name = OBJECTS[int(np.bincount(obj[:, 0].astype(int)).argmax())]
    pngs = ld("rgb", "png")
    frame_ids = np.array([int(os.path.basename(f)[:-4]) for f in ld("hand_pose")], np.int32)
    assert len(pngs) == len(c2w_h) == len(mano) == len(action) == T and mano.shape[1] == 124
    return dict(K=K, W=W, H=H, T=T, valid=valid, cam=cam, c2w=c2w_h, mano=mano, action=action, obj_name=obj_name,
                pngs=pngs, frame_ids=frame_ids)


def build(args):
    """One clip: data.npz, mano_fit.npz (with MANO), frames/, video.mp4, overlay.mp4, meta.json, README.md."""
    import cv2
    from scipy.spatial.transform import Rotation as R
    scene, seq, src, out, mano_dir, name, label = args
    t0 = time.time()
    d = os.path.join(src, "subject1", scene, seq, "cam4")
    s = load_source(d)
    K, W, H, T, valid, cam, c2w_h, mano, action = (s[k] for k in ("K", "W", "H", "T", "valid", "cam", "c2w", "mano", "action"))
    obj_name, pngs, frame_ids = s["obj_name"], s["pngs"], s["frame_ids"]

    up, table_h = table_plane(d, K, c2w_h)
    Rd = display_rotation(up, c2w_h)
    A4 = np.eye(4); A4[:3, :3] = Rd; A4[2, 3] = -table_h  # world_out <- world_h2o
    head = A4[None] @ c2w_h  # camera-to-world, output world
    Rcw, tcw = head[:, :3, :3], head[:, :3, 3]
    world = np.einsum("tij,thkj->thki", Rcw, cam) + tcw[:, None, None]
    z = cam[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        px = cam[..., :2] / z[..., None] * K[[0, 1], [0, 1]] + K[:2, 2]
    px[~(z > 0.05)] = np.nan
    wpx = px[:, :, 0]
    with np.errstate(invalid="ignore"):
        in_view = valid & (z[:, :, 0] > 0.05) & (wpx[..., 0] >= 0) & (wpx[..., 0] < W) & (wpx[..., 1] >= 0) & (wpx[..., 1] < H)
    ts = np.arange(T) / FPS

    # MANO: camera-frame H2O annotation -> output world (the model is needed for mano_fit.npz only)
    mf, quat = {}, np.full((T, 2, 4), np.nan)
    for h, side in enumerate(HANDS):
        m = mano[:, 62 * h:62 * (h + 1)]
        ok = (m[:, 0] > 0) & valid[:, h]
        Rgo = Rcw @ R.from_rotvec(m[:, 4:7]).as_matrix()
        q = R.from_matrix(Rgo).as_quat()[:, [3, 0, 1, 2]]
        q[~ok] = np.nan
        quat[:, h] = q
        if mano_dir is None:
            continue
        betas = np.median(m[ok, 52:62], 0)
        layer = Mano(mano_dir, side)
        J0 = layer.rest(betas)[1][0]
        hp, tr_c = m[:, 7:52], m[:, 1:4]
        go_w = R.from_matrix(Rgo).as_rotvec()
        tr_w = np.einsum("tij,tj->ti", Rcw, J0 + tr_c) + tcw - J0
        j21 = layer.forward(betas, go_w, hp, tr_w)
        err = np.linalg.norm(j21 - world[:, h], axis=-1) * 1000
        for arr in (go_w, hp, tr_w, j21, err):
            arr[~ok] = np.nan
        mf.update({f"{side}_betas": betas.astype(np.float32), f"{side}_global_orient": go_w.astype(np.float32),
                   f"{side}_transl": tr_w.astype(np.float32), f"{side}_hand_pose": hp.astype(np.float32),
                   f"{side}_err": err.astype(np.float32), f"{side}_joints21_world": j21.astype(np.float32)})

    data = dict(
        timestamps=ts.astype(np.float32), frame_index=frame_ids,
        skeleton_world=world.astype(np.float32), skeleton_cam=cam.astype(np.float32), skeleton_px=px.astype(np.float32),
        fingertips_world=world[:, :, TIPS].astype(np.float32), fingertips_cam=cam[:, :, TIPS].astype(np.float32),
        wrist_pos_world=world[:, :, 0].astype(np.float32), wrist_pos_cam=cam[:, :, 0].astype(np.float32),
        wrist_quat_world=quat.astype(np.float32),
        wrist_vel_world=central_diff(world[:, :, 0], ts).astype(np.float32),
        fingertip_vel_world=central_diff(world[:, :, TIPS], ts).astype(np.float32),
        valid=valid, in_view=in_view, head_pose_world=head.astype(np.float32), K=K.astype(np.float32),
        image_size=np.array([W, H], dtype=np.int32))
    clip = Path(out) / name
    clip.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(clip / "data.npz", **data)
    if mf:
        mf = {k: mf[k] for k in ["left_betas", "left_global_orient", "left_transl", "left_hand_pose", "left_err",
                                 "right_betas", "right_global_orient", "right_transl", "right_hand_pose", "right_err",
                                 "left_joints21_world", "right_joints21_world"]}  # key order as egoverse/trace
        np.savez_compressed(clip / "mano_fit.npz", **mf)
    t_npz = time.time() - t0

    fdir = clip / "frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir()
    for f in range(T):
        shutil.copyfile(pngs[f], fdir / f"{f:04d}.png")
    encode(str(fdir / "%04d.png"), clip / "video.mp4", FPS)
    t_video = time.time() - t0 - t_npz

    caption_task = f"{scene}/{seq} {obj_name}"
    ov = writer(clip / "overlay.mp4", FPS)
    for f in range(T):
        fr = overlay_frame(cv2.imread(pngs[f]), f, valid, cam, px, W, H, caption_task, T)
        ov.append_data(fr[:, :, ::-1])
    ov.close()
    t_overlay = time.time() - t0 - t_npz - t_video

    ann = segments(action)
    meta = dict(
        clip=name, dataset="H2O", subject=1, scene=scene, sequence=int(seq), object=obj_name,
        rig="H2O egocentric camera cam4 (head-mounted RGB-D)", task=name, task_label=label,
        task_naming="the longest H2O action segment that is not a plain 'place ...' or 'grab ...', spaces as underscores",
        task_description=", ".join(sorted({a["text"] for a in ann}, key=[a["text"] for a in ann].index)),
        source_path=f"https://h2odataset.ethz.ch/data/dataset/subject1_ego_v1_1.tar.gz:subject1/{scene}/{seq}/cam4",
        source_frames=f"subject1/{scene}/{seq}/cam4/rgb", window=[0, T], episode_frames=T, fps=FPS,
        frames=T, duration_s=round(T / FPS, 3), image_size=[W, H],
        units=dict(position="m", velocity="m/s", pixels="px", time="s"),
        world_from_h2o_world=A4.round(9).tolist(),
        table=dict(up_in_h2o_world=up.round(9).tolist(), height_in_h2o_world_m=round(table_h, 6)),
        conventions=dict(
            hands="axis 1 of every (T, 2, ...) array: 0 = left, 1 = right",
            joints="21: 0 wrist; thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20, base to tip",
            fingertips="joints 4, 8, 12, 16, 20 (thumb, index, middle, ring, pinky)",
            world="H2O's world frame rotated so +z is the table normal (up), x is the mean camera viewing direction "
                  "on the table, and the table is z = 0: p = R_d p_h2o - table_height e_z (world_from_h2o_world)",
            camera="OpenCV: x right, y down, z forward; cam = R^T (world - t) with head_pose_world = [R t; 0 1]",
            head_pose_world="camera-to-world 4x4 (H2O cam_pose moved into the output world)",
            quaternion="wrist_quat_world is [qw, qx, qy, qz]: the rotation of H2O's MANO root (global orient), moved "
                       "into the output world; MANO's root axes, not a tracker wrist frame",
            in_view="hand labelled, wrist more than 5 cm in front of the camera and its pixel inside the image",
            skeleton_px="NaN where the joint is at or behind the camera (z <= 5 cm)",
            annotations="H2O's per-frame action label as segments; start_idx inclusive, end_idx exclusive, "
                        "in clip frames; background frames are not listed",
            missing="NaN, never zeros",
        ),
        label_source="H2O's shipped labels: hand poses from MANO fitting to multi-view RGB-D in a lab, not motion capture",
        annotations=ann,
    )
    (clip / "meta.json").write_text(json.dumps(meta, indent=1))
    write_clip_readme(clip)
    el = time.time() - t0
    row = dict(clip=name, scene=scene, sequence=int(seq), object=obj_name, task_label=label, frames=T, fps=FPS, duration_s=round(T / FPS, 2),
               left_in_view_pct=round(100 * in_view[:, 0].mean(), 1),
               right_in_view_pct=round(100 * in_view[:, 1].mean(), 1), action_segments=len(ann))
    perf = dict(clip=name, frames=T, total_s=round(el, 1), ms_per_frame=round(1000 * el / T, 1),
                npz_mano_s=round(t_npz, 1), frames_copy_video_s=round(t_video, 1), overlay_s=round(t_overlay, 1),
                peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
    log(f"  built {name}: {T} frames in {el:.0f} s")
    return row, perf


TERMS = """## Terms of use

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
"""


def write_clip_readme(clip):
    """Per-run README, same layout as data/egoverse/trace/<task>/README.md. It states only clip facts (sequence, object,
    frame count, rate, image size) and H2O's action label names with their frame ranges, so it can be committed to
    the public repository; it is identical with or without --mano."""
    meta = json.loads((clip / "meta.json").read_text())
    T, n = meta["frames"], clip.name
    acts = "\n".join(f"| {i + 1} | {a['text']} | {a['start_idx']}-{a['end_idx']} |"
                     for i, a in enumerate(meta["annotations"]))
    txt = f"""# H2O run `{n}`: hand skeletons, fingertips and wrists

**The data is not included in this repository** (H2O's terms do not allow passing it on). Run `data/h2o/reproduce_h2o.py`
with your own H2O login to download and build it here; see [`data/h2o/README.md`](../README.md). This README describes what
the script writes into this folder.

One {meta['duration_s']:.1f}-second egocentric clip with per-frame 3D hand skeletons, for testing retargeting from human
hands to robot hands. Same file layout, array keys and conventions as the EgoVerse runs in
[`data/egoverse/trace`](../../egoverse/trace/README.md).

| | |
|---|---|
| source | [H2O](https://h2odataset.ethz.ch/) (Kwon et al., ICCV 2021), subject 1, scene `{meta['scene']}`, sequence `{meta['sequence']}`, egocentric camera `cam4` |
| object | `{meta['object']}` |
| task name | `{n}` |
| window | the whole sequence, frames 0-{T - 1} ({T} frames) |
| rate, length | {meta['fps']:g} fps, {meta['duration_s']:.1f} s |
| image | {meta['image_size'][0]} x {meta['image_size'][1]} |

## Task name and annotations

H2O labels every frame of a sequence with one action from its 36-action vocabulary (verb + object, such as
`grab {meta['object']}`) or background. We name the run after its main action: the longest action segment that is not a
plain `place ...` or `grab ...`, here **`{meta['task_label']}`**, written `{n}`.

All action segments of the sequence, in clip frames (start inclusive, end exclusive; frames not listed are background).
`meta.json` holds the same list under `annotations`.

| # | H2O action label | frames |
|---|---|---|
{acts}

**Label caveat.** The hand and camera poses are H2O's shipped labels: MANO hands fitted to multi-view RGB-D recordings in a
lab (one desk with calibration markers), not motion capture.

## Files

| file | what it is |
|---|---|
| `frames/0000.png` ... `frames/{T - 1:04d}.png` | the original H2O RGB frames (PNG), byte for byte; frame `i` is row `i` of every array |
| `video.mp4` | the same frames as H.264 (CRF 18, yuv420p), for viewing |
| `overlay.mp4` | the video with both 21-joint skeletons drawn on it (left hand blue, right hand green), CRF 18 |
| `{n}_world3d_only.mp4` | floating 3D view in the world frame: MANO hand meshes over the skeletons, the head camera as a pyramid, a fixed virtual camera, 1280 x 960, CRF 18 (needs `--mano`) |
| `data.npz` | per-frame skeleton, fingertip, wrist and camera arrays (below) |
| `mano_fit.npz` | H2O's MANO hand annotation, moved into the world frame (below; needs `--mano`) |
| `meta.json` | clip details, conventions, the world transform and H2O's action labels as annotations |

## `data.npz`

T = {T}. Axis 1 of every `(T, 2, ...)` array is the hand: 0 = left, 1 = right. float32 unless noted. Missing values
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
x_{{\\text{{cam}}}} = R^\\top (x_{{\\text{{world}}}} - t)
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
d = np.load("data/h2o/{n}/data.npz")
tips = d["fingertips_world"]                                 # ({T}, 2, 5, 3) metres
wrist = d["wrist_pos_world"][:, 1]                           # right wrist, ({T}, 3)
speed = np.linalg.norm(d["wrist_vel_world"][:, 1], axis=-1)  # m/s
```

""" + TERMS
    (clip / "README.md").write_text(txt)


def write_index(out, res):
    """Add or replace this run's clips in index.csv; keep rows of clips not rebuilt."""
    idx = os.path.join(out, "index.csv")
    rows = {}
    if os.path.exists(idx):
        with open(idx, newline="") as f:
            rows = {r["clip"]: r for r in csv.DictReader(f)}
    for row, _ in res:
        rows[row["clip"]] = row
    with open(idx, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0][0]))
        w.writeheader(); w.writerows(rows[k] for k in sorted(rows))


# ---------------------------------------------------------------- 3. floating 3D view (same look as the EgoVerse clips)
def frustum_segs(Tcw, K, W, H, depth=0.08):
    c = Tcw[:3, 3]
    corners = []
    for u, v in [(0, 0), (W, 0), (W, H), (0, H)]:
        ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
        corners.append(Tcw[:3, :3] @ (ray / ray[2] * depth) + c)
    segs = [(c, p) for p in corners] + [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    up_tip = Tcw[:3, :3] @ np.array([0, -0.45 * depth * H / K[1, 1] * 1.6, depth]) + c
    segs.append(((corners[0] + corners[1]) / 2, up_tip))
    return segs


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    Tm = np.eye(4); Tm[:3, 0], Tm[:3, 1], Tm[:3, 2], Tm[:3, 3] = r, u, -f, eye
    return Tm


YFOV = np.deg2rad(45)


def view_camera(heads, valid, skel, K, W, H):
    """The one fixed virtual camera of a clip's 3D view (OpenGL camera-to-world): behind, right of and above the person
    (up = world +z), as close as keeping every hand joint and every head-camera pyramid corner inside 90 % of the image
    allows. Returns (Vcam, fwd, right, up, hand points)."""
    up = np.array([0.0, 0, 1])
    fv = -heads[:, :3, 1].mean(0); fv -= (fv @ up) * up; fwd = fv / np.linalg.norm(fv)
    right = np.cross(fwd, up)
    hp = skel[valid].reshape(-1, 3); hp = hp[np.isfinite(hp).all(1)]
    fr = np.array([p for Tcw in heads for seg in frustum_segs(Tcw, K, W, H) for p in seg])
    pts = np.concatenate([hp, fr])
    dirn = -0.8 * fwd + 0.55 * up + 0.5 * right; dirn /= np.linalg.norm(dirn)
    th, tv = np.tan(YFOV / 2) * BIG[0] / BIG[1], np.tan(YFOV / 2)

    def fits(target, dist, margin=0.9):
        V = look_at(target + dist * dirn, target, up)
        c = (pts - V[:3, 3]) @ V[:3, :3]
        zz = -c[:, 2]
        if (zz <= 0.02).any():
            return False, None
        x, y = c[:, 0] / zz / th, c[:, 1] / zz / tv
        return (np.abs(x) <= margin).all() and (np.abs(y) <= margin).all(), (x, y, zz, V)

    def closest(target):
        lo, hi = 0.05, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if fits(target, mid)[0] else (mid, hi)
        return hi

    target = (pts.min(0) + pts.max(0)) / 2
    for _ in range(3):
        _, (x, y, zz, V) = fits(target, closest(target))
        cx, cy, zc = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, np.median(zz)
        target = target + V[:3, 0] * cx * th * zc + V[:3, 1] * cy * tv * zc
    return look_at(target + closest(target) * dirn, target, up), fwd, right, up, hp


def project_view(Vcam, p):
    """World points -> pixel (u, v) in the 3D view."""
    c = (p - Vcam[:3, 3]) @ Vcam[:3, :3]
    th, tv = np.tan(YFOV / 2) * BIG[0] / BIG[1], np.tan(YFOV / 2)
    x, y = c[..., 0] / -c[..., 2] / th, c[..., 1] / -c[..., 2] / tv
    return np.stack([(x + 1) / 2 * BIG[0], (1 - y) / 2 * BIG[1]], -1)


def render_world3d(clip, manos):
    """<clip>_world3d_only.mp4 from data.npz and mano_fit.npz: half-transparent MANO meshes over the 21-joint skeletons,
    a 10 cm floor grid, world axes and the head camera as a pyramid, seen by one fixed virtual camera (behind, right of
    and above the person, as close as keeping every hand joint and pyramid corner inside 90 % of the image allows)."""
    import cv2, pyrender, trimesh

    def cylinder_between(a, b, r):
        v = b - a
        L = np.linalg.norm(v)
        if L < 1e-6:
            return None
        m = trimesh.creation.cylinder(radius=r, height=L, sections=10)
        zz = np.array([0, 0, 1.0]); dd = v / L
        ax = np.cross(zz, dd); s = np.linalg.norm(ax)
        Rm = np.eye(3) if s < 1e-8 else trimesh.transformations.rotation_matrix(np.arctan2(s, zz @ dd), ax)[:3, :3]
        Tm = np.eye(4); Tm[:3, :3] = Rm; Tm[:3, 3] = (a + b) / 2
        m.apply_transform(Tm)
        return m

    def mat(c, alpha=1.0, rough=0.6, blend=False):
        kw = dict(alphaMode="BLEND", doubleSided=True) if blend else {}
        return pyrender.MetallicRoughnessMaterial(baseColorFactor=[*c, alpha], metallicFactor=0.0,
                                                  roughnessFactor=rough, **kw)

    def skeleton_meshes(j, side, r_bone=0.0025, r_joint=0.004):
        res = []
        for fi, chain in enumerate(FINGERS):
            parts = [cylinder_between(j[a], j[b], r_bone) for a, b in zip(chain[:-1], chain[1:])]
            for jj in chain[1:]:
                s = trimesh.creation.icosphere(subdivisions=1, radius=r_joint); s.apply_translation(j[jj]); parts.append(s)
            parts = [p for p in parts if p is not None]
            if parts:
                res.append(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(parts),
                                                      material=mat(COL_RGB[side] * FINGER_SHADE[fi])))
        s = trimesh.creation.icosphere(subdivisions=1, radius=r_joint * 1.4); s.apply_translation(j[0])
        res.append(pyrender.Mesh.from_trimesh(s, material=mat(COL_RGB[side])))
        return res

    def lines_mesh(segs, rgb):
        pos = np.asarray(segs, np.float32).reshape(-1, 3)
        col = np.tile(np.array([*rgb, 1.0], np.float32), (len(pos), 1))
        return pyrender.Mesh([pyrender.Primitive(positions=pos, color_0=col, mode=1)])

    d, mf = np.load(clip / "data.npz"), np.load(clip / "mano_fit.npz")
    meta = json.loads((clip / "meta.json").read_text())
    f64 = lambda x: np.asarray(x, np.float64)
    W, H = (int(v) for v in d["image_size"])
    K, heads, valid, skel = f64(d["K"]), f64(d["head_pose_world"]), d["valid"], f64(d["skeleton_world"])
    T = len(heads)
    verts = {}
    for s in HANDS:  # MANO vertices in the world frame from the stored parameters
        g = lambda k: f64(mf[f"{s}_{k}"])
        ok = np.isfinite(g("global_orient")).all(1)
        v = np.zeros((T, 778, 3))
        if ok.any():
            v[ok] = manos[s].forward(g("betas"), g("global_orient")[ok], g("hand_pose")[ok], g("transl")[ok], verts=True)[1]
        verts[s] = v
    Vcam, fwd, right, up, hp = view_camera(heads, valid, skel, K, W, H)
    yfov = YFOV
    floor = (hp @ up).min() - 0.05
    allc = np.concatenate([hp, heads[:, :3, 3]])
    centre = (np.percentile(allc, 2, 0) + np.percentile(allc, 98, 0)) / 2
    half = max(0.5, 1.3 * np.percentile(np.linalg.norm(allc - centre, axis=1), 99))
    o = centre - (centre @ up - floor) * up
    n = int(np.ceil(half / 0.1))
    gsegs = []
    for i in range(-n, n + 1):  # 10 cm grid on the floor
        gsegs.append((o + i * 0.1 * right - n * 0.1 * fwd, o + i * 0.1 * right + n * 0.1 * fwd))
        gsegs.append((o + i * 0.1 * fwd - n * 0.1 * right, o + i * 0.1 * fwd + n * 0.1 * right))
    grid = lines_mesh(gsegs, (0.55, 0.55, 0.6))
    axes = []
    for k, c in enumerate([(0.9, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.3, 0.95)]):
        e = np.zeros(3); e[k] = 0.1
        axes.append(pyrender.Mesh.from_trimesh(cylinder_between(o, o + e, 0.003), material=mat(c)))
    rend = pyrender.OffscreenRenderer(*BIG)
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=BIG[0] / BIG[1], znear=0.01, zfar=50)
    hand_mat = {s: mat(0.5 * SKIN + 0.5 * COL_RGB[s], alpha=0.5, rough=0.8, blend=True) for s in HANDS}
    wr = writer(clip / f"{clip.name}_world3d_only.mp4", FPS)
    scale = BIG[1] / 480
    task = f"{meta['scene']}/{meta['sequence']} {meta['object']}"
    for f in range(T):
        sc = pyrender.Scene(bg_color=[0.97, 0.97, 0.98, 1.0], ambient_light=[0.4, 0.4, 0.4])
        sc.add(cam, pose=Vcam)
        sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=Vcam)
        sc.add(grid)
        for a in axes:
            sc.add(a)
        sc.add(lines_mesh(frustum_segs(heads[f], K, W, H), (0.15, 0.15, 0.15)))
        for h, s in enumerate(HANDS):
            if valid[f, h]:
                for m in skeleton_meshes(skel[f, h], s):
                    sc.add(m)
                sc.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts[s][f], manos[s].faces, process=False),
                                                  material=hand_mat[s]))
        cb, _ = rend.render(sc, flags=pyrender.RenderFlags.RGBA)
        cb = np.ascontiguousarray(cb[..., :3][..., ::-1])  # BGR for the caption
        text, org = f"world frame  H2O  {task}  frame {f}/{T - 1}", (8, int(26 * scale))
        cv2.putText(cb, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), max(3, int(4 * scale)), cv2.LINE_AA)
        cv2.putText(cb, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)
        wr.append_data(np.ascontiguousarray(cb[..., ::-1]))
    wr.close()


# ---------------------------------------------------------------- 4. check against the manifest
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


NPZ_KEYS = {  # key: (dtype, shape after T)
    "timestamps": ("float32", ()), "frame_index": ("int32", ()), "skeleton_world": ("float32", (2, 21, 3)),
    "skeleton_cam": ("float32", (2, 21, 3)), "skeleton_px": ("float32", (2, 21, 2)),
    "fingertips_world": ("float32", (2, 5, 3)), "fingertips_cam": ("float32", (2, 5, 3)),
    "wrist_pos_world": ("float32", (2, 3)), "wrist_pos_cam": ("float32", (2, 3)), "wrist_quat_world": ("float32", (2, 4)),
    "wrist_vel_world": ("float32", (2, 3)), "fingertip_vel_world": ("float32", (2, 5, 3)), "valid": ("bool", (2,)),
    "in_view": ("bool", (2,)), "head_pose_world": ("float32", (4, 4))}


def content_check(out, rel, src):
    """Used when a tolerance-class file's bytes differ from the manifest: checks by content, using only your own
    download (no H2O-derived reference values are stored). Returns (ok, detail)."""
    import cv2, imageio.v2 as imageio
    clip = (out / rel).parent
    name = (out / rel).name
    if rel == "index.csv":
        with open(out / rel, newline="") as f:
            rows = {r["clip"]: r for r in csv.DictReader(f)}
        bad = [c for c, r in rows.items() if not (out / c / "meta.json").exists()
               or int(r["frames"]) != json.loads((out / c / "meta.json").read_text())["frames"]]
        return not bad, f"{len(rows)} rows" + (f"; inconsistent: {bad}" if bad else ", frame counts match the clips")
    if name == "README.md":
        t = (out / rel).read_text()
        return clip.name in t and "Terms of use" in t, f"{len(t)} characters"
    meta = json.loads((clip / "meta.json").read_text())
    scene, seq = meta["scene"], str(meta["sequence"])
    s = load_source(os.path.join(src, "subject1", scene, seq, "cam4"))
    T = s["T"]
    if name == "data.npz":
        d = np.load(clip / "data.npz")
        bad = [k for k, (dt, sh) in NPZ_KEYS.items() if k not in d or str(d[k].dtype) != dt or d[k].shape != (T,) + sh]
        if bad:
            return False, f"wrong keys, dtypes or shapes: {bad}"
        w, c, hpw = d["skeleton_world"].astype(np.float64), d["skeleton_cam"].astype(np.float64), d["head_pose_world"].astype(np.float64)
        res = {
            "skeleton_cam equals H2O hand_pose": bool(np.array_equal(d["skeleton_cam"], s["cam"].astype(np.float32), equal_nan=True)),
            "fingertips and wrists equal skeleton joints": bool(np.array_equal(d["fingertips_world"], d["skeleton_world"][:, :, TIPS], equal_nan=True)
                                                                and np.array_equal(d["wrist_pos_world"], d["skeleton_world"][:, :, 0], equal_nan=True)),
            "cam = R^T (world - t) within 0.1 mm": float(np.nanmax(np.abs(np.einsum("tji,thkj->thki", hpw[:, :3, :3], w - hpw[:, None, None, :3, 3]) - c))) < 1e-4,
            "projection within 0.01 px": float(np.nanmax(np.abs(c[..., :2] / c[..., 2:3] * d["K"][[0, 1], [0, 1]] + d["K"][:2, 2] - d["skeleton_px"]))) < 1e-2,
            "unit quaternions": float(np.nanmax(np.abs(np.linalg.norm(d["wrist_quat_world"], axis=-1) - 1))) < 1e-4,
            "hands above the table (z > 0)": float(np.nanmedian(w[..., 2])) > 0,
            "valid equals H2O's flags": bool(np.array_equal(d["valid"], s["valid"])),
        }
        return bool(all(res.values())), "; ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in res.items())
    if name == "mano_fit.npz":
        m = np.load(clip / "mano_fit.npz")
        res = [f"{h}: mean {np.nanmean(m[f'{h}_err']):.2f} mm, max {np.nanmax(m[f'{h}_err']):.1f} mm" for h in HANDS]
        ok = all(np.nanmean(m[f"{h}_err"]) < 1.0 and np.nanmax(m[f"{h}_err"]) < 6.0 for h in HANDS)
        return bool(ok), "MANO vs labelled joints " + "; ".join(res) + " (need mean < 1 mm, max < 6 mm)"
    if name == "meta.json":
        want = segments(s["action"])
        Rw = np.array(meta["world_from_h2o_world"])[:3, :3]
        okk = bool(meta["frames"] == T and meta["annotations"] == want and meta["object"] == s["obj_name"]
                   and np.allclose(Rw @ Rw.T, np.eye(3), atol=1e-6))
        return okk, "frames, object and action segments recomputed from your download " + ("match" if okk else "DIFFER")
    rd = imageio.get_reader(clip / name)
    fr = [np.asarray(x) for x in rd]; rd.close()  # plain arrays (imageio's Array subclass warns under numpy 2)
    if len(fr) != T:
        return False, f"{len(fr)} frames, expected {T}"
    if name in ("video.mp4", "overlay.mp4"):
        d = np.load(clip / "data.npz")
        ps = []
        for f in range(0, T, 50):
            img = cv2.imread(str(clip / "frames" / f"{f:04d}.png"))
            if name == "overlay.mp4":
                img = overlay_frame(img, f, d["valid"], d["skeleton_cam"], d["skeleton_px"], s["W"], s["H"],
                                    f"{scene}/{seq} {meta['object']}", T)
            ps.append(psnr(fr[f][..., ::-1], img))
        return bool(min(ps) >= 33.0), f"PSNR against the frames {min(ps):.1f}-{max(ps):.1f} dB (need >= 33)"
    # 3D view: size, frame count, and the hands drawn where data.npz puts them: the fixed view camera is recomputed from
    # data.npz and each labelled wrist and fingertip must land on a non-background pixel (no reference images needed)
    d = np.load(clip / "data.npz")
    size_ok = fr[0].shape[:2] == (BIG[1], BIG[0])
    f64 = lambda x: np.asarray(x, np.float64)
    W, H = (int(v) for v in d["image_size"])
    skel, valid = f64(d["skeleton_world"]), d["valid"]
    Vcam = view_camera(f64(d["head_pose_world"]), valid, skel, f64(d["K"]), W, H)[0]
    bg = np.array([0.97, 0.97, 0.98]) * 255
    hits, tot = 0, 0
    for f in range(0, T, 30):
        img = fr[f].astype(np.float64)
        for h in range(2):
            if not valid[f, h]:
                continue
            for j in [0] + TIPS:
                u, v = np.round(project_view(Vcam, skel[f, h, j])).astype(int)
                if 2 <= u < BIG[0] - 2 and 50 <= v < BIG[1] - 2:  # below the caption band
                    tot += 1
                    hits += np.abs(img[v - 2:v + 3, u - 2:u + 3] - bg).sum(-1).max() > 40
    ok = bool(size_ok and tot > 0 and hits >= 0.9 * tot)
    return ok, (f"{T} frames, {fr[0].shape[1]}x{fr[0].shape[0]}, {hits}/{tot} sampled wrist and fingertip positions "
                f"drawn (need >= 90 %)")


def check_class(rel):
    n = Path(rel).name
    if rel.split("/")[1:2] == ["frames"]:
        return "frames"
    return "world3d_only.mp4" if n.endswith("_world3d_only.mp4") else n


def verify(out, manifest, names, src, have_mano):
    want = {}
    for line in manifest.read_text().splitlines():
        h, p = line.split(maxsplit=1)
        want[p] = h
    names = set(names)
    ok_all, results = True, {}
    for p, h in sorted(want.items()):
        top = p.split("/")[0]
        if "/" in p and top not in names:
            continue  # a clip that was not requested
        cls = CHECK[check_class(p)]
        f = out / p
        if not have_mano and (p.endswith("mano_fit.npz") or p.endswith("_world3d_only.mp4")):
            results[p] = ("skipped", True, "built without --mano")
            continue
        if not f.exists():
            ok, detail = False, "missing"
        elif sha256(f) == h:
            ok, detail = True, "identical"
        elif cls == "exact":
            ok, detail = False, "bytes differ"
        else:
            ok, detail = content_check(out, p, src)
            ok, detail = bool(ok), "bytes differ; " + detail
        results[p] = (cls, ok, detail)
        ok_all &= ok
    for c in sorted(names):
        nf = [r for p, r in results.items() if p.startswith(f"{c}/frames/")]
        if nf:
            log(f"  {c}/frames/ ({len(nf)} files) [exact]: {sum(r[1] for r in nf)}/{len(nf)} identical")
        for p, (cls, ok, detail) in results.items():
            if p.startswith(f"{c}/") and not p.startswith(f"{c}/frames/"):
                log(f"  {p} [{cls}]: {'OK' if ok else 'MISMATCH'} ({detail})")
    for p, (cls, ok, detail) in results.items():
        if "/" not in p:
            log(f"  {p} [{cls}]: {'OK' if ok else 'MISMATCH'} ({detail})")
    unknown = sorted(n for n in names if not any(p.startswith(n + "/") for p in want))
    if unknown:
        log(f"  no reference in the manifest for {unknown} (built, not checked)")
    return ok_all


def write_manifest(out, manifest):
    """Maintainer use: record the current files as the reference (sha256 only)."""
    runs = [d for d in out.iterdir() if d.is_dir() and not d.name.startswith(".") and (d / "meta.json").exists()]
    files = sorted([str(q.relative_to(out)) for d in runs for q in d.rglob("*") if q.is_file()]
                   + (["index.csv"] if (out / "index.csv").exists() else []))
    manifest.write_text("".join(f"{sha256(out / p)}  {p}\n" for p in files))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=HERE,
                    help="output folder (default: this data/h2o/ folder, next to the committed run READMEs)")
    ap.add_argument("--netrc", help="netrc file with 'machine h2odataset.ethz.ch login ... password ...' "
                                    "(or set H2O_USER / H2O_PASSWORD)")
    ap.add_argument("--mano", help="folder with MANO_LEFT.pkl and MANO_RIGHT.pkl (optional: mano_fit.npz and the 3D view)")
    ap.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS, help="subject-1 sequences as <scene>/<seq>")
    ap.add_argument("--cache", type=Path, help="where the extracted sequences go (default: <out>/.cache); "
                                               "complete sequences already there are not downloaded again")
    ap.add_argument("--keep-cache", action="store_true", help="keep the extracted sequences after the build")
    ap.add_argument("--workers", type=int, default=80, help="parallel range requests")
    ap.add_argument("--jobs", type=int, default=5, help="clips built in parallel")
    ap.add_argument("--manifest", type=Path, default=HERE / "MANIFEST.sha256")
    ap.add_argument("--build-only", action="store_true",
                    help="no download: build from sequences already extracted in --cache (<cache>/subject1/<scene>/<seq>/cam4)")
    ap.add_argument("--write-manifest", action="store_true", help=argparse.SUPPRESS)  # maintainer
    a = ap.parse_args()
    clips = sorted(dict.fromkeys(a.clips), key=lambda c: ARCHIVE_ORDER.get(c, 99))
    cache = a.cache or a.out / ".cache"
    a.out.mkdir(parents=True, exist_ok=True)
    if not cache.exists():  # only a cache this script created is ever deleted
        cache.mkdir(parents=True)
        (cache / ".created_by_reproduce_h2o").write_text("delete me freely\n")
    t00 = time.time()
    log(f"1/4 downloading {len(clips)} sequences from the H2O server ...")
    if a.build_only:
        log("  --build-only: using the sequences already in the cache")
    else:
        download(clips, cache, lambda: auth_header(a.netrc), workers=a.workers)
    names = run_names(clips, str(cache))
    for c in clips:
        log(f"  {c} -> {names[c][0]}  (task label: {names[c][1]})")
    log("2/4 building data.npz, mano_fit.npz, frames, video.mp4, overlay.mp4, meta.json, README.md ...")
    with Pool(min(a.jobs, len(clips))) as p:
        res = p.map(build, [(*c.split("/"), str(cache), str(a.out), a.mano, *names[c]) for c in clips])
    log("3/4 rendering the floating 3D views ...")
    if a.mano:
        manos = {s: Mano(a.mano, s) for s in HANDS}
        for c in clips:
            render_world3d(a.out / names[c][0], manos)
            log(f"  rendered {names[c][0]}")
    else:
        log("  skipped (no --mano)")
    write_index(a.out, res)
    log(f"  peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB, "
        f"{(time.time() - t00) / 60:.1f} min so far")
    if a.write_manifest:
        write_manifest(a.out, a.manifest)
        log(f"wrote {a.manifest}")
        return
    log("4/4 checking against the manifest ...")
    ok = verify(a.out, a.manifest, [names[c][0] for c in clips], str(cache), bool(a.mano))
    if not a.keep_cache and ok and (cache / ".created_by_reproduce_h2o").exists():
        shutil.rmtree(cache)
        log(f"  deleted the extracted sequences in {cache} (use --keep-cache to keep them)")
    log(f"{'ALL OK' if ok else 'SOME FILES DIFFER'} in {(time.time() - t00) / 60:.1f} min")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
