# Data

| folder | what | data in the repo? |
|---|---|---|
| [`egoverse/trace/`](egoverse/trace/README.md) | EgoVerse clips from the `trace` lab, one folder per task | yes, CC BY-SA 4.0 |
| [`h2o/`](h2o/README.md) | H2O clips, one folder per task | no: H2O's terms forbid passing the data on; `h2o/reproduce_h2o.py` builds it locally from your own H2O registration |

Every run folder uses the same files and `data.npz` / `mano_fit.npz` keys, so code written for one dataset reads the others.
