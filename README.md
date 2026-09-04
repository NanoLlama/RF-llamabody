# RFantibody on Colab

A Google Colab notebook that runs the full **RFantibody** pipeline
(RFdiffusion-Ab → ProteinMPNN → RF2) to design **nanobodies (VHH)** against a
target of known structure, without Docker.

Upstream: [RosettaCommons/RFantibody](https://github.com/RosettaCommons/RFantibody)
· Bennett et al., *Atomically accurate de novo design of antibodies* (IPD / Baker lab)

| | |
|---|---|
| **Notebook** | [`notebooks/RFantibody_Colab.ipynb`](notebooks/RFantibody_Colab.ipynb) |
| **Verification notes** | [`docs/upstream-findings.md`](docs/upstream-findings.md) |
| **Verified against** | RFantibody `8fe3114` (2026-03-04) |

## Open in Colab

Push this repo to GitHub, then open:

```
https://colab.research.google.com/github/<owner>/<repo>/blob/main/notebooks/RFantibody_Colab.ipynb
```

Set the runtime to a GPU first (**Runtime → Change runtime type → GPU**). Cell 1
refuses to continue without one.

## What it does

Five cells, each independently re-runnable:

1. **Bootstrap** — installs `uv`, clones RFantibody, downloads weights (~3 GB),
   builds the isolated Python 3.10 venv, and smoke-tests all three entrypoints.
   Weights and the uv package cache live on Drive, so later sessions take
   ~1-2 min instead of ~10.
2. **Target preparation** — fetches from RCSB/AlphaFold or takes an upload,
   selects chains, crops, strips what the upstream parser cannot handle, and
   **validates that every hotspot actually exists in the prepared file**.
   Renders the epitope in py3Dmol.
3. **Parameters** — a form, with the fully-resolved command lines echoed back.
4. **Runner** — the three stages as decoupled, resumable steps writing to
   separate directories, with streamed logs, an atomically-written
   `manifest.json`, and a specific message for CUDA OOM.
5. **Results** — parses RF2 metrics into a DataFrame, filters on
   `interaction_pae < 10` and RMSD < 2 Å, plots the distribution against those
   thresholds, renders the top hits, and exports FASTA + CSV + zipped PDBs.

The notebook kernel is a **controller only** — it never imports `rfantibody`.
RFantibody pins Python 3.10 + torch 2.3+cu118 + a cu118 DGL wheel, which will
not co-exist with Colab's preinstalled Python and torch, so every stage runs as
a subprocess inside uv's own venv. cu118 binaries run fine on Colab's newer
T4/L4/A100 drivers.

## Read this before you plan a campaign

* **Throughput is the real constraint.** ~2-4 min/design end to end on a T4,
  dominated by RF2. A free Colab session yields low hundreds of designs. The
  RFantibody authors report that campaigns in the ~10k range are generally
  needed to find hits, because there is still no reliable *in silico* filter.
  Colab is right for method development, hotspot and CDR-length exploration,
  and small pilots — not for handing back three binders.
* **Session death is the default.** Resume is a requirement, not a nicety, and
  Cell 4 is built around two upstream resume bugs that otherwise *abort* a
  resumed run rather than just repeating work. See
  [`docs/upstream-findings.md`](docs/upstream-findings.md).
* **Hotspots are the main control, and they fail silently upstream.** A hotspot
  that matches nothing is ignored without warning, and you get undocked designs
  after hours of GPU. Cell 2 will not let you past that.
* **Target size is what blows up VRAM.** RFdiffusion and RF2 both scale as
  O(N²). Crop to the domain around your epitope, leaving ~10 Å of protein on
  each side of the target site.
* **Weights are non-commercial.** The RoseTTAFold-derived checkpoints are under
  the Rosetta-DL licence. The code is MIT.

## Repo layout

```
notebooks/RFantibody_Colab.ipynb   the notebook
tools/build_notebook.py            generates the notebook (edit here, not the .ipynb)
tools/verify_resume_bugs.py        reproduces the two upstream resume bugs
tools/test_target_prep.py          tests Cell 2 against the shipped example PDBs
tools/test_results_parse.py        tests Cell 5 against a synthetic RF2 output
docs/upstream-findings.md          what was verified, and what wasn't
```

The notebook is **generated**, so that every code cell is syntax-checked before
it can reach Colab. Edit `tools/build_notebook.py` and re-run it:

```bash
python3 tools/build_notebook.py
```

## Tests

The tests extract the relevant functions from the notebook itself, so they
check the code that actually ships:

```bash
git clone --depth 1 https://github.com/RosettaCommons/RFantibody.git
python3 tools/verify_resume_bugs.py ./RFantibody   # 2/2 bugs still present
python3 tools/test_target_prep.py   ./RFantibody   # all checks passed
python3 tools/test_results_parse.py ./RFantibody   # all checks passed
```

`verify_resume_bugs.py` reports `FIXED` if upstream has since repaired either
bug, at which point the corresponding workaround in Cell 3/4 can be dropped.

No GPU is needed for any of the above. What they do **not** cover — because the
build environment has no GPU and blocks the relevant hosts — is `uv sync` itself
and any actual model inference; Cell 1's smoke test is what proves those.
