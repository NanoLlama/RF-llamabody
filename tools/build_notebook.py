#!/usr/bin/env python3
"""Build notebooks/RFantibody_Colab.ipynb from the cell sources defined here.

The notebook is generated rather than hand-edited so that the Python in each
cell can be syntax-checked before it ever reaches Colab.  Run:

    python3 tools/build_notebook.py

Every code cell is compiled with compile() and the emitted file is re-parsed as
JSON as a final check.
"""

import json
import os
import sys

CELLS = []


def md(source):
    CELLS.append(("markdown", source))


def code(source):
    CELLS.append(("code", source))


# =============================================================================
# Title
# =============================================================================

md(r'''# RFantibody on Colab — nanobody design against a target of interest

Runs the full **RFdiffusion-Ab → ProteinMPNN → RF2** pipeline from
[RosettaCommons/RFantibody](https://github.com/RosettaCommons/RFantibody) on a
Colab GPU, without Docker.

**What this is for.** Method development, hotspot/CDR-length exploration, and
small pilot runs. It is *not* a way to get a handful of validated binders: the
RFantibody authors report that campaigns in the ~10k design range are generally
needed to find hits, because there is still no reliable *in silico* filter. A
free Colab session realistically produces low hundreds of designs. Treat the
output as input to a downstream display/screening campaign.

**Runtime.** Roughly 2-4 min/design end to end on a T4, dominated by RF2.

**Licence.** The RoseTTAFold-derived weights are distributed under the
Rosetta-DL licence — **non-commercial use only**. The code is MIT.

---

### How this notebook is organised

| Cell | Does | Re-runnable |
|---|---|---|
| 1 | Environment bootstrap (uv, clone, weights, `uv sync`) | yes, cached to Drive |
| 2 | Target preparation + validation + hotspot check | yes |
| 3 | Parameter form | yes |
| 4 | Pipeline runner (3 decoupled stages, resumable) | yes, resumes |
| 5 | Results parsing, filtering, visualisation, export | yes |

The notebook kernel is a **controller only**. It never imports `rfantibody`.
Every pipeline stage runs as a subprocess inside uv's own Python 3.10 venv,
because RFantibody pins Python 3.10 + torch 2.3+cu118 + a cu118 DGL wheel,
which will not co-exist with Colab's preinstalled Python/torch. cu118 binaries
run fine on Colab's newer drivers.

> **Two upstream bugs are worked around in Cell 4.** RFdiffusion's and RF2's
> "skip work that is already done" logic is broken for Quiver output, and
> hitting it does not just redo work — it aborts the run with
> `SystemExit: Tag ... already exists`. See `docs/upstream-findings.md` in this
> repo for the reproducer. Cell 4 avoids both; don't "simplify" it back.''')

# =============================================================================
# Cell 1 - bootstrap
# =============================================================================

md(r'''## Cell 1 — Environment bootstrap

Installs uv, clones RFantibody, downloads weights (~3 GB), and creates the
isolated venv. Weights and the uv package cache are stored on Drive, so a
second session takes ~1-2 min instead of ~10.

The venv itself is *not* cached to Drive: a uv venv hardlinks into the local
cache and is not reliably relocatable, and Drive's FUSE layer makes thousands
of small files slow. Caching `UV_CACHE_DIR` gives most of the speedup with
none of the flakiness.''')

code(r'''#@title Cell 1 — Environment bootstrap {display-mode: "form"}

#@markdown Cache weights + uv package cache to Google Drive (strongly recommended).
USE_DRIVE = True  #@param {type:"boolean"}
#@markdown Folder on Drive for the cache and for run outputs.
DRIVE_DIR = "/content/drive/MyDrive/rfantibody"  #@param {type:"string"}
#@markdown Git ref of RFantibody to check out. Pin a commit SHA for reproducibility.
REPO_REF = "main"  #@param {type:"string"}

import os, re, shutil, subprocess, sys, textwrap, time
from pathlib import Path

T0 = time.time()

def sh(cmd, check=True, env=None, cwd=None):
    """Run a shell command, streaming output to the notebook."""
    print(f"$ {cmd}")
    p = subprocess.run(cmd, shell=True, executable="/bin/bash",
                       env={**os.environ, **(env or {})}, cwd=cwd)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed (exit {p.returncode}): {cmd}")
    return p.returncode

IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")

# ---------------------------------------------------------------- GPU check
print("=" * 70)
gpu_name, gpu_mem_gb = None, 0.0
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    gpu_name, mem = [x.strip() for x in out.split(",")]
    gpu_mem_gb = float(re.sub(r"[^0-9.]", "", mem)) / 1024.0
    print(f"GPU: {gpu_name}  ({gpu_mem_gb:.1f} GB VRAM)")
except Exception as e:
    raise SystemExit(
        "No NVIDIA GPU found. RFantibody cannot run on CPU.\n"
        "In Colab: Runtime > Change runtime type > Hardware accelerator > GPU."
    ) from e

if gpu_mem_gb < 15:
    print(f"WARNING: only {gpu_mem_gb:.1f} GB VRAM. Keep the target under "
          f"~200 residues and expect OOM on larger systems.")
elif gpu_mem_gb < 24:
    print("Note: T4-class GPU. Keep the target under ~300 residues "
          "(RFdiffusion and RF2 both scale as O(N^2)).")
else:
    print("Plenty of VRAM — targets up to ~600 residues should be fine.")
print("=" * 70)

# ---------------------------------------------------------------- Drive
CACHE = None
if USE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE = Path(DRIVE_DIR)
    (CACHE / "weights").mkdir(parents=True, exist_ok=True)
    (CACHE / "uv_cache").mkdir(parents=True, exist_ok=True)
    (CACHE / "runs").mkdir(parents=True, exist_ok=True)
    os.environ["UV_CACHE_DIR"] = str(CACHE / "uv_cache")
    os.environ["RFAB_DRIVE"] = str(CACHE)
    print(f"Drive cache: {CACHE}")
else:
    print("Drive disabled — everything is ephemeral and dies with the session.")

# ---------------------------------------------------------------- uv
if shutil.which("uv") is None and not Path.home().joinpath(".local/bin/uv").exists():
    sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
os.environ["PATH"] = f"{Path.home()}/.local/bin:" + os.environ["PATH"]
sh("uv --version")

# ---------------------------------------------------------------- clone
REPO = Path("/content/RFantibody" if IN_COLAB else "./RFantibody").resolve()
if not (REPO / "pyproject.toml").exists():
    sh(f"git clone https://github.com/RosettaCommons/RFantibody.git {REPO}")
sh(f"git -C {REPO} fetch --depth 1 origin {REPO_REF} && "
   f"git -C {REPO} checkout --detach FETCH_HEAD")
HEAD = subprocess.run(f"git -C {REPO} rev-parse --short HEAD", shell=True,
                      capture_output=True, text=True).stdout.strip()
print(f"RFantibody at {HEAD}")

# ---------------------------------------------------------------- weights
WEIGHTS = REPO / "weights"
WEIGHTS.mkdir(exist_ok=True)
NEEDED = {
    "RFdiffusion_Ab.pt":            "https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt",
    "ProteinMPNN_v48_noise_0.2.pt": "https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt",
    "RF2_ab.pt":                    "https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt",
}
for fname, url in NEEDED.items():
    dest = WEIGHTS / fname
    if dest.exists() and dest.stat().st_size > 1_000_000:
        continue
    cached = (CACHE / "weights" / fname) if CACHE else None
    if cached and cached.exists() and cached.stat().st_size > 1_000_000:
        print(f"restoring {fname} from Drive")
        shutil.copy2(cached, dest)
    else:
        print(f"downloading {fname}")
        sh(f"wget -q --show-progress -O {dest} {url}")
        if cached:
            shutil.copy2(dest, cached)

missing = [f for f in NEEDED if not (WEIGHTS / f).exists()]
if missing:
    raise SystemExit(f"weights missing after download: {missing}")
print("weights OK:", ", ".join(sorted(NEEDED)))
# Note: download_weights.sh also fetches a 4th checkpoint used only for TCR-MHC
# prediction. It is ~2 GB and irrelevant to antibody/nanobody design, so this
# notebook skips it. Run `bash include/download_weights.sh` if you need it.

# ---------------------------------------------------------------- venv
sh(f"cd {REPO} && uv sync")

# ---------------------------------------------------------------- smoke test
print("\n" + "=" * 70)
print("SMOKE TEST")
for tool in ("rfdiffusion", "proteinmpnn", "rf2"):
    rc = sh(f"cd {REPO} && uv run {tool} --help > /dev/null", check=False)
    print(f"  {tool} --help -> exit {rc}")
    if rc != 0:
        raise SystemExit(f"{tool} is not runnable; see the output above.")

# Import torch/dgl inside the venv via a file, not a -c one-liner: quoting a
# one-liner through the shell is a needless way to break this check.
_check = REPO / "_env_check.py"
_check.write_text(
    "import torch, dgl\n"
    "print('torch', torch.__version__)\n"
    "print('dgl  ', dgl.__version__)\n"
    "print('cuda available:', torch.cuda.is_available())\n"
    "assert torch.cuda.is_available(), 'torch cannot see the GPU'\n"
    "print('device:', torch.cuda.get_device_name(0))\n"
)
rc = sh(f"cd {REPO} && uv run python {_check.name}", check=False)
_check.unlink(missing_ok=True)
if rc != 0:
    raise SystemExit(
        "torch/dgl failed to import inside the venv, or torch cannot see the "
        "GPU. Re-run this cell; if it persists, delete the venv "
        f"(rm -rf {REPO}/.venv) and let uv sync rebuild it.")

os.environ["RFANTIBODY_REPO"] = str(REPO)
print(f"\nReady in {time.time() - T0:.0f}s. Repo: {REPO}")
print("=" * 70)''')

# =============================================================================
# Cell 2 - target prep
# =============================================================================

md(r'''## Cell 2 — Target preparation

Produces a target PDB that RFantibody's parser will accept, and — more
importantly — **checks that your hotspots actually exist in it**.

Three things about the upstream parser drive everything here
(`src/rfantibody/rfdiffusion/parsers.py:HLT_pdb_parser`):

1. It reads only lines starting with `ATOM`. HETATM, waters and ligands are
   ignored, but a non-standard residue written as `ATOM` will raise `KeyError`.
2. It keys residues on `(chain_id, resnum)` taken from columns 22 and 23-26 —
   **the insertion code is not read**. Two residues like `100` and `100A`
   collapse onto the same key and the second silently overwrites the first's
   coordinates. Insertion codes must be renumbered away.
3. Alternate locations are not filtered, so altloc B atoms overwrite altloc A.

**Hotspots use the chain IDs and residue numbers exactly as they appear in
your prepared target file.** They are *not* required to be chain `T` — the
upstream `flu_HA` example uses `B146,B170,B177`. The target chain is renamed to
`T` internally, after hotspot matching. This notebook therefore keeps your
original chain IDs, which also avoids a real corruption risk: renaming two
target chains to `T` when their residue numbering overlaps produces duplicate
`(T, resnum)` keys and mangles the structure via point 2 above.

A hotspot that matches nothing is **silently ignored** upstream — no error, and
you get undocked nonsense after burning an hour of GPU. The check below is the
single most valuable thing in this cell.''')

code(r'''#@title Cell 2 — Target preparation {display-mode: "form"}

#@markdown **Source of the target structure.**
SOURCE = "rcsb"  #@param ["rcsb", "alphafold", "upload", "example"]
#@markdown RCSB PDB ID (e.g. `4YKN`) or AlphaFold accession (e.g. `P00533`).
ACCESSION = "4YKN"  #@param {type:"string"}
#@markdown Chain(s) to keep, comma separated (e.g. `A` or `A,B`). Leave blank to keep all.
KEEP_CHAINS = "A"  #@param {type:"string"}
#@markdown Residue range to crop to, e.g. `A:300-400`. Blank = no crop. Repeat with commas.
CROP = ""  #@param {type:"string"}
#@markdown Hotspots defining the epitope, e.g. `A305,A307,A356`. Chain letters must match the file.
HOTSPOTS = "A305,A307,A356"  #@param {type:"string"}

import glob, gzip, io, os, shutil, subprocess, sys, urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path

REPO = Path(os.environ.get("RFANTIBODY_REPO", "/content/RFantibody"))
if not REPO.exists():
    raise SystemExit("Run Cell 1 first.")

WORK = Path("/content/work") if Path("/content").is_dir() else Path("./work")
WORK.mkdir(parents=True, exist_ok=True)

STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
BACKBONE = ("N", "CA", "C", "O")


def fetch_structure(source, accession, dest):
    if source == "example":
        src = REPO / "scripts/examples/example_inputs/rsv_site3.pdb"
        shutil.copy2(src, dest)
        print(f"using shipped example target: {src.name}")
        return dest
    if source == "upload":
        from google.colab import files
        up = files.upload()
        name = list(up)[0]
        Path(dest).write_bytes(up[name])
        print(f"uploaded {name}")
        return dest
    if source == "rcsb":
        url = f"https://files.rcsb.org/download/{accession.strip().upper()}.pdb"
    elif source == "alphafold":
        acc = accession.strip().upper()
        url = f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"
    else:
        raise ValueError(source)
    print(f"fetching {url}")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    Path(dest).write_bytes(data)
    return dest


def parse_crop(spec):
    """'A:300-400,B:10-90' -> {'A': [(300,400)], 'B': [(10,90)]}"""
    out = defaultdict(list)
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        chain, _, rng = part.partition(":")
        lo, _, hi = rng.partition("-")
        out[chain.strip()].append((int(lo), int(hi)))
    return out


def clean_target(raw_path, out_path, keep_chains, crop_spec):
    """Write a parser-safe target PDB. Returns (residues, warnings)."""
    keep = {c.strip() for c in keep_chains.split(",") if c.strip()}
    crops = parse_crop(crop_spec)
    warnings = []

    # First pass: collect residues in file order, applying every filter.
    residues = OrderedDict()          # (chain, resnum, icode) -> [lines]
    dropped = defaultdict(int)
    for line in Path(raw_path).read_text().splitlines():
        rec = line[:6]
        if rec.startswith("ENDMDL"):
            break                      # NMR / multi-model: keep model 1 only
        if not line.startswith("ATOM"):
            if line.startswith("HETATM"):
                dropped["hetatm"] += 1
            continue
        chain = line[21]
        resname = line[17:20].strip()
        altloc = line[16]
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if keep and chain not in keep:
            dropped["other_chain"] += 1
            continue
        if resname not in STD_AA:
            dropped["nonstandard"] += 1
            continue
        if altloc not in (" ", "A"):
            dropped["altloc"] += 1
            continue
        if element == "H" or line[12:16].strip().startswith("H"):
            dropped["hydrogen"] += 1
            continue
        resnum = int(line[22:26])
        if chain in crops and not any(lo <= resnum <= hi for lo, hi in crops[chain]):
            dropped["cropped"] += 1
            continue
        icode = line[26]
        residues.setdefault((chain, resnum, icode), []).append(line)

    if not residues:
        raise SystemExit(
            "No residues survived filtering. Check KEEP_CHAINS and CROP "
            f"against the file (dropped: {dict(dropped)})."
        )

    # Insertion codes and duplicate (chain,resnum) keys both break the upstream
    # parser, so renumber each chain contiguously from 1.
    had_icode = any(k[2] != " " for k in residues)
    seen = defaultdict(set)
    dup = False
    for chain, resnum, icode in residues:
        if resnum in seen[chain]:
            dup = True
        seen[chain].add(resnum)
    renumber = had_icode or dup
    if renumber:
        warnings.append(
            "Renumbered residues from 1 per chain: the input had "
            + ("insertion codes" if had_icode else "duplicate residue numbers")
            + ", which the upstream parser cannot represent. "
              "Specify hotspots using the NEW numbering printed below."
        )

    counters = defaultdict(int)
    mapping = {}                       # old key -> new resnum
    out_lines, serial = [], 0
    for key, lines in residues.items():
        chain, resnum, icode = key
        if renumber:
            counters[chain] += 1
            new_num = counters[chain]
        else:
            new_num = resnum
        mapping[key] = new_num
        names = {ln[12:16].strip() for ln in lines}
        missing = [a for a in BACKBONE if a not in names]
        if missing:
            warnings.append(f"{chain}{new_num} missing backbone atom(s) {missing} — dropped")
            continue
        for ln in lines:
            serial += 1
            # Rebuild in strict PDB column order, blanking the altloc (col 17)
            # and the insertion code (col 27) that the upstream parser ignores.
            out_lines.append(
                f"{ln[:6]}{serial:5d}{ln[11:16]} {ln[17:22]}{new_num:4d} {ln[27:]}"
            )
    out_lines.append("TER")
    out_lines.append("END")
    Path(out_path).write_text("\n".join(out_lines) + "\n")

    print(f"dropped atoms: {dict(dropped) or 'none'}")
    return warnings


def residue_index(pdb_path):
    """{(chain, resnum): one-letter} for a cleaned file, as the parser sees it."""
    idx = OrderedDict()
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            idx[(line[21], int(line[22:26]))] = THREE_TO_ONE.get(line[17:20].strip(), "X")
    return idx


def check_chain_breaks(pdb_path, cutoff=4.5):
    """Consecutive-CA distances above cutoff indicate a chain break."""
    import math
    cas = defaultdict(list)
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            cas[line[21]].append(
                (int(line[22:26]),
                 (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    breaks = []
    for chain, lst in cas.items():
        for (n1, a), (n2, b) in zip(lst, lst[1:]):
            d = math.dist(a, b)
            if d > cutoff:
                breaks.append((chain, n1, n2, d))
    return breaks


# ----------------------------------------------------------------- run it
raw = WORK / "target_raw.pdb"
TARGET_PDB = WORK / "target.pdb"
fetch_structure(SOURCE, ACCESSION, raw)

warns = clean_target(raw, TARGET_PDB, KEEP_CHAINS, CROP)
index = residue_index(TARGET_PDB)
n_res = len(index)
chains = sorted({c for c, _ in index})

print("\n" + "=" * 70)
print(f"prepared target : {TARGET_PDB}")
print(f"chains          : {', '.join(chains)}")
print(f"residues        : {n_res}")
for c in chains:
    nums = [r for ch, r in index if ch == c]
    print(f"  chain {c}: {len(nums)} residues, numbered {min(nums)}-{max(nums)}")
for w in warns:
    print(f"WARNING: {w}")

breaks = check_chain_breaks(TARGET_PDB)
if breaks:
    print(f"NOTE: {len(breaks)} chain break(s) detected "
          f"(missing loops in the deposited structure):")
    for chain, a, b, d in breaks[:8]:
        print(f"  chain {chain}: {a} -> {b} is {d:.1f} A apart")
    print("  Breaks are tolerated, but keep the epitope away from them.")

# size warning, tied to the GPU actually allocated
try:
    _mem = gpu_mem_gb
except NameError:
    _mem = 15.0
budget = 200 if _mem < 15 else (300 if _mem < 24 else 600)
if n_res > budget:
    print(f"\nWARNING: {n_res} residues is large for a {_mem:.0f} GB GPU "
          f"(~{budget} is the practical limit). RFdiffusion and RF2 both scale "
          f"as O(N^2) and RF2 will likely OOM. Use CROP to trim to the domain "
          f"around your epitope — aim to leave ~10 A of protein on each side of "
          f"the target site.")

# ------------------------------------------------------- HOTSPOT VALIDATION
print("\n" + "-" * 70)
print("HOTSPOT CHECK  (an unmatched hotspot is silently ignored upstream)")
requested = [h.strip() for h in HOTSPOTS.split(",") if h.strip()]
good, bad = [], []
for h in requested:
    if not h[0].isalpha():
        bad.append((h, "must start with a chain letter, e.g. A305"))
        continue
    try:
        key = (h[0], int(h[1:]))
    except ValueError:
        bad.append((h, "residue number is not an integer"))
        continue
    if key in index:
        good.append(h)
        print(f"  OK   {h}  ({index[key]})")
    elif h[0] not in chains:
        bad.append((h, f"chain '{h[0]}' is not in the prepared target "
                       f"(it has chain(s): {', '.join(chains)})"))
    else:
        near = [f"{c}{r}" for c, r in index
                if c == h[0] and abs(r - int(h[1:])) <= 3]
        bad.append((h, f"residue {h[1:]} is not in chain {h[0]}"
                       + (f"; nearby: {', '.join(near)}" if near else "")))

for h, why in bad:
    print(f"  FAIL {h}  — {why}")

if bad:
    raise SystemExit(
        f"{len(bad)} hotspot(s) do not exist in the prepared target. Fix "
        f"HOTSPOTS (or KEEP_CHAINS/CROP) and re-run — do not continue, the "
        f"model would receive fewer hotspots than you think and produce "
        f"undocked designs."
    )
if not good:
    raise SystemExit("No hotspots given. RFantibody needs them to place the binder.")
if len(good) < 3:
    print(f"NOTE: only {len(good)} hotspot(s). 3-5 contiguous surface residues "
          f"is the usual choice; too few tends to give undocked designs.")
print(f"{len(good)} hotspot(s) validated.")

FRAMEWORK_PDB = REPO / "scripts/examples/example_inputs/h-NbBCII10.pdb"
assert FRAMEWORK_PDB.exists(), FRAMEWORK_PDB
print(f"\nnanobody framework: {FRAMEWORK_PDB.name}  (h-NbBCII10, humanised VHH)")
print("nanobody mode = chains H and T only; there is no L chain.")
print("=" * 70)

os.environ["RFAB_TARGET"] = str(TARGET_PDB)
os.environ["RFAB_FRAMEWORK"] = str(FRAMEWORK_PDB)
os.environ["RFAB_HOTSPOTS"] = ",".join(good)''')

md(r'''### Visual check of the epitope

Burn thirty seconds here rather than an hour of GPU time on the wrong face of
the protein. Hotspots are drawn as red spheres.''')

code(r'''#@title Cell 2b — Render the target and hotspots
try:
    import py3Dmol
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py3Dmol"], check=True)
    import py3Dmol

view = py3Dmol.view(width=900, height=550)
view.addModel(open(os.environ["RFAB_TARGET"]).read(), "pdb")
view.setStyle({"cartoon": {"color": "lightgrey"}})
for h in os.environ["RFAB_HOTSPOTS"].split(","):
    view.addStyle({"chain": h[0], "resi": int(h[1:])},
                  {"stick": {"colorscheme": "redCarbon", "radius": 0.3}})
    view.addStyle({"chain": h[0], "resi": int(h[1:])},
                  {"sphere": {"color": "red", "opacity": 0.55}})
view.zoomTo()
view.show()
print("Red = your hotspots. They should sit together on one solvent-exposed "
      "face, ideally with >=3 hydrophobics. Charged/polar-only sites and sites "
      "near glycans are much harder to bind.")''')

# =============================================================================
# Cell 3 - params
# =============================================================================

md(r'''## Cell 3 — Run parameters

Defaults follow `scripts/examples/nanobody_full_pipeline.sh`, scaled down for
Colab. The command lines are echoed fully resolved so you can see exactly what
will run.

`H3` length is the dominant lever on the design — sampling a range (rather than
one length) is standard practice. `num_recycles` is the main RF2 speed knob.''')

code(r'''#@title Cell 3 — Parameters {display-mode: "form"}

#@markdown Name for this run (a folder under the Drive `runs/` directory).
RUN_NAME = "run01"  #@param {type:"string"}
#@markdown **RFdiffusion** — number of backbones. 2-4 min/design end to end on a T4.
NUM_DESIGNS = 20  #@param {type:"integer"}
#@markdown CDR loop lengths. Nanobody default from the upstream example.
DESIGN_LOOPS = "H1:7,H2:6,H3:5-13"  #@param {type:"string"}
#@markdown Diffusion timesteps (upstream default 50).
DIFFUSER_T = 50  #@param {type:"integer"}
#@markdown Final diffusion step (upstream default 1).
FINAL_STEP = 1  #@param {type:"integer"}
#@markdown **ProteinMPNN** — sequences per backbone, and sampling temperature.
SEQS_PER_STRUCT = 4  #@param {type:"integer"}
MPNN_TEMPERATURE = 0.2  #@param {type:"number"}
#@markdown **RF2** — recycles (main speed/accuracy knob) and hotspot visibility.
NUM_RECYCLES = 10  #@param {type:"integer"}
HOTSPOT_SHOW_PROP = 0.0  #@param {type:"number"}
#@markdown Deterministic mode (reproducible, slightly slower).
DETERMINISTIC = True  #@param {type:"boolean"}

import json, os, shlex
from pathlib import Path

REPO = Path(os.environ["RFANTIBODY_REPO"])
TARGET = os.environ["RFAB_TARGET"]
FRAMEWORK = os.environ["RFAB_FRAMEWORK"]
HOTSPOTS_RESOLVED = os.environ["RFAB_HOTSPOTS"]

_drive = os.environ.get("RFAB_DRIVE")
_base = Path(_drive) / "runs" if _drive and Path(_drive).exists() else Path("/content/runs")
RUN_DIR = _base / RUN_NAME
for sub in ("01_diffusion", "02_mpnn", "03_rf2", "logs"):
    (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)

# Slash-free, so RFdiffusion's resume check works. See docs/upstream-findings.md.
DIFF_PREFIX = "nbdes"
DIFF_QV = RUN_DIR / "01_diffusion" / "diffusion.qv"
MPNN_QV = RUN_DIR / "02_mpnn" / "mpnn.qv"
RF2_DIR = RUN_DIR / "03_rf2"

CMD_DIFFUSION = [
    "uv", "run", "rfdiffusion",
    "--target", str(TARGET),
    "--framework", str(FRAMEWORK),
    "--output-quiver", str(DIFF_QV),
    "--num-designs", str(NUM_DESIGNS),
    "--design-loops", DESIGN_LOOPS,
    "--hotspots", HOTSPOTS_RESOLVED,
    "--diffuser-t", str(DIFFUSER_T),
    "--final-step", str(FINAL_STEP),
    "--no-trajectory",
    "-e", f"inference.output_prefix={DIFF_PREFIX}",
] + (["--deterministic"] if DETERMINISTIC else [])

CMD_MPNN = [
    "uv", "run", "proteinmpnn",
    "--input-quiver", str(DIFF_QV),
    "--output-quiver", str(MPNN_QV),
    "--loops", "H1,H2,H3",
    "--seqs-per-struct", str(SEQS_PER_STRUCT),
    "--temperature", str(MPNN_TEMPERATURE),
] + (["--deterministic"] if DETERMINISTIC else [])

CMD_RF2 = [
    "uv", "run", "rf2",
    "--input-quiver", str(MPNN_QV),
    "--output-dir", str(RF2_DIR),
    "--num-recycles", str(NUM_RECYCLES),
    "--hotspot-show-prop", str(HOTSPOT_SHOW_PROP),
    "--cautious",
]

PARAMS = dict(
    run_name=RUN_NAME, run_dir=str(RUN_DIR), target=str(TARGET),
    framework=str(FRAMEWORK), hotspots=HOTSPOTS_RESOLVED,
    num_designs=NUM_DESIGNS, design_loops=DESIGN_LOOPS,
    diffuser_t=DIFFUSER_T, final_step=FINAL_STEP,
    seqs_per_struct=SEQS_PER_STRUCT, mpnn_temperature=MPNN_TEMPERATURE,
    num_recycles=NUM_RECYCLES, hotspot_show_prop=HOTSPOT_SHOW_PROP,
    deterministic=DETERMINISTIC,
)
(RUN_DIR / "params.json").write_text(json.dumps(PARAMS, indent=2))

def show(name, cmd):
    print(f"\n# {name}")
    print("  " + " \\\n    ".join(shlex.quote(c) for c in cmd))

print("=" * 70)
print(f"run directory: {RUN_DIR}")
print(f"total RF2 predictions to run: "
      f"{NUM_DESIGNS * SEQS_PER_STRUCT}  "
      f"({NUM_DESIGNS} backbones x {SEQS_PER_STRUCT} seqs)")
print(f"rough wall-clock estimate: "
      f"{NUM_DESIGNS * SEQS_PER_STRUCT * 2.5 / 60:.1f}-"
      f"{NUM_DESIGNS * SEQS_PER_STRUCT * 4.0 / 60:.1f} h on a T4")
show("Stage 1 - RFdiffusion", CMD_DIFFUSION)
show("Stage 2 - ProteinMPNN", CMD_MPNN)
show("Stage 3 - RF2", CMD_RF2)
print("\n" + "=" * 70)''')

# =============================================================================
# Cell 4 - runner
# =============================================================================

md(r'''## Cell 4 — Pipeline runner

Three decoupled stages, each independently re-runnable, each resuming from
whatever is already on disk. Set `STAGES` to re-run just one — RF2 is the
expensive stage and you will want to re-run it against existing MPNN output
without re-diffusing.

**Resume semantics, and why the commands look the way they do:**

* **RFdiffusion** skips a design when its tag is already in the output quiver,
  but the check compares `inference.output_prefix + "_N"` against tags that had
  `/` replaced by `_`. With the default prefix `samples/design` the comparison
  can never match, so a resumed run re-generates design 0 and then dies in
  `Quiver.add_pdb` with `SystemExit: Tag samples_design_0 already exists`.
  Cell 3 passes `-e inference.output_prefix=nbdes` — no slash, so the check
  works and resume is clean.
* **ProteinMPNN** resumes correctly on its own (it strips the `_dldesign_N`
  suffix off existing output tags). Nothing to work around.
* **RF2** with `--output-quiver` has the same class of bug: `get_done_list()`
  returns the written tags, which carry a `_best` suffix, and compares them
  against un-suffixed input tags. Never matches, so a resumed run re-predicts
  and then dies on the duplicate tag. With `--output-dir` the same function
  strips `_best` and resume works correctly, so this notebook writes RF2 output
  as PDBs. That also makes the per-design `SCORE` lines trivial to parse in
  Cell 5.

Both bugs are verified against upstream in `docs/upstream-findings.md`.''')

code(r'''#@title Cell 4 — Run the pipeline {display-mode: "form"}

#@markdown Which stages to run. Stages are decoupled; re-running is safe and resumes.
STAGES = "all"  #@param ["all", "diffusion", "mpnn", "rf2", "mpnn+rf2"]

import json, os, re, signal, subprocess, sys, time
from datetime import datetime
from pathlib import Path

LOGS = RUN_DIR / "logs"
MANIFEST = RUN_DIR / "manifest.json"

OOM_PAT = re.compile(r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|"
                     r"torch\.cuda\.OutOfMemoryError", re.I)


def quiver_tags(path):
    """Tags currently in a quiver file (cheap: one scan, no parsing)."""
    p = Path(path)
    if not p.exists():
        return []
    return [line.split(maxsplit=1)[1].strip()
            for line in p.read_text(errors="ignore").splitlines()
            if line.startswith("QV_TAG")]


def rf2_done(rf2_dir):
    return sorted(p.stem[:-5] for p in Path(rf2_dir).glob("*_best.pdb"))


def load_manifest():
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            pass
    return {"run": RUN_NAME, "params": PARAMS, "stages": {}}


def save_manifest(m):
    m["updated"] = datetime.now().isoformat(timespec="seconds")
    m["progress"] = {
        "diffusion_backbones": len(quiver_tags(DIFF_QV)),
        "mpnn_sequences": len(quiver_tags(MPNN_QV)),
        "rf2_predictions": len(rf2_done(RF2_DIR)),
    }
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(MANIFEST)          # atomic: a killed session never truncates it


def run_stage(name, cmd, progress_fn=None):
    """Stream a stage to the notebook and to logs/, watching for OOM."""
    manifest = load_manifest()
    log_path = LOGS / f"{name}.log"
    started = time.time()
    manifest["stages"][name] = {"status": "running",
                                "started": datetime.now().isoformat(timespec="seconds"),
                                "cmd": cmd}
    save_manifest(manifest)

    print(f"\n{'=' * 70}\n[{name}] starting  (log: {log_path})\n{'=' * 70}")
    oom, tail = False, []
    with open(log_path, "a") as log:
        log.write(f"\n\n### {datetime.now().isoformat()}\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            last_beat = time.time()
            for line in proc.stdout:
                log.write(line)
                tail.append(line)
                del tail[:-40]
                if OOM_PAT.search(line):
                    oom = True
                # Keep the notebook readable: echo signal, not every hydra line.
                if any(k in line for k in ("[RF2]", "Making design", "Finished design",
                                           "reported success", "Attempting pose",
                                           "Error", "error", "Traceback", "Skipping")):
                    print(line.rstrip())
                elif time.time() - last_beat > 30:
                    if progress_fn:
                        print(f"  ... {progress_fn()}")
                    last_beat = time.time()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            manifest = load_manifest()
            manifest["stages"][name]["status"] = "interrupted"
            save_manifest(manifest)
            print(f"[{name}] interrupted — re-run this cell to resume.")
            raise
        proc.wait()

    manifest = load_manifest()
    elapsed = time.time() - started
    if proc.returncode != 0:
        manifest["stages"][name] = {**manifest["stages"][name],
                                    "status": "failed",
                                    "returncode": proc.returncode,
                                    "seconds": round(elapsed)}
        save_manifest(manifest)
        print("".join(tail[-25:]))
        if oom:
            raise SystemExit(
                f"\n[{name}] CUDA OUT OF MEMORY.\n"
                f"  This is target size, almost always. In order of effect:\n"
                f"   1. Crop the target further in Cell 2 (CROP) — runtime and\n"
                f"      memory both scale as O(N^2). Leave ~10 A around the epitope.\n"
                f"   2. Lower NUM_RECYCLES in Cell 3 (10 -> 3) for the RF2 stage.\n"
                f"   3. Switch to an L4/A100 runtime (Colab Pro).\n"
                f"  Nothing is lost — re-run this cell and it resumes.")
        raise SystemExit(
            f"[{name}] failed with exit code {proc.returncode}. "
            f"Full log: {log_path} (last lines printed above). "
            f"Re-running this cell resumes from where it stopped.")

    manifest["stages"][name] = {**manifest["stages"][name],
                                "status": "complete",
                                "seconds": round(elapsed)}
    save_manifest(manifest)
    print(f"[{name}] complete in {elapsed / 60:.1f} min")
    return elapsed


# ------------------------------------------------------------------ stages
want = {"all": {"diffusion", "mpnn", "rf2"},
        "diffusion": {"diffusion"},
        "mpnn": {"mpnn"},
        "rf2": {"rf2"},
        "mpnn+rf2": {"mpnn", "rf2"}}[STAGES]

t_start = time.time()

if "diffusion" in want:
    have = len(quiver_tags(DIFF_QV))
    if have >= NUM_DESIGNS:
        print(f"[diffusion] {have}/{NUM_DESIGNS} backbones already present — skipping.")
    else:
        if have:
            print(f"[diffusion] resuming: {have}/{NUM_DESIGNS} backbones already done.")
        run_stage("diffusion", CMD_DIFFUSION,
                  progress_fn=lambda: f"{len(quiver_tags(DIFF_QV))}/{NUM_DESIGNS} backbones")

if "mpnn" in want:
    n_bb = len(quiver_tags(DIFF_QV))
    if n_bb == 0:
        raise SystemExit("No backbones in the diffusion quiver — run the diffusion stage first.")
    expect = n_bb * SEQS_PER_STRUCT
    if len(quiver_tags(MPNN_QV)) >= expect:
        print(f"[mpnn] {expect} sequences already present — skipping.")
    else:
        run_stage("mpnn", CMD_MPNN,
                  progress_fn=lambda: f"{len(quiver_tags(MPNN_QV))}/{expect} sequences")

if "rf2" in want:
    n_seq = len(quiver_tags(MPNN_QV))
    if n_seq == 0:
        raise SystemExit("No sequences in the MPNN quiver — run the mpnn stage first.")
    done = len(rf2_done(RF2_DIR))
    if done >= n_seq:
        print(f"[rf2] {done}/{n_seq} predictions already present — skipping.")
    else:
        if done:
            print(f"[rf2] resuming: {done}/{n_seq} predictions already done.")
        run_stage("rf2", CMD_RF2,
                  progress_fn=lambda: f"{len(rf2_done(RF2_DIR))}/{n_seq} predictions")

save_manifest(load_manifest())
m = load_manifest()
print("\n" + "=" * 70)
print(f"total elapsed this cell: {(time.time() - t_start) / 60:.1f} min")
print(f"backbones   : {m['progress']['diffusion_backbones']}")
print(f"sequences   : {m['progress']['mpnn_sequences']}")
print(f"predictions : {m['progress']['rf2_predictions']}")
print(f"manifest    : {MANIFEST}")
print("=" * 70)''')

# =============================================================================
# Cell 5 - results
# =============================================================================

md(r'''## Cell 5 — Results, filtering, export

RF2 writes one `<tag>_best.pdb` per design with its metrics as `SCORE key: value`
lines, which is what gets parsed here.

The filter is the one the RFantibody authors recommend:

* **`interaction_pae` < 10** — RF2's predicted aligned error across the
  nanobody/target interface. This is the "pAE_interaction" of the spec; that
  exact name does not exist upstream.
* **`target_aligned_antibody_rmsd` < 2 Å** — the RF2 prediction versus the
  design model, after superimposing on the target. This is the "design vs
  predicted RMSD" of the filtering guidance.

A caveat worth keeping in front of you: the authors are explicit that the lack
of a reliable filter is the pipeline's main limitation, and that RF2 shows at
best weak enrichment of binders over non-binders. Passing designs are
candidates to screen, not predicted binders.''')

code(r'''#@title Cell 5 — Parse, filter, visualise, export {display-mode: "form"}

#@markdown Filter thresholds (upstream recommendation: 10 and 2.0).
PAE_CUTOFF = 10.0  #@param {type:"number"}
RMSD_CUTOFF = 2.0  #@param {type:"number"}
#@markdown How many top designs to render.
TOP_N = 5  #@param {type:"integer"}

import json, os, re, zipfile
from pathlib import Path

import pandas as pd

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_rf2_pdb(path):
    """Scores, chain sequences and CDR sequences from one RF2 output PDB."""
    rec = {"design": Path(path).stem, "path": str(path)}
    residues, labels = [], {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("SCORE"):
            key, _, val = line[6:].partition(":")
            try:
                rec[key.strip()] = float(val)
            except ValueError:
                rec[key.strip()] = val.strip()
        elif line.startswith("REMARK PDBinfo-LABEL:"):
            # 1-indexed ABSOLUTE residue position, not per-chain
            parts = line.split()
            labels[int(parts[2])] = parts[3]
        elif line.startswith("ATOM") and line[12:16].strip() == "CA":
            residues.append((line[21], THREE_TO_ONE.get(line[17:20].strip(), "X")))

    chains = {}
    for ch, aa in residues:
        chains[ch] = chains.get(ch, "") + aa
    rec["H_seq"] = chains.get("H", "")
    rec["H_len"] = len(rec["H_seq"])
    rec["target_len"] = len(chains.get("T", ""))

    cdrs = {}
    for abs_i, loop in labels.items():
        if 1 <= abs_i <= len(residues):
            cdrs.setdefault(loop, "")
            cdrs[loop] += residues[abs_i - 1][1]
    for loop in ("H1", "H2", "H3"):
        rec[loop] = cdrs.get(loop, "")
        rec[f"{loop}_len"] = len(rec[loop])
    return rec


pdbs = sorted(Path(RF2_DIR).glob("*_best.pdb"))
if not pdbs:
    raise SystemExit(f"No RF2 outputs in {RF2_DIR}. Run Cell 4's rf2 stage first.")

df = pd.DataFrame([parse_rf2_pdb(p) for p in pdbs])

PAE_COL = "interaction_pae"
RMSD_COL = ("target_aligned_antibody_rmsd"
            if "target_aligned_antibody_rmsd" in df.columns else None)
if PAE_COL not in df.columns:
    raise SystemExit(f"No '{PAE_COL}' in the RF2 output. Columns: {list(df.columns)}")
if RMSD_COL is None:
    print("NOTE: no RMSD columns. RF2 skips RMSD when the target or framework "
          "length differs between the design and the prediction. Filtering on "
          "pAE alone.")

df = df.sort_values(PAE_COL).reset_index(drop=True)
mask = df[PAE_COL] < PAE_CUTOFF
if RMSD_COL:
    mask &= df[RMSD_COL] < RMSD_CUTOFF
df["passes"] = mask
passing = df[mask]

print("=" * 70)
print(f"designs predicted : {len(df)}")
print(f"passing filter    : {len(passing)}  "
      f"({PAE_COL} < {PAE_CUTOFF}"
      + (f" and {RMSD_COL} < {RMSD_CUTOFF}" if RMSD_COL else "") + ")")
if len(df):
    print(f"best {PAE_COL}   : {df[PAE_COL].min():.2f}")
    if RMSD_COL:
        print(f"best RMSD         : {df[RMSD_COL].min():.2f} A")
if len(passing) == 0:
    print("\nNothing passed. That is a normal outcome for a small pilot run. "
          "Worth trying: a different hotspot set (the model is very sensitive "
          "to this), a wider H3 range, or simply more designs.")
print("=" * 70)

cols = [c for c in ["design", PAE_COL, RMSD_COL, "pred_lddt",
                    "H3", "H3_len", "H1", "H2", "passes"] if c and c in df.columns]
try:
    display(df[cols].head(30))
except NameError:          # not in IPython
    print(df[cols].head(30).to_string())

# ------------------------------------------------------------------- plot
import matplotlib.pyplot as plt

if RMSD_COL:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df.loc[~mask, RMSD_COL], df.loc[~mask, PAE_COL],
               s=42, c="#b9c0c8", edgecolor="none", label="filtered out")
    ax.scatter(df.loc[mask, RMSD_COL], df.loc[mask, PAE_COL],
               s=52, c="#2b7bba", edgecolor="white", linewidth=0.6, label="passes")
    ax.axvline(RMSD_CUTOFF, color="#c0392b", ls="--", lw=1.2)
    ax.axhline(PAE_CUTOFF, color="#c0392b", ls="--", lw=1.2)
    ax.set_xlabel(f"{RMSD_COL} (A)   — design vs RF2 prediction")
    ax.set_ylabel(f"{PAE_COL}   — RF2 interface pAE")
    ax.set_title(f"{RUN_NAME}: {len(passing)}/{len(df)} pass")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()
else:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[PAE_COL], bins=25, color="#2b7bba")
    ax.axvline(PAE_CUTOFF, color="#c0392b", ls="--")
    ax.set_xlabel(PAE_COL)
    ax.set_ylabel("designs")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------- 3D viewer
try:
    import py3Dmol
    show_df = passing if len(passing) else df
    for _, row in show_df.head(TOP_N).iterrows():
        print(f"\n{row['design']}   {PAE_COL}={row[PAE_COL]:.2f}"
              + (f"  rmsd={row[RMSD_COL]:.2f}" if RMSD_COL else "")
              + f"\n  H3 = {row['H3']}")
        v = py3Dmol.view(width=800, height=460)
        v.addModel(Path(row["path"]).read_text(), "pdb")
        v.setStyle({"chain": "T"}, {"cartoon": {"color": "lightgrey"}})
        v.setStyle({"chain": "H"}, {"cartoon": {"color": "#2b7bba"}})
        for h in os.environ["RFAB_HOTSPOTS"].split(","):
            v.addStyle({"chain": "T", "resi": int(h[1:])},
                       {"sphere": {"color": "red", "opacity": 0.5}})
        v.zoomTo()
        v.show()
except ImportError:
    print("py3Dmol not installed — skipping 3D view.")

# ----------------------------------------------------------------- export
EXPORT = RUN_DIR / "results"
EXPORT.mkdir(exist_ok=True)

df.to_csv(EXPORT / "all_designs.csv", index=False)
passing.to_csv(EXPORT / "passing_designs.csv", index=False)

with open(EXPORT / "passing_designs.fasta", "w") as fa:
    for _, r in passing.iterrows():
        fa.write(f">{r['design']} {PAE_COL}={r[PAE_COL]:.2f}"
                 + (f" rmsd={r[RMSD_COL]:.2f}" if RMSD_COL else "")
                 + f" H3={r['H3']}\n{r['H_seq']}\n")

with zipfile.ZipFile(EXPORT / "passing_designs.zip", "w",
                     zipfile.ZIP_DEFLATED) as z:
    for _, r in passing.iterrows():
        z.write(r["path"], arcname=Path(r["path"]).name)

print(f"\nwritten to {EXPORT}:")
for f in sorted(EXPORT.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
print("\nThese are screening candidates, not predicted binders — the authors "
      "are explicit that RF2 is at best a weak filter. Feed them to a display "
      "or expression campaign.")''')

# =============================================================================
# Emit
# =============================================================================

def build():
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "code":
            try:
                compile(src, "<cell>", "exec")
            except SyntaxError as e:
                print(f"SYNTAX ERROR in code cell at line {e.lineno}: {e.msg}",
                      file=sys.stderr)
                print("   " + (e.text or "").rstrip(), file=sys.stderr)
                raise SystemExit(1)
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": source})
        else:
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": source})

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4",
                      "toc_visible": True, "name": "RFantibody_Colab.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "RFantibody_Colab.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    with open(out) as f:
        rt = json.load(f)
    n_code = sum(1 for c in rt["cells"] if c["cell_type"] == "code")
    print(f"wrote {out}")
    print(f"  {len(rt['cells'])} cells ({n_code} code, "
          f"{len(rt['cells']) - n_code} markdown), all code cells compile")


if __name__ == "__main__":
    build()
