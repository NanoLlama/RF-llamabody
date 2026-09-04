# What was verified against upstream RFantibody

Everything below was checked against
[RosettaCommons/RFantibody](https://github.com/RosettaCommons/RFantibody) at
commit **`8fe3114`** (2026-03-04), not assumed. Where a claim is from reading
code rather than running it, that is stated.

Re-check the two bugs at any time:

```bash
git clone --depth 1 https://github.com/RosettaCommons/RFantibody.git
python3 tools/verify_resume_bugs.py ./RFantibody
python3 tools/test_target_prep.py   ./RFantibody
python3 tools/test_results_parse.py ./RFantibody
```

---

## 1. The interface is not what the handoff spec assumed

The build spec was written against an older layout. Current reality:

| Spec said | Actually |
|---|---|
| Hydra scripts `rfdiffusion_inference.py` etc. | Console entrypoints `rfdiffusion`, `proteinmpnn`, `rf2` (`pyproject.toml` `[project.scripts]`). The Hydra scripts still exist and the CLIs shell out to them. |
| Per-design PDB directories | The pipeline is built around **Quiver files** (`.qv`) — one flat file holding many PDBs plus a score line each. |
| Hotspots like `[T305,T307]`, target renamed to `T` | Hotspots are `A305,A307` — **no brackets**, and the **chain letter is whatever is in your target PDB**. The shipped example uses `B146,B170,B177` against a chain-`B` target. |
| `download_weights.sh` also fetches the DGL wheel | It does not. DGL comes from a pinned URL in `pyproject.toml` (`data.dgl.ai/wheels/torch-2.3/cu118/...`). The script fetches 4 checkpoints, one of which (~TCR) is irrelevant to antibody design. |
| `scripts/util/chothia2HLT.py` | Correct filename, but the **README's invocation is wrong**: it documents `python scripts/util/chothia_to_HLT.py -inpdb x -outpdb y`. The real interface is `chothia2HLT.py <input.pdb> -H h -L l -T t -o out.pdb`. |

`scripts/examples/nanobody_full_pipeline.sh` is the ground truth for argument
names and ordering, and this notebook mirrors it.

Verified pins (`pyproject.toml`, `uv.lock`): Python `==3.10.*`, `torch==2.3.*`
from the cu118 index, `dgl 2.4.0+cu118` as a direct wheel URL, numpy `<2.0`.
The spec's core architectural call — never install into the Colab kernel, drive
everything through `uv run` — is correct and is what the notebook does.

---

## 2. Two resume bugs, both of which abort the run

This matters more than it sounds. `Quiver.add_pdb` does not overwrite or skip a
duplicate tag — it calls `sys.exit`:

```python
# src/rfantibody/util/quiver.py
if tag in self.tags:
    sys.exit(f'Tag {tag} already exists in this file.')
```

So a broken "already done" check does not merely redo work. The stage re-runs
the first completed design, tries to write it, and **kills the run**. On Colab,
where sessions die routinely, this turns a resume into a hard stop.

### Bug 1 — RFdiffusion, `--output-quiver`

`scripts/rfdiffusion_inference.py` stores the tag with slashes replaced:

```python
outtag = out_prefix.replace('/', '_')      # 'samples/design_0' -> 'samples_design_0'
quiver.add_pdb(pdblines, outtag, scoreline)
```

but the skip check compares the *unreplaced* name:

```python
if out_prefix in tags:                     # 'samples/design_0' in ['samples_design_0']
    log.info('Skipping this design because tag ... already exists.')
```

The default `inference.output_prefix` is `samples/design` (`base.yaml`), and
the CLI only overrides it when writing PDBs, not Quiver — so on the Quiver path
the comparison can never match.

**Workaround (Cell 3):** pass `-e inference.output_prefix=nbdes`. With no slash
in the prefix, `outtag == out_prefix` and the check works. Verified.

Note also: the loop is `range(design_startnum, design_startnum + num_designs)`,
so `--num-designs` is "how many indices to walk", and already-present tags are
skipped within that window. Keeping `--num-designs` at the campaign total across
re-runs is correct; do not reduce it to "the remainder".

### Bug 2 — RF2, `--output-quiver`

`write_output` names outputs `f'{tag}_{suffix}'` where suffix is `best`, but the
done-list for Quiver output returns those written tags verbatim:

```python
# src/rfantibody/rf2/modules/util.py:get_done_list
elif conf.output.quiver is not None:
    qv = Quiver(f'{conf.output.quiver}', mode='r')
    return qv.tags                       # ['X_best', ...]
```

while `scripts/rf2_predict.py` tests the *input* tag against it:

```python
if tag in done_list and conf.inference.cautious:   # 'X' in ['X_best'] -> False
```

The `pdb_dir` branch of the same function strips the suffix correctly
(`strip(i)[:-5]`), so it does not have the bug.

**Workaround (Cell 4):** RF2 writes to `--output-dir`, not `--output-quiver`.
Resume then works, and the per-design `SCORE` lines are trivial to parse.
Verified.

### Not a bug: ProteinMPNN

`src/rfantibody/proteinmpnn/struct_manager.py` handles this correctly — it
strips the `_dldesign_N` suffix off existing output tags to recover input tags.
No workaround needed.

---

## 3. Target preparation is constrained by the parser, not by convention

From `src/rfantibody/rfdiffusion/parsers.py:HLT_pdb_parser`:

```python
pdb_idx = [(l[21:22].strip(), l[22:26].strip()) for l in lines
           if l[:4] == "ATOM" and l[12:16].strip() == "CA"]
...
idx = pdb_idx.index((chain, resNo))
```

Consequences, all handled in Cell 2 and covered by `tools/test_target_prep.py`:

* **Only `ATOM` records are read.** HETATM, waters and ligands are ignored
  outright. But a non-standard residue written as `ATOM` reaches
  `aa2long[aa2num[aa]]` and raises `KeyError`, so those must be dropped.
* **Insertion codes are not read** (`l[22:26]` stops before column 27). Residues
  `100` and `100A` collapse to the same key, and `pdb_idx.index()` returns the
  first, so the second residue's atoms are written onto the first. Structures
  with insertion codes must be renumbered.
* **Alternate locations are not filtered.** Altloc B atoms overwrite altloc A
  through the same mechanism.
* **Duplicate `(chain, resnum)` keys corrupt coordinates.** This is the concrete
  reason the notebook does *not* follow the spec's "rename the target chain to
  `T`": renaming two target chains to `T` when their numbering overlaps produces
  exactly this collision. Renaming is also unnecessary — see below.

### Hotspots reference the original chain, not `T`

`AbPose.parse_hotspots` matches against `self.T.pdb_idx`, which
`target_from_HLT` fills from the target file with **original chain letters
preserved**. The relabelling to `T` happens on output, after matching. This is
why the shipped `flu_HA` example uses `B146,B170,B177`.

`tools/test_target_prep.py` confirms end to end that `B146/B170/B177` resolve
against the cleaned `flu_HA.pdb` and `T305` against `rsv_site3.pdb`.

**The silent-failure mode this creates is the single biggest footgun in the
pipeline.** `parse_hotspots` builds an all-`False` mask when nothing matches and
returns it without warning:

```python
for idx, res in enumerate(self.T.pdb_idx):
    if (res[0], idx2int(res[1])) in hotspots:
        hotspot_idx[idx + binderlen] = True
return hotspot_idx        # silently all-False if nothing matched
```

Wrong chain letter, or numbering shifted by cropping, and you get no error — just
undocked designs after hours of GPU. Cell 2 refuses to continue unless every
hotspot resolves against the prepared file.

---

## 4. Scores: what the metrics are actually called

The filter names in the spec do not exist upstream. From
`src/rfantibody/rf2/modules/model_runner.py`:

| Spec name | Real key | Source |
|---|---|---|
| `pAE_interaction` | **`interaction_pae`** | `pae[0, ~pose.same_chain].mean()` — mean pAE across the inter-chain block |
| Backbone RMSD vs design | **`target_aligned_antibody_rmsd`** | `get_rmsds`, after superimposing on the target |

Also written: `pae`, `pred_lddt`, `target_aligned_cdr_rmsd`,
`framework_aligned_antibody_rmsd`, `framework_aligned_cdr_rmsd`, and
`framework_aligned_{H1,H2,H3}_rmsd`.

RMSDs are **skipped entirely** if the target or framework lengths differ between
design and prediction, so the RMSD columns can be absent. Cell 5 handles that
and falls back to filtering on pAE alone.

Output format depends on the sink:

* `--output-quiver` → `QV_SCORE <tag> key=val|key=val` (readable with `qvscorefile`)
* `--output-dir` → `SCORE <key>: <value>` lines inside each `<tag>_best.pdb`

The notebook uses the latter. `tools/test_results_parse.py` builds a file in
exactly that format and checks every metric, plus CDR extraction from the
`REMARK PDBinfo-LABEL` lines (which are **1-indexed absolute** positions over
the whole pose, not per-chain).

The recommended filter is the README's: `interaction_pae < 10` and RMSD < 2 Å.
The authors are explicit that this is a weak filter and the main limitation of
the pipeline.

---

## 5. What could not be verified here

The build sandbox blocks egress to `data.dgl.ai`, `download.pytorch.org` and
`files.ipd.uw.edu` (403 at the proxy), and has no GPU. So:

* **`uv sync` was not executed end to end.** `uv` resolved the project to
  CPython 3.10.20 and got as far as fetching the DGL wheel before the proxy
  refused it. The lockfile is coherent and pins a hashed DGL wheel, so the
  install is expected to work on Colab, which has open egress — but Cell 1's
  smoke test is what actually proves it, and it is written to fail loudly.
* **No pipeline stage was run on a GPU.** Runtime figures (2-4 min/design on a
  T4) come from the spec and upstream guidance, not measurement here.
* The RF2 output parser was validated against a synthetic file built to the
  format the upstream writer produces, not against real RF2 output.
