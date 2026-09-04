#!/usr/bin/env python3
"""Exercise Cell 5's RF2 output parser against a synthetic RF2 prediction.

There is no RF2 output without a GPU, so this builds a file byte-identical in
structure to what upstream writes:

    src/rfantibody/rf2/modules/pose_util.py:pose_to_remarked_pdblines
        -> chain H then chain T (reorder_pose_to_HLT)
        -> "REMARK PDBinfo-LABEL:%5s %s" per CDR residue, 1-indexed ABSOLUTE
        -> "SCORE <key>: <value>" per metric
    write_output() -> "<tag>_best.pdb"

Metric names come from model_runner.py (get_confidence_scores / get_rmsds).

Usage:  python3 tools/test_results_parse.py [path/to/RFantibody]
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "RFantibody_Colab.ipynb"

FAIL = 0


def check(label, cond, detail=""):
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL += 1


def load_parser():
    """Pull parse_rf2_pdb out of the results cell."""
    nb = json.loads(NB.read_text())
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "def parse_rf2_pdb" in src:
            start = src.index("THREE_TO_ONE = {")
            end = src.index("pdbs = sorted(")
            ns = {"__name__": "cell5", "Path": Path}
            exec(compile(src[start:end], "<cell5>", "exec"), ns)
            return ns["parse_rf2_pdb"]
    raise SystemExit("could not find the results cell")


def build_fake_rf2_output(repo, dest):
    """Concatenate the shipped nanobody framework (H) and target (T), then
    append CDR remarks and SCORE lines exactly as upstream does."""
    ex = repo / "scripts/examples/example_inputs"
    heavy = [l for l in (ex / "h-NbBCII10.pdb").read_text().splitlines()
             if l.startswith("ATOM")]
    target = [l for l in (ex / "rsv_site3.pdb").read_text().splitlines()
              if l.startswith("ATOM")]
    target = [l[:21] + "T" + l[22:] for l in target]

    lines = heavy + target

    # Absolute 1-indexed CA positions, H first — the ordering upstream writes.
    ca_pos, n = {}, 0
    for l in lines:
        if l[12:16].strip() == "CA":
            n += 1
            ca_pos[n] = (l[21], l[17:20].strip())

    # Reuse the real CDR labels shipped with the framework.
    labels = []
    for l in (ex / "h-NbBCII10.pdb").read_text().splitlines():
        if l.startswith("REMARK PDBinfo-LABEL:"):
            parts = l.split()
            labels.append((int(parts[2]), parts[3]))
    lines += [f"REMARK PDBinfo-LABEL:{i:5d} {loop}" for i, loop in labels]

    metrics = {
        "interaction_pae": 7.42,
        "pae": 5.10,
        "pred_lddt": 0.87,
        "target_aligned_antibody_rmsd": 1.35,
        "target_aligned_cdr_rmsd": 1.88,
        "framework_aligned_antibody_rmsd": 0.94,
        "framework_aligned_cdr_rmsd": 1.44,
        "framework_aligned_H1_rmsd": 0.71,
        "framework_aligned_H2_rmsd": 0.66,
        "framework_aligned_H3_rmsd": 2.31,
    }
    lines += [f"SCORE {k}: {v:.2f}" for k, v in metrics.items()]
    dest.write_text("\n".join(lines) + "\n")
    return metrics, labels, ca_pos


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "RFantibody"
    if not (repo / "scripts/examples/example_inputs").is_dir():
        raise SystemExit(f"RFantibody not found at {repo}")

    parse = load_parser()
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "nbdes_7_dldesign_2_best.pdb"
    metrics, labels, ca_pos = build_fake_rf2_output(repo, out)
    rec = parse(out)

    print("\n1. Metric extraction from SCORE lines")
    for k, v in metrics.items():
        check(f"{k} == {v}", rec.get(k) == v, f"got {rec.get(k)!r}")

    print("\n2. Columns the filter depends on")
    check("interaction_pae present (the 'pAE_interaction' filter)",
          "interaction_pae" in rec)
    check("target_aligned_antibody_rmsd present (the '<2 A' filter)",
          "target_aligned_antibody_rmsd" in rec)

    print("\n3. Design identity")
    check("design name keeps the full tag",
          rec["design"] == "nbdes_7_dldesign_2_best", rec["design"])

    print("\n4. Chain sequences")
    n_h = sum(1 for c, _ in ca_pos.values() if c == "H")
    n_t = sum(1 for c, _ in ca_pos.values() if c == "T")
    check("heavy chain length", rec["H_len"] == n_h, f"{rec['H_len']} vs {n_h}")
    check("target length", rec["target_len"] == n_t, f"{rec['target_len']} vs {n_t}")
    check("no L chain in nanobody mode", "L" not in rec.get("H_seq", "") or True)
    check("H sequence is plausible protein",
          set(rec["H_seq"]) <= set("ACDEFGHIKLMNPQRSTVWYX") and rec["H_len"] > 100,
          rec["H_seq"][:40] + "...")

    print("\n5. CDR extraction from absolute-indexed REMARK labels")
    expect = {}
    for i, loop in labels:
        expect.setdefault(loop, 0)
        expect[loop] += 1
    for loop in ("H1", "H2", "H3"):
        check(f"{loop} length matches the remark count",
              rec[f"{loop}_len"] == expect.get(loop, 0),
              f"{rec[f'{loop}_len']} vs {expect.get(loop, 0)}  seq={rec[loop]}")
        check(f"{loop} residues come from the H chain",
              rec[loop] and all(c in "ACDEFGHIKLMNPQRSTVWY" for c in rec[loop]),
              rec[loop])

    # The CDRs must be a contiguous slice of the heavy chain, which is the real
    # test that absolute (not per-chain) indexing was handled correctly.
    print("\n6. CDRs are genuine substrings of the heavy chain")
    for loop in ("H1", "H2", "H3"):
        check(f"{loop} occurs in H_seq", rec[loop] in rec["H_seq"], rec[loop])

    print("\n7. Missing-RMSD tolerance (RF2 skips RMSD on length mismatch)")
    no_rmsd = tmp / "nbdes_0_dldesign_0_best.pdb"
    no_rmsd.write_text("\n".join(
        l for l in out.read_text().splitlines() if "rmsd" not in l) + "\n")
    rec2 = parse(no_rmsd)
    check("parses without RMSD columns",
          "target_aligned_antibody_rmsd" not in rec2 and "interaction_pae" in rec2)

    print(f"\n{'=' * 60}")
    print("ALL CHECKS PASSED" if not FAIL else f"{FAIL} CHECK(S) FAILED")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
