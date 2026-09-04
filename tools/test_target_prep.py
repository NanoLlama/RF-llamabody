#!/usr/bin/env python3
"""Exercise Cell 2's target-preparation functions outside Colab.

The functions are extracted from the notebook itself (everything above the
"run it" marker), so this tests the code that actually ships, not a copy.

Usage:  python3 tools/test_target_prep.py [path/to/RFantibody]
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "RFantibody_Colab.ipynb"
MARKER = "# ----------------------------------------------------------------- run it"


def load_cell_functions():
    nb = json.loads(NB.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "def clean_target" in src:
            head = src.split(MARKER)[0]
            # Drop the @param form header, which references Colab-only things.
            head = head[head.index("import glob"):]
            ns = {"__name__": "cell2"}
            exec(compile(head, "<cell2>", "exec"), ns)
            return ns
    raise SystemExit("could not find the target-prep cell")


FAIL = 0


def check(label, cond, detail=""):
    global FAIL
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL += 1


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "RFantibody"
    examples = repo / "scripts/examples/example_inputs"
    if not examples.is_dir():
        raise SystemExit(
            f"RFantibody example inputs not found at {examples}.\n"
            f"Pass the repo path: python3 tools/test_target_prep.py /path/to/RFantibody")

    # The cell resolves the repo and a scratch dir from the environment.
    os.environ["RFANTIBODY_REPO"] = str(repo)
    ns = load_cell_functions()
    clean_target = ns["clean_target"]
    residue_index = ns["residue_index"]
    check_chain_breaks = ns["check_chain_breaks"]
    tmp = Path(tempfile.mkdtemp())

    print("\n1. flu_HA.pdb — the upstream example target (chain B)")
    out = tmp / "flu.pdb"
    clean_target(examples / "flu_HA.pdb", out, "B", "")
    idx = residue_index(out)
    check("chain B preserved, not renamed to T", {c for c, _ in idx} == {"B"},
          f"chains={sorted({c for c, _ in idx})}")
    for hs in ("B146", "B170", "B177"):          # upstream example hotspots
        check(f"upstream hotspot {hs} resolves", (hs[0], int(hs[1:])) in idx)

    print("\n2. rsv_site3.pdb — the other shipped target (chain T)")
    out2 = tmp / "rsv.pdb"
    clean_target(examples / "rsv_site3.pdb", out2, "T", "")
    idx2 = residue_index(out2)
    check("chain T target parses", len(idx2) > 0, f"{len(idx2)} residues")
    check("README hotspot T305 resolves", ("T", 305) in idx2)

    print("\n3. Chain selection and cropping")
    out3 = tmp / "fv_H.pdb"
    clean_target(examples / "hu-4D5-8_Fv.pdb", out3, "H", "")
    idx3 = residue_index(out3)
    check("KEEP_CHAINS drops the L chain", {c for c, _ in idx3} == {"H"},
          f"chains={sorted({c for c, _ in idx3})}")

    out4 = tmp / "flu_crop.pdb"
    clean_target(examples / "flu_HA.pdb", out4, "B", "B:140-180")
    idx4 = residue_index(out4)
    nums = [r for _, r in idx4]
    check("CROP restricts to the requested range",
          nums and min(nums) >= 140 and max(nums) <= 180,
          f"{min(nums)}-{max(nums)}, {len(nums)} residues")

    print("\n4. Insertion codes must be renumbered away")
    src = (examples / "flu_HA.pdb").read_text().splitlines()
    mangled = []
    for line in src:
        if line.startswith("ATOM") and int(line[22:26]) == 150:
            line = line[:26] + "A" + line[27:]      # 150 -> 150A
        mangled.append(line)
    mpath = tmp / "icode.pdb"
    mpath.write_text("\n".join(mangled) + "\n")
    out5 = tmp / "icode_clean.pdb"
    warns = clean_target(mpath, out5, "B", "")
    idx5 = residue_index(out5)
    icodes = {ln[26] for ln in out5.read_text().splitlines() if ln.startswith("ATOM")}
    check("no insertion codes survive", icodes == {" "}, f"col27={icodes}")
    check("renumbering is announced to the user",
          any("Renumber" in w for w in warns), f"warnings={len(warns)}")
    check("no duplicate (chain,resnum) keys", len(idx5) == len(set(idx5)))
    check("residue count preserved through renumbering",
          len(idx5) == len(idx), f"{len(idx5)} vs {len(idx)}")

    print("\n5. Alternate locations must not double up")
    alt = []
    for line in src:
        alt.append(line)
        if line.startswith("ATOM") and int(line[22:26]) == 160:
            alt.append(line[:16] + "B" + line[17:])   # altloc B duplicate
    apath = tmp / "altloc.pdb"
    apath.write_text("\n".join(alt) + "\n")
    out6 = tmp / "altloc_clean.pdb"
    clean_target(apath, out6, "B", "")
    idx6 = residue_index(out6)
    check("altloc B atoms dropped", len(idx6) == len(idx), f"{len(idx6)} vs {len(idx)}")
    altcol = {ln[16] for ln in out6.read_text().splitlines() if ln.startswith("ATOM")}
    check("altloc column blanked", altcol == {" "}, f"col17={altcol}")

    print("\n6. HETATM / waters / non-standard residues")
    het = list(src) + [
        "HETATM 9999  O   HOH B 999      10.000  10.000  10.000  1.00 20.00           O",
        "ATOM   9998  CA  MSE B 998      11.000  11.000  11.000  1.00 20.00           C",
    ]
    hpath = tmp / "het.pdb"
    hpath.write_text("\n".join(het) + "\n")
    out7 = tmp / "het_clean.pdb"
    clean_target(hpath, out7, "B", "")
    idx7 = residue_index(out7)
    check("HETATM water and non-standard MSE both excluded",
          len(idx7) == len(idx), f"{len(idx7)} vs {len(idx)}")

    print("\n7. Output is byte-level valid PDB the upstream parser can key on")
    for ln in out.read_text().splitlines():
        if ln.startswith("ATOM"):
            assert ln[21] != " ", "chain id column empty"
            int(ln[22:26])            # resseq must parse
            float(ln[30:38]); float(ln[38:46]); float(ln[46:54])
    check("all ATOM lines parse in fixed PDB columns", True)

    print("\n8. Chain-break detection")
    brk = check_chain_breaks(out4)
    check("cropped fragment reports breaks sanely", isinstance(brk, list),
          f"{len(brk)} break(s)")

    print("\n9. Empty-selection guard")
    try:
        clean_target(examples / "flu_HA.pdb", tmp / "none.pdb", "Z", "")
        check("bad chain selection raises", False, "no exception")
    except SystemExit as e:
        check("bad chain selection raises a clear error", "No residues" in str(e))

    print(f"\n{'=' * 60}")
    print("ALL CHECKS PASSED" if not FAIL else f"{FAIL} CHECK(S) FAILED")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
