#!/usr/bin/env python3
"""Reproduce the two upstream Quiver resume bugs that Cell 4 works around.

Both are exercised against RFantibody's real Quiver class, so this stays honest
if upstream changes. A PASS line means the bug is still present and the
workaround in the notebook is still needed; FIXED means it can be dropped.

Usage:  python3 tools/verify_resume_bugs.py [path/to/RFantibody]
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_quiver(repo):
    path = repo / "src/rfantibody/util/quiver.py"
    if not path.exists():
        raise SystemExit(f"Quiver not found at {path}")
    spec = importlib.util.spec_from_file_location("rfab_quiver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Quiver


def main():
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "RFantibody"
    Quiver = load_quiver(repo)
    tmp = Path(tempfile.mkdtemp())
    still_broken = 0

    # ------------------------------------------------------------------ #1
    print("=" * 72)
    print("BUG 1  RFdiffusion quiver resume, when output_prefix contains '/'")
    print("  scripts/rfdiffusion_inference.py stores   outtag = out_prefix.replace('/','_')")
    print("  but the skip check tests                  `if out_prefix in tags`")
    print("-" * 72)
    fn = tmp / "diffusion.qv"
    output_prefix = "samples/design"      # scripts/config/inference/base.yaml default
    out_prefix = f"{output_prefix}_0"
    outtag = out_prefix.replace("/", "_")

    Quiver(str(fn), mode="w").add_pdb(["ATOM  dummy\n"], outtag, "mindist=1.00")
    tags = Quiver(str(fn), mode="w").get_tags()
    matched = out_prefix in tags
    print(f"  stored tag         : {tags}")
    print(f"  skip check compares: {out_prefix!r}")
    print(f"  match              : {matched}")
    try:
        Quiver(str(fn), mode="w").add_pdb(["ATOM  dummy\n"], outtag, "mindist=1.00")
        crashed = None
    except SystemExit as e:
        crashed = str(e)
    print(f"  re-adding the tag  : {crashed or 'no error'}")
    if not matched and crashed:
        still_broken += 1
        print("  => STILL BROKEN. A resumed run re-generates design 0 and then")
        print("     aborts. Workaround: pass a slash-free inference.output_prefix.")
    else:
        print("  => FIXED upstream; the -e inference.output_prefix override can go.")

    print("\n  control: the same check with a slash-free prefix")
    fn2 = tmp / "diffusion2.qv"
    Quiver(str(fn2), mode="w").add_pdb(["ATOM  dummy\n"], "nbdes_0", None)
    ok = "nbdes_0" in Quiver(str(fn2), mode="w").get_tags()
    print(f"    'nbdes_0' in tags: {ok}   <- the workaround the notebook uses")

    # ------------------------------------------------------------------ #2
    print("\n" + "=" * 72)
    print("BUG 2  RF2 quiver resume: done-list carries a '_best' suffix")
    print("  model_runner.write_output writes  f'{tag}_best'")
    print("  util.get_done_list returns        qv.tags   (i.e. the '_best' names)")
    print("  rf2_predict.py then tests         `if tag in done_list`")
    print("-" * 72)
    fn3 = tmp / "rf2.qv"
    input_tag = "nbdes_0_dldesign_0"
    Quiver(str(fn3), mode="w").add_pdb(["ATOM  dummy\n"], f"{input_tag}_best",
                                       "interaction_pae=7.10")
    done = Quiver(str(fn3), mode="r").tags
    matched = input_tag in done
    print(f"  done_list          : {done}")
    print(f"  input tag          : {input_tag!r}")
    print(f"  match              : {matched}")
    try:
        Quiver(str(fn3), mode="w").add_pdb(["ATOM  dummy\n"], f"{input_tag}_best",
                                           "interaction_pae=7.10")
        crashed = None
    except SystemExit as e:
        crashed = str(e)
    print(f"  re-writing output  : {crashed or 'no error'}")
    if not matched and crashed:
        still_broken += 1
        print("  => STILL BROKEN with --output-quiver. Workaround: --output-dir.")
    else:
        print("  => FIXED upstream; RF2 may write quiver output again.")

    print("\n  control: the pdb_dir path, which strips '_best' correctly")
    strip = lambda x: os.path.splitext(os.path.basename(x))[0]
    done_pdb = [strip(f"out/{input_tag}_best.pdb")[:-5]]
    print(f"    done_list: {done_pdb}   tag matched: {input_tag in done_pdb}"
          "   <- the workaround the notebook uses")

    print("\n" + "=" * 72)
    print(f"{still_broken}/2 bug(s) still present in {repo}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
