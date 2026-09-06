#!/usr/bin/env python3
"""
audit_stage1.py — cross-check the stage-1 logs against the stage-1 JSONs.

A class ends up with mean width 0.0 in the JSON for two very different reasons,
and the JSON cannot tell them apart:

  * the pre-screen proved it empty — legitimate, there is nothing to measure;
  * the pre-screen could NOT decide, so the class was kept on purpose, and then
    every one of its direction LPs failed — in which case it is recorded as 0.0
    and stage 2 will skip it, i.e. the polytope is zeroed without proof after all.

The second case defeats the very safety net that keeps undecided classes alive.
Per-class LP failures are counted nowhere ("Failed directions: 0/100" tracks only
the base and P2), so the only way to find them is to compare what the pre-screen
said survived against what the JSON actually contains.

Reports, per tile: retries, kept-without-proof classes, and above all any
MISMATCH between the two counts. A mismatch means a class silently vanished.

On a Jean Zay login node this needs nothing but the standard library.

Usage
-----
    python scripts/audit_stage1.py \
        --logs_dir logs --results_dir results/volumes_v3k_cnn_gen150 \
        --task_file data/gen150/volume_tasks.txt
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

RE_TASK    = re.compile(r"_(\d+)\.err$")
RE_EMPTY   = re.compile(r"Pre-screening: (\d+)/(\d+) P3\(k\) polytopes provably empty")
RE_KEPT    = re.compile(r"Pre-screening: (\d+) polytope\(s\) KEPT")
RE_RETRIED = re.compile(r"Pre-screening: (\d+) LP\(s\) retried")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs_dir",    default="logs")
    ap.add_argument("--results_dir", default="results/volumes_v3k_cnn_gen150")
    ap.add_argument("--task_file",   default="data/gen150/volume_tasks.txt")
    ap.add_argument("--pattern",     default="vol_gen150_cnn_*_*.err")
    a = ap.parse_args()

    tasks = [l.split() for l in Path(a.task_file).read_text().split("\n") if l.strip()]
    logs = sorted(Path(a.logs_dir).glob(a.pattern))
    if not logs:
        raise SystemExit(f"[ABORT] no log matching {a.pattern} under {a.logs_dir}")

    seen, mismatches, kept_tiles, retried_tiles = {}, [], [], []
    no_prescreen = missing_json = 0

    for f in logs:
        m = RE_TASK.search(f.name)
        if not m:
            continue
        tid = int(m.group(1))
        if tid >= len(tasks):
            continue
        bits, aug = int(tasks[tid][0]), int(tasks[tid][1])
        txt = f.read_text(errors="replace")

        # a log can hold several runs (a relaunch appends); the last one is the live result
        em = RE_EMPTY.findall(txt)
        if not em:
            no_prescreen += 1
            continue
        n_empty, n_total = int(em[-1][0]), int(em[-1][1])
        n_kept    = int(RE_KEPT.findall(txt)[-1])    if RE_KEPT.search(txt)    else 0
        n_retried = int(RE_RETRIED.findall(txt)[-1]) if RE_RETRIED.search(txt) else 0

        js = Path(a.results_dir) / f"b{bits:02d}" / f"volumes_sample{aug}.json"
        if not js.exists():
            missing_json += 1
            continue
        W = json.loads(js.read_text())["widths_both"][str(bits)]
        n_positive = sum(1 for w in W if w is not None and w > 0)

        expected = n_total - n_empty          # kept classes are NOT counted as empty
        rec = dict(task=tid, bits=bits, aug=aug, expected=expected,
                   found=n_positive, kept=n_kept, retried=n_retried, log=f.name)
        seen[(bits, aug)] = rec
        if n_kept:
            kept_tiles.append(rec)
        if n_retried:
            retried_tiles.append(rec)
        if n_positive != expected:
            mismatches.append(rec)

    print(f"tuiles auditees          : {len(seen)}")
    print(f"logs sans pre-criblage   : {no_prescreen}   (taches sautees ou interrompues)")
    print(f"JSON absents             : {missing_json}")
    print(f"tuiles avec reprise IPM->simplexe : {len(retried_tiles)}")
    print(f"tuiles avec classe CONSERVEE sans preuve : {len(kept_tiles)}")
    for r in kept_tiles:
        print(f"   b={r['bits']:<3} aug={r['aug']:<4} kept={r['kept']}  "
              f"attendu={r['expected']} trouve={r['found']}  "
              f"{'<-- LA CLASSE A DISPARU' if r['found'] < r['expected'] else 'ok (mesuree)'}")

    print(f"\nMISMATCHES (classe non vide absente du JSON) : {len(mismatches)}")
    for r in mismatches[:40]:
        print(f"   b={r['bits']:<3} aug={r['aug']:<4} attendu={r['expected']} "
              f"trouve={r['found']}  (kept={r['kept']})   {r['log']}")
    if len(mismatches) > 40:
        print(f"   ... et {len(mismatches)-40} de plus")

    if not mismatches:
        print("\n==> Aucun ecart : tout ce que le pre-criblage a laisse passer a bien"
              "\n    ete mesure. L'angle mort est vide, les resultats sont exploitables.")
    else:
        by_b = Counter(r["bits"] for r in mismatches)
        print(f"\n==> {len(mismatches)} tuile(s) concernee(s), par b : {dict(by_b)}"
              "\n    Ces classes valent 0.0 dans le JSON sans que ce soit prouve, et"
              "\n    l'etape 2 les saute. A traiter avant d'exploiter gamma.")
        sys.exit(1)


if __name__ == "__main__":
    main()
