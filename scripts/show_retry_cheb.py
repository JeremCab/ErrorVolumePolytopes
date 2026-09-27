#!/usr/bin/env python3
"""
show_retry_cheb.py — verdict of the two retry arms, and why one may be empty.

Arm A reran the failed Chebyshev LPs with the campaign's own solver (dual
simplex): it asks whether the failures REPRODUCE. Arm B reran them with
`--lp_method auto` (interior point first): it asks whether IPM RECOVERS them.

The script first checks the prerequisites, because an array whose task file is
missing finishes in seconds with every task aborting — which looks exactly like
success.

    python scripts/show_retry_cheb.py
"""
import argparse, json
from pathlib import Path


def scan(d):
    """(tiles, polytopes, converged, failed, per-tile detail)"""
    p = Path(d)
    if not p.is_dir():
        return None
    tiles, npoly, ok, bad, detail = 0, 0, 0, 0, {}
    for f in sorted(p.rglob("chebyshev_sample*.json")):
        j = json.loads(f.read_text())
        tiles += 1
        c = sum(1 for q in j["polytopes"] if q["status"] != "failed")
        b = sum(1 for q in j["polytopes"] if q["status"] == "failed")
        npoly += c + b; ok += c; bad += b
        detail[int(j["sample_idx"])] = (c, c + b)
    return tiles, npoly, ok, bad, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_file", default="data/gen150/retry_cheb_tasks.txt")
    ap.add_argument("--orig_dir", default="results/chebyshev_cnn_gen150")
    ap.add_argument("--arms", nargs="+",
                    default=["results/retry_cheb_armA", "results/retry_cheb_armB"])
    ap.add_argument("--logs", default="logs/retry_cheb_*.out")
    a = ap.parse_args()

    print("\n--- prerequis ---")
    tf = Path(a.task_file)
    if not tf.exists():
        print(f"✗ {a.task_file} ABSENT — c'est la cause la plus probable : chaque")
        print( "  tache sort immediatement. Lancer d'abord :")
        print( "      python scripts/list_failed_cheb.py --bits 4 --limit 20")
        print( "  puis resoumettre les deux sbatch.")
    else:
        lines = [l for l in tf.read_text().split("\n") if l.strip()]
        print(f"✓ {a.task_file} : {len(lines)} taches"
              + (f"  (premiere : {lines[0]})" if lines else "  ⚠ VIDE"))

    logs = sorted(Path().glob(a.logs))
    print(f"{'✓' if logs else '✗'} journaux : {len(logs)} fichiers {a.logs}")
    if logs:
        txt = logs[0].read_text(errors="replace").strip().split("\n")
        print(f"  extrait de {logs[0].name} :")
        for l in txt[:4]:
            print(f"    {l}")

    print("\n--- resultats ---")
    orig = scan(a.orig_dir)
    for arm in a.arms:
        r = scan(arm)
        name = Path(arm).name
        if r is None:
            print(f"✗ {name:<20} repertoire absent — l'array n'a rien ecrit")
            continue
        tiles, npoly, ok, bad, detail = r
        if tiles == 0:
            print(f"✗ {name:<20} repertoire vide")
            continue
        print(f"✓ {name:<20} {tiles:>3} tuiles, {npoly:>4} polytopes, "
              f"{ok:>4} converges ({100*ok/npoly:.0f}%), {bad:>4} echecs")
        if orig:
            gained = sum(1 for i, (c, n) in detail.items()
                         if i in orig[4] and c > orig[4][i][0])
            print(f"{'':>22} dont {gained} tuiles ou l'on converge MIEUX qu'a la campagne")

    if orig:
        print(f"\n  rappel campagne d'origine ({a.orig_dir}) : "
              f"{orig[2]}/{orig[1]} polytopes converges")
    print("\n  arm A = meme solveur : si les echecs se reproduisent a l'identique,"
          "\n          le cas de l'echantillon 683 etait isole et un simple reessai"
          "\n          ne donne rien."
          "\n  arm B = point interieur : c'est lui qui a une base mesuree.")


if __name__ == "__main__":
    main()
