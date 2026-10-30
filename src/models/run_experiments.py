"""Experiences de nuit (v0.1) : ablation des traces + variabilite entre graines.

    python -m src.models.run_experiments            ->  outputs/experiments/summary.md

1. Ablation : RGB seul, RGB + SRM, RGB + ELA, RGB + SRM + ELA (graine 0) :
   que rapporte chaque trace forensique ?
2. Graines : le modele complet avec les graines 0, 1, 2 (meme decoupage des
   donnees, initialisation et ordre d'entrainement differents) : moyenne et
   ecart-type, pour savoir si un ecart entre modeles est reel.

Une experience deja terminee (son JSON existe) est sautee : la commande peut
etre relancee apres une interruption.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.models.train_forensic import CHANNEL_SETS, run

ATTACKS = ("splice", "color_shift", "copy_move", "blur")


def plan(seeds=(0, 1, 2)):
    runs = [(ch, 0) for ch in CHANNEL_SETS]
    runs += [("rgb+srm+ela", s) for s in seeds if s != 0]
    return runs


def _row(res: dict) -> dict:
    o = res["overall"]
    row = {"auc": o["auc"], "false_alarm_rate": o["false_alarm_rate"], "iou": o["iou"],
           "recall_at_far_5pct": o["recall_at_far_5pct"]}
    for a in ATTACKS:
        row[f"auc_{a}"] = res["per_attack"].get(a, {}).get("auc", float("nan"))
    return row


def summarize(results: dict) -> str:
    cols = ["auc", "false_alarm_rate", "recall_at_far_5pct", "iou"] + [f"auc_{a}" for a in ATTACKS]
    head = ["AUC", "False alarms", "Recall @5 % FA", "Mask IoU",
            "AUC splice", "AUC colour", "AUC copy-move", "AUC blur (held out)"]
    fmt = lambda k, v: "n/a" if v != v else (f"{100 * v:.1f} %" if k == "false_alarm_rate" else f"{v:.3f}")
    lines = ["## Ablation (seed 0)", "", "| Input channels | " + " | ".join(head) + " |",
             "| --- |" + " --- |" * len(head)]
    for ch in CHANNEL_SETS:
        r = results.get(f"{ch}_seed0")
        if r:
            lines.append(f"| {ch} | " + " | ".join(fmt(k, r[k]) for k in cols) + " |")
    seeds = [v for k, v in results.items() if k.startswith("rgb+srm+ela_seed")]
    if len(seeds) >= 2:
        lines += ["", f"## Full model over {len(seeds)} seeds (mean ± standard deviation)", "",
                  "| Metric | Mean | Std | Min | Max |", "| --- | --- | --- | --- | --- |"]
        for k, h in zip(cols, head):
            v = np.array([s[k] for s in seeds], dtype=float)
            if np.isnan(v).any():
                continue
            if k == "false_alarm_rate":
                lines.append(f"| {h} | {100 * v.mean():.1f} % | {100 * v.std(ddof=1):.1f} | "
                             f"{100 * v.min():.1f} % | {100 * v.max():.1f} % |")
            else:
                lines.append(f"| {h} | {v.mean():.3f} | {v.std(ddof=1):.3f} | {v.min():.3f} | {v.max():.3f} |")
    lines += ["", "Synthetic manipulations of EuroSAT tiles; same data split for every run."]
    return "\n".join(lines) + "\n"


def main(npz: str = "data/processed/eurosat_tampered.npz", out: str = "outputs/experiments",
         epochs: int = 30, seeds=(0, 1, 2)) -> dict:
    root = Path(out)
    results = {}
    todo = plan(seeds)
    for k, (ch, seed) in enumerate(todo, 1):
        name = f"{ch}_seed{seed}"
        run_dir = root / name
        done = run_dir / "forensic_unet_run.json"
        if done.exists():
            print(f"[{k}/{len(todo)}] {name} : deja fait, lu depuis {done}")
            results[name] = _row(json.loads(done.read_text(encoding="utf-8")))
            continue
        print(f"[{k}/{len(todo)}] {name} : entrainement ({time.strftime('%H:%M')})", flush=True)
        res = run(npz=npz, epochs=epochs, seed=seed, channels=ch, out_dir=str(run_dir), verbose=False)
        results[name] = _row(res)
        print(f"    AUC={res['overall']['auc']:.3f}  fausses alarmes={res['overall']['false_alarm_rate']:.3f}"
              f"  IoU={res['overall']['iou']:.3f}  ({res['train_seconds'] / 60:.0f} min)", flush=True)
        md = summarize(results)
        (root / "summary.md").write_text(md, encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = summarize(results)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.md").write_text(md, encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + md)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    main(epochs=a.epochs)
