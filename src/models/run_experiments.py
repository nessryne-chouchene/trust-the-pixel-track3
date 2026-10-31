"""Experiences (v0.2) : ablation des traces, variabilite entre graines, entrainement robuste.

    python -m src.models.run_experiments                                   # plan par defaut
    python -m src.models.run_experiments --channels rgb --seeds 0 1 2      # RGB seul, 3 graines
    python -m src.models.run_experiments --channels rgb --seeds 0 --robust # RGB seul, robuste

Plan par defaut : ablation RGB / RGB + SRM / RGB + ELA / RGB + SRM + ELA (graine 0),
puis le modele complet avec les graines 1 et 2.

Chaque experience va dans outputs/experiments/<canaux>_seed<graine>[_robust]/.
Une experience deja terminee est sautee. Le resume (summary.md) reprend TOUTES
les experiences du dossier, avec la meilleure AUC de VALIDATION : c'est elle
qui sert a choisir un modele ; le test ne sert qu'a rapporter.
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
    best = res["history"][res["best_epoch"] - 1] if res.get("best_epoch") else {}
    row = {"val_auc": best.get("val_auc", float("nan")), "val_iou": best.get("val_iou", float("nan")),
           "auc": o["auc"], "false_alarm_rate": o["false_alarm_rate"], "iou": o["iou"],
           "recall_at_far_5pct": o["recall_at_far_5pct"],
           "train_minutes": res.get("train_seconds", float("nan")) / 60}
    for a in ATTACKS:
        row[f"auc_{a}"] = res["per_attack"].get(a, {}).get("auc", float("nan"))
    return row


def collect(root: Path) -> dict:
    """Toutes les experiences terminees du dossier."""
    results = {}
    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        for fname in ("forensic_unet_run.json", "forensic_unet_robust_run.json"):
            f = d / fname
            if f.exists():
                results[d.name] = _row(json.loads(f.read_text(encoding="utf-8")))
    return results


def _fmt(k, v):
    if v != v:
        return "n/a"
    return f"{100 * v:.1f} %" if k == "false_alarm_rate" else f"{v:.3f}"


def summarize(results: dict) -> str:
    cols = ["val_auc", "auc", "false_alarm_rate", "recall_at_far_5pct", "iou"] + [f"auc_{a}" for a in ATTACKS]
    head = ["Val AUC", "Test AUC", "False alarms", "Recall @5 % FA", "Mask IoU",
            "AUC splice", "AUC colour", "AUC copy-move", "AUC blur (held out)"]
    lines = ["## Ablation (seed 0, standard training)", "",
             "| Input channels | " + " | ".join(head) + " |", "| --- |" + " --- |" * len(head)]
    for ch in CHANNEL_SETS:
        r = results.get(f"{ch}_seed0")
        if r:
            lines.append(f"| {ch} | " + " | ".join(_fmt(k, r[k]) for k in cols) + " |")

    groups = {}
    for name, r in results.items():
        cfg, _, rest = name.partition("_seed")
        robust = rest.endswith("_robust")
        groups.setdefault((cfg, robust), []).append(r)
    multi = {g: rs for g, rs in groups.items() if len(rs) >= 2}
    if multi:
        keys = ["val_auc", "auc", "false_alarm_rate", "iou", "auc_blur"]
        names = ["Val AUC", "Test AUC", "False alarms", "Mask IoU", "AUC blur (held out)"]
        lines += ["", "## Across seeds (mean ± standard deviation)", "",
                  "| Configuration | Seeds | " + " | ".join(names) + " |", "| --- | --- |" + " --- |" * len(names)]
        for (cfg, robust), rs in sorted(multi.items()):
            cells = []
            for k in keys:
                v = np.array([r[k] for r in rs], dtype=float)
                if np.isnan(v).any():
                    cells.append("n/a")
                elif k == "false_alarm_rate":
                    cells.append(f"{100 * v.mean():.1f} ± {100 * v.std(ddof=1):.1f} %")
                else:
                    cells.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}")
            label = cfg + (" (robust)" if robust else "")
            lines.append(f"| {label} | {len(rs)} | " + " | ".join(cells) + " |")

    robust_rows = {n: r for n, r in results.items() if n.endswith("_robust")}
    if robust_rows:
        lines += ["", "## Robust training (each run)", "",
                  "| Run | " + " | ".join(head) + " |", "| --- |" + " --- |" * len(head)]
        for n, r in sorted(robust_rows.items()):
            lines.append(f"| {n} | " + " | ".join(_fmt(k, r[k]) for k in cols) + " |")
    lines += ["", "Synthetic manipulations of EuroSAT tiles; same data split for every run. "
                  "Choose models on validation AUC; test columns are for reporting only."]
    return "\n".join(lines) + "\n"


def main(npz: str = "data/processed/eurosat_tampered.npz", out: str = "outputs/experiments",
         epochs: int = 30, seeds=(0, 1, 2), channels=None, robust: bool = False) -> dict:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    todo = plan(seeds) if channels is None else [(ch, s) for ch in channels for s in seeds]
    suffix = "_robust" if robust else ""
    fname = "forensic_unet_robust_run.json" if robust else "forensic_unet_run.json"
    for k, (ch, seed) in enumerate(todo, 1):
        name = f"{ch}_seed{seed}{suffix}"
        run_dir = root / name
        if (run_dir / fname).exists():
            print(f"[{k}/{len(todo)}] {name} : deja fait")
            continue
        print(f"[{k}/{len(todo)}] {name} : entrainement ({time.strftime('%H:%M')})", flush=True)
        res = run(npz=npz, epochs=epochs, seed=seed, channels=ch, robust=robust,
                  out_dir=str(run_dir), verbose=False)
        print(f"    AUC={res['overall']['auc']:.3f}  fausses alarmes={res['overall']['false_alarm_rate']:.3f}"
              f"  IoU={res['overall']['iou']:.3f}  ({res['train_seconds'] / 60:.0f} min)", flush=True)
        (root / "summary.md").write_text(summarize(collect(root)), encoding="utf-8")
    results = collect(root)
    md = summarize(results)
    (root / "summary.md").write_text(md, encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + md)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--channels", nargs="+", choices=sorted(CHANNEL_SETS),
                    help="jeux de canaux (par defaut : le plan d'ablation)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--robust", action="store_true", help="entrainement robuste (blanchiment aleatoire)")
    a = ap.parse_args()
    main(epochs=a.epochs, seeds=tuple(a.seeds), channels=a.channels, robust=a.robust)
