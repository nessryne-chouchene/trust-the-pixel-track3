"""Mesures de faisabilite locale (v0.1) : latence, memoire, surcout de securite.

    python -m src.pipeline.benchmark       ->  outputs/benchmark.json

Pour N images de test signees : temps de la couche 1 (securite), des traces
forensiques, du modele, et memoire du processus (pic, via psutil si installe).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.forensic_unet import count_parameters
from src.pipeline.analyze import DEFAULT_MODEL, detect, load_model, load_rgb_uint8
from src.pipeline.sender import sign_image
from src.security import security_assessment as sa
from src.security.adapter import DEFAULT_CONFIG_DIR


def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return None


def run(npz: str = "data/processed/eurosat_tampered.npz", model_path: str = DEFAULT_MODEL,
        n: int = 200, out: str = "outputs/benchmark.json", config_dir=DEFAULT_CONFIG_DIR) -> dict:
    cfg = Path(config_dir)
    rss0 = _rss_mb()
    model, threshold = load_model(model_path)
    rss_model = _rss_mb()
    trust, policy = sa.load_trust_store(cfg / "trusted_keys.json"), sa.load_policy(cfg / "policy.json")
    ledger, now = sa.load_ledger(cfg / "ledger.json"), sa.load_reference_time(cfg / "reference_time.txt")
    with np.load(npz) as z:
        imgs = z["images"][z["split"] == "test"][:n]
    peak = rss_model or 0
    timings = {"cold": {k: [] for k in ("sec", "sig", "mod", "tot")},
               "warm": {k: [] for k in ("sec", "sig", "mod", "tot")}}
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, img in enumerate(imgs):        # preparation (non chronometree) : envoi signe
            p = Path(tmp) / f"b_{i:04d}.png"
            Image.fromarray(img).save(p)
            m, s = sign_image(p, f"b_{i:04d}", observed=now - timedelta(hours=2), signed_at=now - timedelta(hours=2))
            paths.append((p, m, s))
        detect(model, load_rgb_uint8(paths[0][0]))   # echauffement du modele
        # Passe "cold" : premiere ouverture de fichiers tout juste recus. Sous Windows,
        # l'antivirus analyse souvent un fichier neuf a sa premiere ouverture : ce cout
        # d'E/S du systeme tombe sur la premiere etape qui lit le fichier (la securite).
        # Passe "warm" : memes fichiers, deja en cache : cout de calcul seul.
        for mode in ("cold", "warm"):
            t_ = timings[mode]
            for p, m, s in paths:
                t0 = time.perf_counter()
                r = sa.assess(p, m, s, trust, policy, ledger, now)
                t1 = time.perf_counter()
                _, _, t = detect(model, load_rgb_uint8(p))
                t2 = time.perf_counter()
                assert r["security_status"] == "PASSED"
                t_["sec"].append(1000 * (t1 - t0)); t_["sig"].append(t["signals_ms"])
                t_["mod"].append(t["model_ms"]); t_["tot"].append(1000 * (t2 - t0))
                rss = _rss_mb()
                if rss:
                    peak = max(peak, rss)

    def stats(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}

    def block(t_):
        return {"security_ms": stats(t_["sec"]), "signals_ms": stats(t_["sig"]), "model_ms": stats(t_["mod"]),
                "total_ms": stats(t_["tot"]),
                "security_overhead_percent": float(100 * np.mean(t_["sec"]) / np.mean(t_["tot"])),
                "throughput_images_per_s": float(1000 / np.mean(t_["tot"]))}

    res = {
        "n_images": int(len(imgs)), "image_size": list(imgs.shape[1:3]),
        "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
        "python": platform.python_version(), "torch": torch.__version__,
        "warm": block(timings["warm"]), "cold_first_open": block(timings["cold"]),
        "model_parameters": count_parameters(model), "model_size_mb": count_parameters(model) * 4 / 1e6,
        "memory_mb": {"process_start": rss0, "after_model_load": rss_model, "peak": peak or None,
                      "note": None if rss0 else "pip install psutil pour mesurer la memoire"},
        "hardware_cost": "0 (laptop CPU, no GPU, software trust anchor)",
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    r = run(n=a.n, model_path=a.model)
    print(f"{r['n_images']} images {r['image_size']} - CPU, {r['threads']} threads")
    for mode, label in (("warm", "fichiers en cache (calcul seul)"), ("cold_first_open", "premiere ouverture (E/S systeme incluses)")):
        b = r[mode]
        print(f"\n{label} :")
        for k in ("security_ms", "signals_ms", "model_ms", "total_ms"):
            print(f"  {k:12s} moyenne {b[k]['mean']:7.2f} ms   p95 {b[k]['p95']:7.2f} ms")
        print(f"  surcout de securite : {b['security_overhead_percent']:.1f} %   debit : {b['throughput_images_per_s']:.0f} images/s")
    print(f"\nmodele : {r['model_size_mb']:.1f} Mo   memoire : {r['memory_mb']}")
