"""Demonstration rejouable de la defense a deux couches (v0.1).

    python -m src.pipeline.demo            ->  outputs/demo/index.html

Quatre scenarios, construits a partir d'une vignette EuroSAT de TEST (jamais
vue a l'entrainement) et de sa copie falsifiee (masque exact connu) :

  A. Image authentique, signee par l'emetteur      -> acceptee, rien detecte
  B. Falsifiee PENDANT le transfert (apres signature) -> rejetee par la couche 1
  C. Falsifiee AVANT la signature (initie, chaine de traitement compromise)
                                                    -> la signature est valide,
                                                       la couche 2 detecte et localise
  D. Rejeu de l'image A deja acceptee               -> rejetee par la couche 1

La vignette est la premiere source du test qui a une copie 'splice' : elle n'est
pas choisie pour son resultat. Les scores globaux sont dans outputs/*.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from src.pipeline.analyze import DEFAULT_MODEL, analyze, load_model, render_html, report_to_dict
from src.pipeline.sender import sign_image
from src.security import security_assessment as sa
from src.security.adapter import DEFAULT_CONFIG_DIR

SCENARIOS = {
    "A_authentic": "A. Authentic image, signed by the trusted sender",
    "B_tampered_in_transit": "B. Edited in transit, after signing (man-in-the-middle)",
    "C_tampered_before_signing": "C. Edited before signing (insider / compromised processing chain)",
    "D_replay": "D. Replay of image A, already accepted",
}
EXPECTED = {"A_authentic": "NOT_DETECTED", "B_tampered_in_transit": "REJECTED",
            "C_tampered_before_signing": "SUSPECTED", "D_replay": "REJECTED"}


def pick_example(data: dict, attack: str = "splice"):
    """Premiere source du test ayant une copie du type demande (repli sur un
    autre type s'il est absent). Retourne aussi le type reellement utilise."""
    te = np.where(data["split"] == "test")[0]
    for kind in dict.fromkeys([attack, "splice", "color_shift", "copy_move", "blur"]):
        for i in te:
            if data["attack"][i] == kind:
                src = data["source"][i]
                auth = [j for j in te if data["source"][j] == src and data["attack"][j] == "none"][0]
                return data["images"][auth], data["images"][i], data["masks"][i], int(src), kind
    raise ValueError("Aucune image falsifiee dans le test.")


def run(npz: str = "data/processed/eurosat_tampered.npz", model_path: str = DEFAULT_MODEL,
        out_dir: str = "outputs/demo", config_dir=DEFAULT_CONFIG_DIR, attack: str = "splice") -> dict:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "received").mkdir(parents=True)
    cfg = Path(config_dir)
    now = sa.load_reference_time(cfg / "reference_time.txt")
    observed, signed_at = now - timedelta(hours=6), now - timedelta(hours=6) + timedelta(minutes=5)
    ledger_path = out / "ledger.json"
    shutil.copy(cfg / "ledger.json", ledger_path)            # registre de demo (l'original n'est pas touche)
    model, threshold = load_model(model_path)
    with np.load(npz) as z:
        data = {k: z[k] for k in z.files}
    authentic, tampered, gt_mask, src, attack = pick_example(data, attack)
    rec = out / "received"

    def save(name, arr):
        p = rec / f"{name}.png"
        Image.fromarray(arr).save(p)
        return p

    def run_one(key, img_path, manifest, gt=None):
        return analyze(img_path, manifest, config_dir=cfg, ledger=sa.load_ledger(ledger_path), now=now,
                      model=model, threshold=threshold, ground_truth_mask=gt)

    results, items = {}, []
    # A. authentique
    pa = save("demo_a", authentic)
    ma, _ = sign_image(pa, "demo_a", observed=observed, signed_at=signed_at)
    ra = run_one("A_authentic", pa, ma)
    sa.record_accepted(ledger_path, {"security_status": ra.validation.status.value,
                                     "image_id": ra.validation.image_id,
                                     "evidence": ra.validation.evidence}, now)
    # B. modifiee apres signature : la signature couvre l'image authentique
    pb = save("demo_b", authentic)
    mb, _ = sign_image(pb, "demo_b", observed=observed, signed_at=signed_at)
    Image.fromarray(tampered).save(pb)
    rb = run_one("B_tampered_in_transit", pb, mb)
    # C. modifiee avant signature : la signature couvre l'image falsifiee
    pc = save("demo_c", tampered)
    mc, _ = sign_image(pc, "demo_c", observed=observed, signed_at=signed_at)
    rc = run_one("C_tampered_before_signing", pc, mc, gt=gt_mask)
    # D. rejeu de A
    rd = run_one("D_replay", pa, ma)

    images = {"A_authentic": authentic, "B_tampered_in_transit": tampered,
              "C_tampered_before_signing": tampered, "D_replay": authentic}
    notes = {"B_tampered_in_transit": "The sender signed the authentic tile; the file was then replaced by the edited one.",
             "C_tampered_before_signing": "The edited tile was signed with a valid trusted key: the signature alone cannot catch this.",
             "D_replay": "Same image_id and same bytes as A, sent again after A was accepted."}
    for key, rep in zip(SCENARIOS, (ra, rb, rc, rd)):
        items.append((SCENARIOS[key], rep, images[key], notes.get(key, "")))
        results[key] = {**report_to_dict(rep), "expected_verdict": EXPECTED[key],
                        "as_expected": rep.verdict == EXPECTED[key]}
    intro = (f"Replayable demo on a EuroSAT test tile (source #{src}, never used for training), "
             f"manipulation '{attack}'. Layer 1 checks provenance; layer 2 checks the pixels. "
             f"Detector threshold {threshold:.2f}, chosen on validation data.")
    (out / "index.html").write_text(render_html(items, "Trust the Pixel: two-layer defence demo", intro),
                                    encoding="utf-8")
    summary = {"source": src, "attack": attack, "threshold": threshold, "scenarios": results,
               "all_as_expected": all(r["as_expected"] for r in results.values())}
    (out / "demo_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--attack", default="splice", choices=["splice", "copy_move", "color_shift", "blur"])
    a = ap.parse_args()
    s = run(model_path=a.model, attack=a.attack)
    for k, r in s["scenarios"].items():
        mark = "OK " if r["as_expected"] else "!! "
        sc = r["classification"]["score"] if r["classification"] else None
        print(f"{mark}{k:28s} {r['verdict']:13s} security={r['validation']['status']:7s} "
              f"codes={r['validation']['reason_codes']} score={'-' if sc is None else f'{sc:.2f}'}")
    print("\nOuvre outputs/demo/index.html dans ton navigateur.")
