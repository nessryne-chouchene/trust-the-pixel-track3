"""Analyse de bout en bout d'une image recue (v0.1).

    python -m src.pipeline.analyze --image X.png --manifest X.manifest.json

Couche 1, provenance (src/security) : l'image vient-elle d'un emetteur de
confiance, et est-elle intacte depuis la signature ? Un echec arrete tout :
un contenu dont l'origine n'est pas prouvee n'est pas analyse comme fiable.
Couche 2, pixels (ForensicUNet) : l'image a-t-elle ete retouchee AVANT d'etre
signee ? La signature ne peut pas le voir ; le detecteur, si.

Sorties : rapport JSON, rapport HTML autonome, image avec le masque superpose.
Un score de modele n'est jamais une preuve.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.forensics.signals import detector_inputs
from src.models.forensic_unet import load_detector
from src.security import security_assessment as sa
from src.security.adapter import DEFAULT_CONFIG_DIR, to_validation_report
from src.utils.contracts import (ClassificationOutput, ForensicReport, LocalizationOutput,
                                 ValidationStatus)

TILE = 64                      # taille des vignettes EuroSAT vues a l'entrainement
MASK_THRESHOLD = 0.5
DEFAULT_MODEL = "outputs/forensic_unet_robust.pt"
LIMITATIONS = [
    "Detector trained on synthetic manipulations of EuroSAT RGB tiles (splice, copy-move, "
    "colour shift); validity outside these attack types is not established.",
    "A model score is evidence for an analyst, never proof of manipulation or authenticity.",
    "Provenance checks prove origin and integrity since signing; they cannot see edits made "
    "before signing, nor protect against a stolen signing key.",
]


def load_model(path=DEFAULT_MODEL):
    return load_detector(path)


def load_rgb_uint8(path) -> np.ndarray:
    """Image 8 bits RGB (H, W, 3). Les GeoTIFF multibandes ou 16 bits passent par
    le pretraitement (choix des bandes RGB, normalisation) puis sont quantifies."""
    try:
        with Image.open(path) as im:
            if im.mode in ("RGB", "RGBA", "L", "P"):
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        pass
    from src.preprocessing.preprocess import load_image, normalize_to_unit, select_bands
    arr, dtype = load_image(path)
    rgb, _ = select_bands(arr)
    x, _ = normalize_to_unit(rgb, dtype)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)


def _tile_starts(n: int) -> list[int]:
    if n <= TILE:
        return [0]
    starts = list(range(0, n - TILE + 1, TILE))
    if starts[-1] != n - TILE:
        starts.append(n - TILE)          # derniere vignette alignee sur le bord
    return starts


@torch.no_grad()
def detect(model, image: np.ndarray) -> tuple[float, np.ndarray, dict]:
    """Analyse par vignettes de 64x64 (comme a l'entrainement).
    Score image = maximum des scores des vignettes ; masque = moyenne des recouvrements."""
    h, w = image.shape[:2]
    if h < TILE or w < TILE:
        tiles, boxes = [image], [(0, 0, h, w)]
    else:
        tiles, boxes = [], []
        for y in _tile_starts(h):
            for x in _tile_starts(w):
                tiles.append(image[y:y + TILE, x:x + TILE])
                boxes.append((y, x, TILE, TILE))
    t0 = time.perf_counter()
    inputs = detector_inputs(np.stack(tiles), getattr(model, "input_channels", None))
    t1 = time.perf_counter()
    cls_logits, mask_logits = model(inputs)
    t2 = time.perf_counter()
    scores = torch.sigmoid(cls_logits).numpy()
    probs = torch.sigmoid(mask_logits)[:, 0].numpy()
    acc, cnt = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for p, (y, x, th, tw) in zip(probs, boxes):
        acc[y:y + th, x:x + tw] += p[:th, :tw]
        cnt[y:y + th, x:x + tw] += 1
    return float(scores.max()), acc / np.maximum(cnt, 1), {
        "signals_ms": 1000 * (t1 - t0), "model_ms": 1000 * (t2 - t1), "tiles": len(tiles)}


def _overlay(image: np.ndarray, mask_prob: np.ndarray) -> np.ndarray:
    flagged = mask_prob >= MASK_THRESHOLD
    out = image.astype(np.float32).copy()
    out[flagged] = 0.45 * out[flagged] + 0.55 * np.array([230, 30, 30], np.float32)
    return out.clip(0, 255).astype(np.uint8)


def _heatmap(mask_prob: np.ndarray) -> np.ndarray:
    p = mask_prob.clip(0, 1)
    return np.stack([255 * p, 40 * (1 - p), 90 * (1 - p)], -1).astype(np.uint8)


def _png_b64(arr: np.ndarray, min_side: int = 256) -> str:
    im = Image.fromarray(arr)
    f = max(1, min_side // max(1, min(im.size)))
    if f > 1:
        im = im.resize((im.width * f, im.height * f), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _verdict(v, score, threshold) -> tuple[str, str]:
    codes = ", ".join(v.reason_codes)
    if v.status is ValidationStatus.FAILED:
        return "REJECTED", (f"Provenance or integrity check failed ({codes}). The content cannot be "
                            "trusted, so it is not passed to the analysts or the models.")
    pixel = (f"pixel evidence of manipulation (model score {score:.2f} >= threshold {threshold:.2f})"
             if score >= threshold else
             f"no pixel evidence of the tested manipulation types (score {score:.2f} < {threshold:.2f})")
    if v.status is ValidationStatus.REVIEW:
        return "REVIEW", f"Signed by a trusted key, but metadata checks failed ({codes}); {pixel}. A human must review."
    if score >= threshold:
        return "SUSPECTED", ("Origin and integrity verified, yet " + pixel +
                             ": the image was most likely edited before it was signed.")
    return "NOT_DETECTED", "Origin and integrity verified; " + pixel + "."


def analyze(image_path, manifest_path, sig_path=None, *, config_dir=DEFAULT_CONFIG_DIR,
            ledger: Optional[dict] = None, now=None, model=None, threshold: Optional[float] = None,
            model_path=DEFAULT_MODEL, out_dir: Optional[str] = None,
            ground_truth_mask: Optional[np.ndarray] = None) -> ForensicReport:
    cfg = Path(config_dir)
    if model is None:
        model, threshold = load_model(model_path)
    t0 = time.perf_counter()
    result = sa.assess(image_path, manifest_path,
                       sig_path if sig_path is not None else sa.default_sig_path(manifest_path),
                       trust_store=sa.load_trust_store(cfg / "trusted_keys.json"),
                       policy=sa.load_policy(cfg / "policy.json"),
                       ledger=ledger if ledger is not None else sa.load_ledger(cfg / "ledger.json"),
                       now=now or sa.load_reference_time(cfg / "reference_time.txt"))
    t_sec = 1000 * (time.perf_counter() - t0)
    v = to_validation_report(result)
    report = ForensicReport(validation=v, limitations=list(LIMITATIONS),
                            timings_ms={"security_ms": t_sec})
    image = None
    if v.status is not ValidationStatus.FAILED:
        image = load_rgb_uint8(image_path)
        score, mask_prob, t = detect(model, image)
        report.classification = ClassificationOutput(score=score, threshold=threshold,
                                                     decision=score >= threshold)
        report.localization = LocalizationOutput(mask_prob=mask_prob,
                                                 mask_binary=(mask_prob >= MASK_THRESHOLD).astype(np.uint8))
        report.timings_ms.update(t)
        if ground_truth_mask is not None:
            from src.utils.metrics import segmentation_metrics
            report.artifacts["iou_vs_ground_truth"] = segmentation_metrics(
                report.localization.mask_binary, ground_truth_mask)["iou"]
    report.timings_ms["total_ms"] = 1000 * (time.perf_counter() - t0)
    score = report.classification.score if report.classification else 0.0
    report.verdict, report.explanation = _verdict(v, score, threshold)
    if out_dir:
        write_outputs(report, image, Path(out_dir), Path(image_path).stem)
    return report


def report_to_dict(r: ForensicReport) -> dict:
    d = {"verdict": r.verdict, "explanation": r.explanation,
         "validation": {**asdict(r.validation), "status": r.validation.status.value},
         "classification": asdict(r.classification) if r.classification else None,
         "localization": None, "warnings": r.warnings, "limitations": r.limitations,
         "timings_ms": r.timings_ms, "artifacts": r.artifacts}
    if r.localization is not None:
        m = r.localization.mask_binary
        ys, xs = np.nonzero(m)
        d["localization"] = {"flagged_fraction": float(m.mean()),
                             "bbox_yxhw": [int(ys.min()), int(xs.min()), int(np.ptp(ys) + 1), int(np.ptp(xs) + 1)]
                             if len(ys) else None, "shape": list(m.shape)}
    return d


def write_outputs(r: ForensicReport, image: Optional[np.ndarray], out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if image is not None and r.localization is not None:
        Image.fromarray(_overlay(image, r.localization.mask_prob)).save(out / f"{stem}_overlay.png")
        r.artifacts["overlay_png"] = str(out / f"{stem}_overlay.png")
    r.artifacts["report_json"] = str(out / f"{stem}_report.json")
    r.artifacts["report_html"] = str(out / f"{stem}_report.html")
    (out / f"{stem}_report.json").write_text(json.dumps(report_to_dict(r), indent=2), encoding="utf-8")
    (out / f"{stem}_report.html").write_text(render_html([(stem, r, image)], "Forensic report"), encoding="utf-8")


_CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;
--ok:#1f7a4d;--warn:#a35c00;--bad:#b3261e;--sus:#7a1fa3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#21211f;--ink:#ecebe6;--muted:#a3a29b;
--line:#363632;--ok:#4cc38a;--warn:#e7a245;--bad:#ff8a80;--sus:#d18cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1060px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:18px;margin:0 0 16px}.row{display:flex;gap:16px;flex-wrap:wrap}
.imgs{display:flex;gap:12px;flex-wrap:wrap}.imgs figure{margin:0}.imgs img{width:200px;max-width:100%;
image-rendering:pixelated;border-radius:8px;border:1px solid var(--line)}figcaption{font-size:12px;color:var(--muted)}
.badge{display:inline-block;font-weight:700;font-size:13px;padding:3px 10px;border-radius:99px;border:1.5px solid}
.REJECTED{color:var(--bad)}.SUSPECTED{color:var(--sus)}.REVIEW{color:var(--warn)}.NOT_DETECTED{color:var(--ok)}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}
code{font-size:12px}.fine{color:var(--muted);font-size:12px}h2{font-size:18px;margin:0 0 8px}
"""


def _card(name: str, r: ForensicReport, image: Optional[np.ndarray], note: str = "") -> str:
    v = r.validation
    imgs = ""
    if image is not None:
        imgs += f'<figure><img alt="received image" src="data:image/png;base64,{_png_b64(image)}"><figcaption>received image</figcaption></figure>'
    if r.localization is not None:
        imgs += (f'<figure><img alt="flagged regions" src="data:image/png;base64,{_png_b64(_overlay(image, r.localization.mask_prob))}">'
                 '<figcaption>flagged regions (red)</figcaption></figure>'
                 f'<figure><img alt="manipulation heatmap" src="data:image/png;base64,{_png_b64(_heatmap(r.localization.mask_prob))}">'
                 '<figcaption>pixel probability map</figcaption></figure>')
    checks = "".join(
        f"<tr><td>{c['step']}</td><td>{c['check']}</td><td>{'pass' if c['passed'] else '<b>FAIL ' + c.get('code', '') + '</b>'}</td></tr>"
        for c in v.evidence.get("checks", []))
    score = (f"{r.classification.score:.3f} (threshold {r.classification.threshold:.2f})"
             if r.classification else "not computed (rejected before analysis)")
    extra = ""
    if "iou_vs_ground_truth" in r.artifacts:
        extra = f"<p class='fine'>Localization IoU against the known ground-truth mask: {r.artifacts['iou_vs_ground_truth']:.2f}</p>"
    t = r.timings_ms
    return f"""<section class="card"><h2>{name}</h2>{f'<p class="fine">{note}</p>' if note else ''}
<p><span class="badge {r.verdict}">{r.verdict}</span> &nbsp;security: <b>{v.status.value}</b>
{('(' + ', '.join(v.reason_codes) + ')') if v.reason_codes else ''}</p>
<p>{r.explanation}</p><div class="row"><div class="imgs">{imgs}</div></div>
<p>Image-level score: <b>{score}</b></p>{extra}
<details><summary>Security checks ({len(v.evidence.get('checks', []))}) and evidence</summary>
<table><tr><th>step</th><th>check</th><th>result</th></tr>{checks}</table>
<p class="fine">SHA-256 {v.sha256 or '-'} · key {v.evidence.get('key_id')} · policy {v.evidence.get('policy_version')}</p></details>
<p class="fine">Time: security {t.get('security_ms', 0):.1f} ms · signals {t.get('signals_ms', 0):.1f} ms ·
model {t.get('model_ms', 0):.1f} ms · total {t.get('total_ms', 0):.1f} ms (CPU)</p></section>"""


def render_html(items, title: str, intro: str = "") -> str:
    cards = "".join(_card(*it) for it in items)  # (nom, rapport, image[, note])
    lim = "".join(f"<li>{x}</li>" for x in LIMITATIONS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{_CSS}</style></head>
<body><main><h1>{title}</h1><p class="sub">{intro or 'Two-layer check: provenance (signature, hash, metadata, replay) then pixel forensics (noise residuals, ELA, ForensicUNet).'}</p>
{cards}<section class="card"><h2>Limits</h2><ul>{lim}</ul></section></main></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Two-layer analysis of one received image")
    ap.add_argument("--image", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sig")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--now", help="assessment time YYYY-MM-DDTHH:MM:SSZ (default: reference_time.txt)")
    ap.add_argument("--out", default="outputs/reports")
    a = ap.parse_args()
    rep = analyze(a.image, a.manifest, a.sig, config_dir=a.config_dir, now=a.now,
                  model_path=a.model, out_dir=a.out)
    print(f"{rep.verdict}: {rep.explanation}")
    print(f"Report: {rep.artifacts.get('report_html')}")
