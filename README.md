# Trust the Pixel: two-layer verification of Earth-observation imagery

TSYP14 Technical Challenge, Track 3. Anonymous Phase 1 submission.

A satellite image can be altered **after** its sender signs it (in transit, in storage) or **before** (by an insider or a compromised processing chain). This prototype checks both, on a laptop CPU:

1. **Layer 1, provenance** (`src/security`): format checks, an Ed25519-signed manifest, SHA-256 integrity, metadata plausibility and replay detection. Any change after signing is rejected, and a rejected image never reaches the models.
2. **Layer 2, pixel forensics** (`src/forensics`, `src/models`): noise residuals (SRM) and Error Level Analysis feed **ForensicUNet**, a 1.1 M-parameter detector that returns an image score and a pixel mask of the suspected edit.

```
Acquisition ──► Signing ──► Transfer ──► Receiver ──► Layer 1: provenance ──FAILED──► REJECTED
   (B1: edits here are signed as genuine)                     │ PASSED / REVIEW
                                                              ▼
                                       Layer 2: SRM + ELA + ForensicUNet ──► Forensic report
                                                                             (verdict, score, mask, evidence)
```

## Results (EuroSAT test split, 720 tiles)

| Model | AUC | False alarms | Mask IoU |
| --- | --- | --- | --- |
| Frozen EfficientNet-B0 (ImageNet) + heads, baseline | 0.718 | 17.5 % | 0.250 |
| **ForensicUNet (RGB + SRM + ELA)** | **0.877** | **0.4 %** | **0.598** |
| ForensicUNet, robust training | 0.820 | 5.8 % | 0.466 |

* Per attack (AUC, baseline → ForensicUNet): splice 0.95 → 1.00, colour shift 0.96 → 0.99, copy-move 0.65 → 0.94, **held-out blur** (never seen in training) 0.57 → 0.78.
* Evasion: under 10 laundering operations (JPEG, resampling, noise, blur, chains) the standard model's recall falls to 0.00–0.43; with robust training and a threshold capped at 5 % false alarms on validation it stays at 0.36–0.51. False alarms rise to 17–19 % under the strongest laundering.
* Cost: 33.6 ms per 64 × 64 image for both layers (30 images/s), 240 MB peak memory, 4.3 MB model, CPU only, no hardware purchase.
* Replayable demo: an authentic tile is accepted, a tile edited in transit is rejected (`HASH_MISMATCH`), a tile edited before signing passes layer 1 and is flagged and localized by layer 2 (score 0.98), and a replay is rejected (`REPLAY_ID`, `REPLAY_CONTENT`).

All manipulations are synthetic; see [Limitations](#limitations).

## Quick start

Requirements: **Python 3.12** (PyTorch has no wheels for 3.14 yet), no GPU. Results were produced on Windows; the code uses no Windows-specific API.

```powershell
py -3.12 -m venv .venv312                 # Linux/macOS: python3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1           # Linux/macOS: source .venv312/bin/activate
pip install -r requirements.txt
python -m pytest -q                       # 75 tests, a few minutes on CPU
```

`requirements.txt` pins the exact versions used for the results (Windows, PyTorch 2.14.1 CPU). On Linux, `pip` installs the CUDA build of PyTorch by default; it works on CPU but is a much larger download.

### Data

EuroSAT RGB (the official reference dataset for Track 3) is downloaded once, then the tampered set is built with exact masks (seed 0):

```powershell
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='data/raw', download=True)"
python -m src.preprocessing.build_tampered_eurosat
```

This writes `data/processed/eurosat_tampered.npz`: 1,200 source tiles, 2,640 images (train 1,440 / validation 480 / test 720). A source tile and its copies always stay in one split; blur appears only in the test split.

## Reproduce every result

| Result | Command | Output |
| --- | --- | --- |
| Baseline | `python -m src.models.train_eurosat` | `outputs/eurosat_run.json` |
| ForensicUNet | `python -m src.models.train_forensic` | `outputs/forensic_unet_run.json`, `outputs/forensic_unet.pt` |
| Evasion, standard model | `python -m src.models.evaluate_evasion` | `outputs/evasion_results.json` |
| Robust model | `python -m src.models.train_forensic --robust` | `outputs/forensic_unet_robust_run.json` |
| Evasion, robust model + 5 % cap | `python -m src.models.evaluate_evasion --model outputs/forensic_unet_robust.pt --calibration far5 --out outputs/evasion_results_robust_far5.json` | JSON + `outputs/forensic_unet_robust_far5.pt` |
| Two-layer demo | `python -m src.pipeline.demo` | `outputs/demo/index.html` |
| Benchmark | `python -m src.pipeline.benchmark --model outputs/forensic_unet_robust_far5.pt` | `outputs/benchmark.json` |
| Security specification suite | `cd src/security` then `python run_security_cases.py` | 53 / 53 cases |

Analyse any received image with its manifest:

```powershell
python -m src.pipeline.analyze --image X.png --manifest X.manifest.json --out outputs/reports
```

It writes a JSON report, a self-contained HTML report and the image with the suspected region in red. Images larger than 64 × 64 are analysed in tiles. Multiband or 16-bit GeoTIFFs go through the preprocessing first.

## Repository layout

```
src/
├── security/        Layer 1: assessment, specification test cases, adapter to the contracts
├── preprocessing/   loading, band selection, normalization, tamper generator, dataset builder
├── forensics/       SRM residuals, ELA, laundering (evasion) operations
├── features/        EfficientNet-B0 feature extractor (baseline)
├── models/          ForensicUNet, heads, losses, training, evaluation, evasion study
├── pipeline/        emulated sender (trust anchor), two-layer analysis, demo, benchmark
└── utils/           typed contracts between modules, metrics
docs/
├── threat_model.md  adversaries, STRIDE table, controls, evidence, residual risks
├── detector.md      signal processing, model and evaluation protocol
└── reference/       architecture diagram
tests/               75 automated tests
```

`data/` and `outputs/` are not versioned: everything in them is regenerated by the commands above.

## Limitations

* The manipulations are synthetic rectangles (splice, copy-move, colour shift, blur). Realistic edits and AI-generated scenes are not evaluated.
* Only one attack family (blur) is held out; recall on it stays low (about 0.25).
* EuroSAT has no scene identifier, so neighbouring tiles of one scene may fall into different splits: scores may be slightly optimistic.
* The signing keys are **disposable test keys** derived from public labels. They demonstrate the mechanism and must never protect real data; a deployment keeps the key in a TPM or HSM.
* A model score is evidence for an analyst, never proof of manipulation or authenticity.

## References

SRM residuals: Fridrich and Kodovsky, IEEE TIFS 2012; manipulation detection with noise streams: Zhou et al., CVPR 2018; U-Net: Ronneberger et al., MICCAI 2015; EuroSAT: Helber et al., IEEE JSTARS 2019; Ed25519: RFC 8032; STRIDE: Shostack, 2014.
