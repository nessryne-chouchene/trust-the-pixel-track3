# Trust the Pixel: two-layer verification of Earth-observation imagery

TSYP14 Technical Challenge, Track 3. Anonymous Phase 1 submission.

A satellite image can be altered **after** its sender signs it (in transit, in storage) or **before** (by an insider or a compromised processing chain). This prototype checks both, on a laptop CPU:

1. **Layer 1, provenance** (`src/security`): format checks, an Ed25519-signed manifest, SHA-256 integrity, metadata plausibility and replay detection. Any change after signing is rejected, and a rejected image never reaches the models.
2. **Layer 2, pixel forensics** (`src/models`): **ForensicUNet**, a 1.1 M-parameter encoder-decoder working on the RGB pixels at native resolution, returns an image score and a pixel mask of the suspected edit. Noise residuals (SRM) and Error Level Analysis were tested as extra inputs (`src/forensics`); the ablation below shows they did not help on our data.

```
Acquisition ──► Signing ──► Transfer ──► Receiver ──► Layer 1: provenance ──FAILED──► REJECTED
   (B1: edits here are signed as genuine)                     │ PASSED / REVIEW
                                                              ▼
                                       Layer 2: ForensicUNet (RGB) ──────► Forensic report
                                                                             (verdict, score, mask, evidence)
```

## Results (EuroSAT test split, 720 tiles)

ForensicUNet rows: mean ± standard deviation over 3 training seeds. Models are selected on validation AUC.

| Model | Validation AUC | Test AUC | False alarms | Mask IoU | AUC blur (held out) |
| --- | --- | --- | --- | --- | --- |
| Frozen EfficientNet-B0 (ImageNet) + heads, baseline | – | 0.718 | 17.5 % | 0.250 | 0.572 |
| **ForensicUNet, RGB (selected)** | **0.997 ± 0.002** | **0.916 ± 0.003** | **1.4 ± 0.9 %** | **0.648 ± 0.028** | **0.837 ± 0.007** |
| ForensicUNet, RGB + SRM + ELA | 0.985 ± 0.002 | 0.866 ± 0.011 | 2.9 ± 2.3 % | 0.565 ± 0.029 | 0.757 ± 0.021 |

* **Ablation (seed 0):** RGB 0.915 test AUC, RGB + SRM 0.880, RGB + ELA 0.901, RGB + SRM + ELA 0.877. On these synthetic edits, the noise traces add noise rather than evidence.
* **Per attack** (AUC, baseline → ForensicUNet RGB, seed 0): splice 0.953 → 1.000, colour shift 0.960 → 0.997, copy-move 0.648 → 0.995, **held-out blur** (never seen in training) 0.572 → 0.832.
* **Evasion:** laundering is the main weakness. One JPEG q90 recompression drops the standard model's AUC from 0.915 to 0.68, and light noise makes it flag 64 % of authentic tiles. Robust training plus a threshold capped at 5 % false alarms on validation raises AUC under all 10 laundering operations (0.66–0.82) and keeps 36–56 % of manipulated tiles detected; false alarms stay at 15–22 % under strong recompression and the resampling + JPEG chain.
* **Cost:** the detector needs 11 ms per 64 × 64 tile on a 4-thread laptop CPU (about 90 tiles/s), 238 MB peak memory, a 4.3 MB model, no GPU and no hardware purchase. The security stage took 23 to 78 ms per image across two runs on the same Windows laptop, dominated by file access (1.9 ms on a Linux machine).
* **Replayable demo:** an authentic tile is accepted (score 0.16), a tile edited in transit is rejected (`HASH_MISMATCH`), a tile edited before signing passes layer 1 and is flagged and localized by layer 2 (score 1.00), and a replay is rejected (`REPLAY_ID`, `REPLAY_CONTENT`).

All manipulations are synthetic; see [Limitations](#limitations).

## Quick start

Requirements: **Python 3.12** (PyTorch has no wheels for 3.14 yet), no GPU. Results were produced on Windows; the code uses no Windows-specific API.

```powershell
py -3.12 -m venv .venv312                 # Linux/macOS: python3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1           # Linux/macOS: source .venv312/bin/activate
pip install -r requirements.txt
python -m pytest -q                       # 83 tests, a few minutes on CPU
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
| Ablation + full model over 3 seeds | `python -m src.models.run_experiments` | `outputs/experiments/summary.md` |
| RGB model over 3 seeds | `python -m src.models.run_experiments --channels rgb --seeds 0 1 2` | `outputs/experiments/rgb_seed*/` |
| Robust RGB model | `python -m src.models.run_experiments --channels rgb --seeds 0 --robust` | `outputs/experiments/rgb_seed0_robust/` |
| Evasion, standard model | `python -m src.models.evaluate_evasion --model outputs/experiments/rgb_seed0/forensic_unet.pt --out outputs/experiments/rgb_seed0/evasion.json` | JSON |
| Evasion, robust model + 5 % cap | `python -m src.models.evaluate_evasion --model outputs/experiments/rgb_seed0_robust/forensic_unet_robust.pt --calibration far5 --out outputs/experiments/rgb_seed0_robust/evasion_far5.json` | JSON + `forensic_unet_robust_far5.pt` |
| Two-layer demo | `python -m src.pipeline.demo --model outputs/experiments/rgb_seed0_robust/forensic_unet_robust_far5.pt` | `outputs/demo/index.html` |
| Benchmark | `python -m src.pipeline.benchmark --model outputs/experiments/rgb_seed0_robust/forensic_unet_robust_far5.pt` | `outputs/benchmark.json` |
| Security specification suite | `cd src/security` then `python run_security_cases.py` | 53 / 53 cases |

Each training run takes 22 to 30 minutes on a laptop CPU.

Analyse any received image with its manifest:

```powershell
python -m src.pipeline.analyze --image X.png --manifest X.manifest.json --model outputs/experiments/rgb_seed0_robust/forensic_unet_robust_far5.pt --out outputs/reports
```

It writes a JSON report, a self-contained HTML report and the image with the suspected region in red. Images larger than 64 × 64 are analysed in tiles. Multiband or 16-bit GeoTIFFs (for example 13-band Sentinel-2) are accepted by layer 1 and reduced to RGB by the preprocessing.

## Repository layout

```
src/
├── security/        Layer 1: assessment, specification test cases, adapter to the contracts
├── preprocessing/   loading, band selection, normalization, tamper generator, dataset builder
├── forensics/       SRM residuals and ELA (tested in the ablation), laundering (evasion) operations
├── features/        EfficientNet-B0 feature extractor (baseline)
├── models/          ForensicUNet, heads, losses, training, experiments, evaluation, evasion study
├── pipeline/        emulated sender (trust anchor), two-layer analysis, demo, benchmark
└── utils/           typed contracts between modules, metrics
docs/
├── threat_model.md  adversaries, STRIDE table, controls, evidence, residual risks
├── detector.md      signal processing, model and evaluation protocol
└── reference/       architecture diagram
tests/               83 automated tests
```

`data/` and `outputs/` are not versioned: everything in them is regenerated by the commands above.

## Limitations

* The manipulations are synthetic rectangles (splice, copy-move, colour shift, blur). Realistic edits and AI-generated scenes are not evaluated.
* Only one attack family (blur) is held out, and it stays the hardest case (AUC 0.837).
* Noise residuals and ELA did not help on these edits; they may matter for realistic edits that leave no visible colour trace, which we have not tested.
* EuroSAT has no scene identifier, so neighbouring tiles of one scene may fall into different splits: scores may be slightly optimistic.
* The signing keys are **disposable test keys** derived from public labels. They demonstrate the mechanism and must never protect real data; a deployment keeps the key in a TPM or HSM.
* A model score is evidence for an analyst, never proof of manipulation or authenticity.

## References

SRM residuals (tested): Fridrich and Kodovsky, IEEE TIFS 2012; manipulation detection with noise streams: Zhou et al., CVPR 2018; U-Net: Ronneberger et al., MICCAI 2015; EuroSAT: Helber et al., IEEE JSTARS 2019; Ed25519: RFC 8032; STRIDE: Shostack, 2014.
