# Pixel-forensics detector (Track 3)

The security module (`src/security/`) proves **where an image comes from and that it was not changed in transit**. It cannot see a manipulation made **before** a trusted sender signed the image. This detector covers that gap: it looks at the pixels themselves.

## Signal processing: forensic traces (`src/forensics/signals.py`)

A local edit leaves traces that the eye does not see but the signal keeps. The detector receives 7 channels at the image's **native resolution** (no upscaling):

| Channels | What they capture |
|---|---|
| R, G, B | image content (centred and scaled) |
| 3 SRM noise residuals | high-pass filters that remove the content and keep the noise texture of the sensor and of the compression (Fridrich & Kodovsky 2012; used for manipulation detection by Zhou et al., CVPR 2018). A pasted or smoothed area has a different noise texture. |
| ELA map | Error Level Analysis (the technique behind Forensically, listed in the official resources): the image is recompressed as JPEG q90; regions with a different compression history react differently. |

These maps are **inputs**, not proof.

## Model: `ForensicUNet` (`src/models/forensic_unet.py`)

A light encoder-decoder (U-Net, Ronneberger et al. 2015) with about 1.1 M parameters (4.3 MB), two outputs:

* a **pixel-level mask** (localization head): where the image looks manipulated;
* an **image-level score** (classification head), computed from the global context and the mask statistics.

It is trained and runs on a CPU. Outputs follow the shared contracts D and E (`src/utils/contracts.py`). Without a validated threshold, no decision is returned, only scores.

## Data and protocol

* **EuroSAT RGB** (official reference dataset for Track 3), 1,200 source tiles. EuroSAT has no manipulation labels, so we create the manipulated copies ourselves, with **exact masks** (`src/preprocessing/tamper.py`): `splice`, `copy_move`, `color_shift`, `blur`.
* **No leakage**: a source tile and all its copies go to the same split (train 60 %, validation 20 %, test 20 %).
* **Held-out attack**: `blur` never appears in training or validation, only in the test. This simulates an attack the jury has not shown us.
* **Threshold** chosen on validation only (best F1, middle of the optimal range), then applied unchanged to the test.
* Seeds are fixed. Every run writes its full results to `outputs/*.json`.

## Adversarial validation (`src/forensics/evasion.py`, `src/models/evaluate_evasion.py`)

An attacker who knows the detector "launders" the image after editing it. Our standard evasion scenario applies each operation to **all** test images (authentic and manipulated), with the threshold fixed from clean validation:

* JPEG recompression (q90, q75, q60);
* resampling (×0.75, ×0.5);
* Gaussian noise (σ 2, 5);
* Gaussian blur (r 0.6, 1.0);
* resampling + JPEG chain;
* **adaptive attack**: splices with feathered borders.

**Response**: robust training (`--robust`), where half of each training batch is laundered at random. The evaluation reports detection, false alarms and IoU for each operation, before and after this response.

## Commands

```powershell
python -m src.models.train_eurosat                       # baseline: frozen ImageNet EfficientNet-B0 + heads
python -m src.models.train_forensic                      # ForensicUNet (RGB + SRM + ELA)
python -m src.models.evaluate_evasion                    # evasion table for the standard model
python -m src.models.train_forensic --robust             # response: robust training
python -m src.models.evaluate_evasion --model outputs/forensic_unet_robust.pt --out outputs/evasion_results_robust.json
```

## Limits

* The manipulations are **synthetic** (rectangular regions). The scores are valid only in this setting. They say nothing about AI-generated imagery or carefully crafted edits.
* EuroSAT has no scene identifier: neighbouring tiles of one Sentinel-2 scene can fall into different splits, so scores may be slightly optimistic.
* Generalization to an **unseen attack** (held-out `blur`) is the main open problem. It is reported, not hidden.
* A model score is never proof of authenticity or manipulation.
