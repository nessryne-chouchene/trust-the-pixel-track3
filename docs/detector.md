# Pixel-forensics detector (Track 3)

The security module (`src/security/`) proves **where an image comes from and that it was not changed in transit**. It cannot see a manipulation made **before** a trusted sender signed the image. This detector covers that gap by looking at the pixels themselves.

## Model: `ForensicUNet` (`src/models/forensic_unet.py`)

ForensicUNet is a light encoder-decoder (U-Net, Ronneberger et al. 2015) with 1.1 M parameters (4.3 MB). It has two outputs:

* a **pixel-level mask** (localization head) of where the image looks manipulated;
* an **image-level score** (classification head), computed from the global context and the mask statistics.

It works at the image's **native resolution** (64 × 64 for EuroSAT), with no upscaling to 224 × 224, and analyses larger images in overlapping 64 × 64 tiles. It trains and runs on a CPU. Outputs follow the shared contracts D and E (`src/utils/contracts.py`). Without a validated threshold, the detector returns scores only, never a decision.

A saved model records its input channels; `load_detector()` restores them, and the whole pipeline (analysis, demo, benchmark, evasion study) feeds each model the channels it expects.

## Signal processing, and what the ablation showed (`src/forensics/signals.py`)

Two classic forensic traces are available as extra input channels:

| Channels | What they capture |
|---|---|
| 3 SRM noise residuals | High-pass filters that remove the content and keep the sensor and compression noise (Fridrich & Kodovsky 2012). Noise streams are reported to help find manipulations (Zhou et al., CVPR 2018). |
| ELA map | Error Level Analysis, the technique behind Forensically (listed in the official resources): the image is recompressed as JPEG q90, and regions with a different compression history react differently. |

We tested whether these channels help, keeping the model, data and training identical and changing only the inputs (`python -m src.models.run_experiments`):

| Input channels | Validation AUC | Test AUC | False alarms | Mask IoU |
|---|---|---|---|---|
| **RGB** | **0.999** | **0.915** | 2.1 % | **0.668** |
| RGB + SRM | 0.981 | 0.880 | 8.3 % | 0.555 |
| RGB + ELA | 0.996 | 0.901 | 2.5 % | 0.618 |
| RGB + SRM + ELA | 0.984 | 0.877 | 0.4 % | 0.598 |

Each row uses seed 0. Over 3 training seeds, the RGB model scores 0.997 ± 0.002 on validation and 0.916 ± 0.003 on test, against 0.985 ± 0.002 and 0.866 ± 0.011 for RGB + SRM + ELA. **We therefore select RGB, on validation AUC.**

A likely explanation: our manipulations change colour and texture visibly, and SRM and ELA are fixed functions of the same pixels, which a network trained end to end can learn on its own. The extra channels then mostly add noise. This finding is specific to our synthetic edits. For realistic edits that leave no visible colour trace, the traces may matter again, so they remain available (`--channels`).

## Data and protocol

* **EuroSAT RGB** (the official reference dataset for Track 3), 1,200 source tiles. EuroSAT has no manipulation labels, so we create the manipulated copies ourselves, with **exact masks** (`src/preprocessing/tamper.py`): `splice`, `copy_move`, `color_shift` and `blur`.
* **No leakage:** a source tile and all its copies go to the same split (train 60 %, validation 20 %, test 20 %).
* **Held-out attack:** `blur` never appears in training or validation, only in the test. This simulates an attack the jury has not shown us.
* **Model selection and thresholds** use validation only (best F1, middle of the optimal range, or the 5 %-false-alarm cap). The test is used only to report.
* Seeds are fixed. Every run writes its full results to `outputs/`, and results are reported as mean ± standard deviation over 3 seeds.

## Adversarial validation (`src/forensics/evasion.py`, `src/models/evaluate_evasion.py`)

An attacker who knows the detector can "launder" the image after editing it. Our standard evasion scenario applies each operation to **all** test images, authentic and manipulated, with the threshold fixed beforehand on validation. The operations are:

* JPEG recompression (q90, q75, q60);
* resampling (× 0.75, × 0.5);
* Gaussian noise (σ 2, 5);
* Gaussian blur (radius 0.6, 1.0);
* resampling + JPEG;
* an **adaptive attack**: splices with feathered borders.

**Response:** robust training (`--robust`) launders half of each training batch at random. The threshold is then set as the lowest value that keeps at most 5 % of the authentic tiles in a clean + laundered validation set flagged (`--calibration far5`).

The result for the RGB model: robust training raises AUC under all 10 operations, from 0.53–0.81 to 0.66–0.82. Under light noise, false alarms fall from 64 % to 3.7 %. Under strong recompression and the resampling + JPEG chain, they stay at 15–22 %. The feathered-border attack evades neither model.

## Commands

See the README for the full table. The main ones are:

```powershell
python -m src.models.run_experiments --channels rgb --seeds 0 1 2
python -m src.models.run_experiments --channels rgb --seeds 0 --robust
python -m src.models.evaluate_evasion --model outputs/experiments/rgb_seed0_robust/forensic_unet_robust.pt --calibration far5 --out outputs/experiments/rgb_seed0_robust/evasion_far5.json
```

## Limits

* The manipulations are **synthetic** rectangles. The scores hold only in this setting. They say nothing about AI-generated imagery or carefully crafted edits.
* EuroSAT has no scene identifier, so neighbouring tiles of one Sentinel-2 scene can fall into different splits. Scores may be slightly optimistic.
* Generalization to an **unseen attack** (the held-out `blur`, AUC 0.837) and robustness to strong recompression are the main open problems. They are reported, not hidden.
* A model score is never proof of authenticity or manipulation.
