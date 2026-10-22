# Threat model: Track 3, Trust the Pixel

**Segment:** Earth-observation imagery, from the moment a product is generated to the moment an analyst or model uses it.
**Question it answers:** can a decision-maker trust that this image shows what the sensor saw?

## 1. System and trust boundaries

```
 [Sensor + on-board / ground processing] --(B1)--> [Signing: trust anchor] --(B2: transfer)--> [Receiver]
                                                                                                   |
                                    Layer 1, provenance (src/security): format, signature, hash, metadata, replay
                                                                                                   |
                                    Layer 2, pixels (src/forensics + ForensicUNet): local manipulation evidence
                                                                                                   |
                                                                                  [Forensic report -> analyst]
```

* **B1, before signing:** everything that happens to the pixels before the sender signs. The signature **cannot** see edits made here.
* **B2, after signing:** the network, storage, mirrors and intermediaries. Any change here breaks the signature or the hash.
* **Trust anchor:** an Ed25519 key per sender (`trusted_keys.json`, public keys only on the receiver side). In this prototype it is a software key (`src/pipeline/sender.py`, disposable test keys). A deployment would keep it in an HSM or TPM. The challenge rules allow a software or emulated anchor.

## 2. Assets

| Asset | Why it matters |
|---|---|
| Pixel content | Decisions in agriculture, insurance, water and disaster response depend on it |
| Acquisition metadata (time, sensor, location) | A true image with a false date or place is still a false observation |
| Signing keys | Whoever holds them can make any content "trusted" |
| Receiver configuration (trust store, policy, ledger) | Changing it changes what is accepted |
| Detector (weights, threshold) | A weakened detector fails silently, which the challenge statement singles out as the main danger |

## 3. Adversaries

| ID | Adversary | Capability | Goal |
|---|---|---|---|
| A1 | Network attacker | Reads and changes data in transit (B2) | Swap or alter an image or its metadata |
| A2 | Replayer | Captures and resends genuine traffic | Make an old observation look new; flood the receiver |
| A3 | Insider or compromised processing chain | Edits pixels **before** signing (B1) and has the product signed normally | Validly signed false content |
| A4 | Key thief | Holds a stolen private key | Sign arbitrary content as a trusted sender |
| A5 | Detector-aware adversary | Knows the detector; launders edits (recompression, resampling, noise, blur, feathered borders) | Evade layer 2 |
| A6 | Malformed-input attacker | Sends crafted files (decompression bomb, truncated or renamed files, oversized files) | Crash or exhaust the receiver |
| A7 | Metadata forger | Controls fields before signing, or sends unsigned data | False time, place or sensor |

## 4. Threats, controls and evidence (STRIDE)

| STRIDE | Threat | Adv. | Control in this prototype | Evidence (reproducible) | Residual risk |
|---|---|---|---|---|---|
| **S**poofing | Image signed by an unknown or forged key | A1 | Trust store lookup, Ed25519 verification over a domain-separated canonical manifest | Scenes 05 `SIG_INVALID`, 06 `KEY_UNKNOWN`, 13 `SIG_MISSING` | None known for the cryptography; depends on key custody |
| Spoofing | Use of a revoked or expired key | A4 | Status checked **before** verification; validity window on `signed_at` | Scene 07 `KEY_REVOKED`; edge cases 29, 30 `KEY_NOT_VALID` | Window between key theft and revocation |
| **T**ampering | Pixels changed after signing | A1 | SHA-256 of the received bytes compared with the signed hash | Scene 03, demo B → `HASH_MISMATCH`, REJECTED | None |
| Tampering | Manifest or metadata changed after signing | A1 | The signature covers the whole manifest | Scene 04 `SIG_INVALID`; edge case 42 | None |
| Tampering | Pixels changed **before** signing | A3 | **Layer 2:** SRM noise residuals + ELA + ForensicUNet (image score and pixel mask) | Demo C → SUSPECTED, mask over the edit; `outputs/forensic_unet_run.json` (per-attack AUC, IoU) | Attack types not seen in training (held-out `blur`); realistic or AI-generated edits not evaluated |
| Tampering | Detector evasion by laundering | A5 | Robust training (`--robust`); threshold calibrated on clean + laundered validation (`--calibration mixed`, `far5`) | `outputs/evasion_results*.json`: AUC, recall and false alarms for 10 laundering operations + feathered splice | Detection drops under strong recompression and resampling; adaptive white-box attacks not evaluated |
| **R**epudiation | Sender denies having sent an image | n/a | Signature with the sender's key; evidence trail (hash, key, policy version, every check) in each report | `evidence` field of every report | Relies on the key not being shared |
| Repudiation / replay | Old genuine image resent as new | A2 | Ledger of accepted `image_id` **and** content hashes; only PASSED images are recorded | Scenes 08, 09; demo D → `REPLAY_ID`, `REPLAY_CONTENT` | Ledger must be protected and kept between restarts |
| **I**nformation disclosure | Leak of private keys from the receiver | n/a | Receiver holds public keys only; a trust store with private material is refused | Invariant `INV_no_private_keys` | Sender-side key custody is out of scope |
| **D**enial of service | Decompression bomb, huge or malformed files | A6 | Size limits, magic bytes, decoder restricted to the claimed format, pixel limit **before** decoding, bomb guard | Scene 12 `FORMAT_MAGIC`; edge cases 14–21 | Volumetric flooding at network level is out of scope |
| **E**levation / plausibility | False time, place or sensor | A7 | Signed metadata + plausibility checks (latitude, longitude, timestamp format, future, staleness, sensor allowed for the key, file name) → REVIEW by a human | Scenes 10, 11; edge cases 32–41 | A signer who lies with plausible values (A3/A4) is not caught by these checks |

## 5. Design decisions that follow from this model

1. **Fail closed on provenance.** A FAILED image is not passed to the detector or the analyst: analysing untrusted content as if it were a product would be a silent failure.
2. **Two independent layers.** Cryptography covers B2 (after signing) completely. Pixel forensics is the only control on B1 (before signing). Neither replaces the other.
3. **Scores are evidence, not verdicts.** The report gives the score, the threshold, how it was chosen, and the mask, so a human can check the evidence.
4. **False alarms are a cost.** A detector that flags real products erodes trust as surely as one that misses fakes. Thresholds are fixed on validation data only, and the false-alarm rate is reported for every condition.
5. **Everything is measured on a laptop CPU** (`src/pipeline/benchmark.py`): no hardware purchase, and the security overhead is reported separately.

## 6. Out of scope and open issues

* Sensor-level attacks (spoofed GNSS or navigation, physical interference) and on-board compromise: covered by Tracks 1 and 2.
* Fully synthetic (AI-generated) scenes: not evaluated. A planned extension compares an image with an independent acquisition of the same place and date, for example in the Copernicus Browser listed in the official resources.
* Integrity of the detector itself: the weights and the policy should be pinned by hash and signed like the images. This is recommended, not yet implemented.
* The ledger and the trust store must be stored with integrity protection in a deployment.
