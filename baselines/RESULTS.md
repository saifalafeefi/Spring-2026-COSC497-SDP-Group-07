# Model Performance

Final model results on the UBC PPG dataset (Khalili et al.). All numbers are
on the held-out participant-disjoint test split — no patient overlap between
training and test data.

## Dataset

- 17,692 low-pass-filtered PPG windows (512 samples ≈ 10.24 s @ 50 Hz)
- 31 participants + dedicated off-body recordings
- Three classes: `cardiac` (12,214) / `non_cardiac` (3,642) / `occlusion` (1,836)
- Sensor placements: fingertip, finger base, wrist, off-body

## Architecture

Hybrid 1D CNN with engineered-feature side-branch:

- Convolutional branch: 4 conv blocks (20 / 32 / 48 / 64 filters) + residual block,
  followed by global average pooling
- Feature branch: 18 engineered features (time-domain statistics + PSD band powers
  + spectral entropy + cardiac-band power ratio), fed through a small MLP
- Branches concatenated and projected through a dense fusion layer to a 3-class softmax

**Total parameters: 112,121** (float32 Keras: 444 KB → int8 TFLite: **135 KB**).

## Training

- Focal loss (γ = 2.0) with inverse-frequency class weights
- Adam optimizer, cosine learning-rate schedule with 5% warmup
- 150 epochs (early-stopped at 76), batch size 64
- Augmentation: random time shift, magnitude scaling, Gaussian noise, time warping

## Results

| Metric | Value |
|---|---:|
| Overall accuracy | **92.4 %** |
| Macro F1 score | **0.868** |

### Per-class performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Cardiac | 0.960 | 0.940 | 0.950 |
| Non-Cardiac | 0.922 | 0.988 | 0.954 |
| Occlusion | 0.698 | 0.705 | 0.702 |

### Confusion matrix

|                | **Predicted Cardiac** | **Predicted Non-Cardiac** | **Predicted Occlusion** |
|---|---:|---:|---:|
| **True Cardiac** | 2,533 (94.0 %) | 41 (1.5 %) | 121 (4.5 %) |
| **True Non-Cardiac** | 6 (0.8 %) | 732 (98.8 %) | 3 (0.4 %) |
| **True Occlusion** | 99 (24.3 %) | 21 (5.2 %) | 287 (70.5 %) |

## Embedded deployment

The trained model converts cleanly to int8 TFLite:

- File size: **135 KB**
- Accuracy after quantization: 91.8 % (no meaningful degradation)
- Inference latency on PC: ~0.2 ms per window

This fits comfortably on an ESP32-S3 (512 KB SRAM + up to 8 MB PSRAM) and runs
in real time at the target 50 Hz sample rate.

## Reproducing

```bash
python3 baselines/train.py --preset phase_a    # ~12 min on CPU
python3 baselines/quantize.py                  # produce int8 TFLite
python3 baselines/inference_demo.py            # verify end-to-end
python3 baselines/make_plots.py                # regenerate result PNGs
```

Run artifacts are written to `baselines/runs/{timestamp}_phase_a/` with
`config.json`, `model.keras`, `model_int8.tflite`, `results.json`,
`history.json`, `training_curves.png`, and `confusion_matrix.png`.
