# model performance

final model results on the [UBC PPG dataset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/HF0OS9)
(Khalili et al.). all numbers are on the held-out participant-disjoint test
split — no patient overlap between train and test.

## dataset

- 17,692 low-pass-filtered PPG windows (512 samples ≈ 10.24 s @ 50 Hz)
- 31 participants + dedicated off-body recordings
- three classes: `cardiac` (12,214) / `non_cardiac` (3,642) / `occlusion` (1,836)
- placements: fingertip, finger base, wrist, off-body

## architecture

hybrid 1D CNN with an engineered-feature side branch:

- conv branch: 4 conv blocks (20 / 32 / 48 / 64 filters) + a residual block, then global average pooling
- feature branch: 18 engineered features (time-domain stats + PSD band powers + spectral entropy + cardiac-band power ratio) through a small MLP
- branches concatenated and projected through a dense fusion layer to a 3-class softmax

**total parameters: 112,121** (float32 Keras: 444 KB → int8 TFLite: **135 KB**).

## training

- focal loss (γ = 2.0) with inverse-frequency class weights
- Adam, cosine LR schedule with 5% warmup
- 150 epochs (early-stopped at 76), batch size 64
- augmentation: random time shift, magnitude scaling, Gaussian noise, time warping

## results

| Metric | Value |
|---|---:|
| overall accuracy | **92.4%** |
| macro F1 | **0.868** |

### per-class

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| cardiac | 0.960 | 0.940 | 0.950 |
| non-cardiac | 0.922 | 0.988 | 0.954 |
| occlusion | 0.698 | 0.705 | 0.702 |

### confusion matrix

|                | **Pred. Cardiac** | **Pred. Non-Cardiac** | **Pred. Occlusion** |
|---|---:|---:|---:|
| **true cardiac** | 2,533 (94.0%) | 41 (1.5%) | 121 (4.5%) |
| **true non-cardiac** | 6 (0.8%) | 732 (98.8%) | 3 (0.4%) |
| **true occlusion** | 99 (24.3%) | 21 (5.2%) | 287 (70.5%) |

## embedded deployment

the model converts cleanly to int8 TFLite:

- size: **135 KB**
- accuracy after quantization: 91.8% (no meaningful drop)
- inference latency on PC: ~0.2 ms per window

fits comfortably on an ESP32-S3 (512 KB SRAM + up to 8 MB PSRAM) and runs in
real time at the 50 Hz target.

## reproducing

```bash
python3 baselines/train.py --preset phase_a    # ~12 min on CPU
python3 baselines/quantize.py                  # int8 TFLite
python3 baselines/inference_demo.py            # verify end-to-end
python3 baselines/make_plots.py                # regenerate result PNGs
```

artifacts land in `baselines/runs/{timestamp}_phase_a/`: `config.json`,
`model.keras`, `model_int8.tflite`, `results.json`, `history.json`,
`training_curves.png`, and `confusion_matrix.png`.
