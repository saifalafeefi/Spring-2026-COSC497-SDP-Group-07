# edge-AI powered remote health monitoring system

**Khalifa University of Science and Technology** — Department of Computer Science
**COSC497: Senior Design Project**, Spring 2026
**Supervisor:** Dr. Emadeldeen Eldele

## team

| Name | ID |
|---|---|
| Khaleifah Alhefeiti | 100059431 |
| Khalfan Alantali | 100059479 |
| Mohammed Alremeithi | 100060448 |
| Saif Alafeefi | 100061144 |
| Zayed Alnuami | 100061300 |

## what it is

a wearable that runs AI on-device to catch cardiac anomalies and falls in real
time, then alerts family through a phone app. built for elderly and at-risk
people living on their own.

the model reads 10-second PPG (pulse) windows and sorts them into three states:
`cardiac` (normal), `non_cardiac` (sensor off-body), or `occlusion` (blood-flow
blockage — the danger flag). it hits **92.4% accuracy** on a participant-disjoint
split with ~112K parameters, quantized to **135 KB** for the edge.

## quick links

| Document | What it covers |
|---|---|
| [`baselines/RESULTS.md`](baselines/RESULTS.md) | model performance metrics |
| [`pipeline/README.md`](pipeline/README.md) | real-time pipeline + web dashboard |

## quick start

```bash
# install dependencies (one-time)
pip3 install -r baselines/requirements.txt

# train the model (~12 min on CPU)
python3 baselines/train.py --preset phase_a

# quantize to int8 TFLite for the device
python3 baselines/quantize.py --preset phase_a

# verify the inference API end-to-end
python3 baselines/inference_demo.py

# regenerate the 3 result PNGs
python3 baselines/make_plots.py

# real-time dashboard (web UI on localhost:8000)
pip3 install -r pipeline/requirements.txt
python3 pipeline/server.py
# open http://localhost:8000 — or http://<this-device-ip>:8000 from a phone
```

## repo layout

```
Code & Data/                         UBC PPG dataset (not in git — see below)
baselines/                           all ML code
  train.py                           trainer
  configs.py                         training config + presets
  models.py / losses.py / augment.py / features.py / data.py
  inference_lib.py                   Classifier API for runtime use
  inference_demo.py                  end-to-end verification
  quantize.py                        TFLite int8 conversion
  make_plots.py                      result PNG generator
  RESULTS.md                         performance metrics
  runs/                              trained model artifacts
pipeline/                            real-time inference demo
  replay.py                          50 Hz data-source simulator
  pipeline.py                        buffer + classifier glue
  server.py                          FastAPI + WebSocket dashboard
  static/                            browser UI (index.html + vendored uPlot)
  run_cli.py                         terminal-only runner
  make_demo_data.py                  build the curated demo file
  demo_data.csv                      92 s of curated PPG (ships with repo)
  README.md                          pipeline docs
01_data_overview.png                 dataset overview
02_training_progress.png             training curves
03_model_performance.png             confusion matrix + per-class metrics
README.md                            this file
```

## dataset (3.8 GB, not in git)

the training data (UBC PPG, Khalili et al.) is too big for GitHub, so it isn't
committed. to run the training code, download it from [Borealis Data](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/HF0OS9)
and drop it in the repo root, keeping this structure:

```
Code & Data/
├── PPG_Raw_Processed/        512-sample LP windows per participant
├── Classification/           original paper notebooks
├── Statistical_Analysis/     demographics, feature CSVs
├── Participant_P1/ ... P32/  raw per-participant PPG + ECG
└── NonParticipant/           raw off-body recordings
```

see [`baselines/RESULTS.md`](baselines/RESULTS.md) for the dataset details used in training.