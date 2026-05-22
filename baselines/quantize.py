"""Convert a trained Keras model to TFLite int8 and verify it still works.

The whole point: confirm before SDP2 Week 6 that our model converts cleanly,
fits in the ESP32-S3 flash budget, and keeps acceptable accuracy after
quantization. Catching this now (vs at Week 6 with parts already shipping)
saves the project.

Run from repo root:
  python3 baselines/quantize.py                    # latest phase_a run
  python3 baselines/quantize.py --preset phase_a   # default — the only preset
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import tensorflow as tf

from data import CLASSES, load_dataset, participant_split
from features import extract_features, standardize

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))


def latest_run(preset: str) -> str:
    runs = sorted(glob.glob(os.path.join(HERE, "runs", f"*_{preset}")))
    if not runs:
        raise SystemExit(f"No runs/*_{preset} folder found. Run train.py first.")
    return runs[-1]


def _interp_inputs(interp):
    """Return signal-input and feature-input details, identifying by rank."""
    details = interp.get_input_details()
    sig = next(d for d in details if len(d["shape"]) == 3)
    feat = next((d for d in details if len(d["shape"]) == 2), None)
    return sig, feat


def _maybe_quantize(arr, det):
    """If the tensor expects int8, quantize using its (scale, zero_point)."""
    if det["dtype"] == np.int8:
        scale, zp = det["quantization"]
        return np.clip(np.round(arr / scale + zp), -128, 127).astype(np.int8)
    return arr.astype(np.float32)


def run_tflite_inference(interp, sig, feat):
    sig_in, feat_in = _interp_inputs(interp)
    interp.set_tensor(sig_in["index"], _maybe_quantize(sig, sig_in))
    if feat_in is not None and feat is not None:
        interp.set_tensor(feat_in["index"], _maybe_quantize(feat, feat_in))
    interp.invoke()
    out_det = interp.get_output_details()[0]
    out = interp.get_tensor(out_det["index"])
    if out_det["dtype"] == np.int8:
        scale, zp = out_det["quantization"]
        out = (out.astype(np.float32) - zp) * scale
    return out


def evaluate(interp, X_sig, X_feat, y, batch_log=None):
    n = len(y)
    correct = 0
    t0 = time.perf_counter()
    for i in range(n):
        sig = X_sig[i:i+1]
        feat = X_feat[i:i+1] if X_feat is not None else None
        out = run_tflite_inference(interp, sig, feat)
        pred = int(np.argmax(out))
        correct += int(pred == y[i])
        if batch_log and i and i % batch_log == 0:
            print(f"    {i}/{n} ...")
    dt = time.perf_counter() - t0
    return correct / n, dt / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="phase_a",
                    help="which preset's latest run to quantize (default: phase_a)")
    ap.add_argument("--n-eval", type=int, default=500,
                    help="number of test samples to evaluate (default: 500)")
    ap.add_argument("--n-calib", type=int, default=200,
                    help="number of calibration samples for int8 (default: 200)")
    args = ap.parse_args()

    run_dir = latest_run(args.preset)
    cfg_path = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "model.keras")
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"Loading run: {run_dir}")
    print(f"  model: {cfg['model']}  placement: {cfg['placement']}  "
          f"split: {cfg['split']}")

    # ---- load data and reproduce the splits used at training time ----
    ds = load_dataset(placement=cfg["placement"], zscore=True)
    tr, _, te = participant_split(ds, seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    calib_idx = rng.choice(tr, size=min(args.n_calib, len(tr)), replace=False)
    eval_idx = rng.choice(te, size=min(args.n_eval, len(te)), replace=False)

    model = tf.keras.models.load_model(model_path, compile=False)
    needs_features = any(inp.name.startswith("features") for inp in model.inputs)
    print(f"  loaded model: {model.name}  params: {model.count_params():,}  "
          f"needs_features: {needs_features}")

    if needs_features:
        ds_raw = load_dataset(placement=cfg["placement"], zscore=False)
        F_all = extract_features(ds_raw.X)
        F_tr_full, F_te_full = standardize(F_all[tr], F_all[te])
        F_tr = F_tr_full[np.searchsorted(np.sort(tr), calib_idx)]
        # Build by index in tr/te
        tr_pos = {i: k for k, i in enumerate(tr)}
        te_pos = {i: k for k, i in enumerate(te)}
        F_calib = np.stack([F_tr_full[tr_pos[i]] for i in calib_idx]).astype(np.float32)
        F_eval = np.stack([F_te_full[te_pos[i]] for i in eval_idx]).astype(np.float32)
    else:
        F_calib = F_eval = None

    X_calib = ds.X[calib_idx, :, None].astype(np.float32)
    X_eval = ds.X[eval_idx, :, None].astype(np.float32)
    y_eval = ds.y[eval_idx]

    # ---- baseline: Keras model accuracy on the eval subset ----
    if needs_features:
        keras_pred = model.predict([X_eval, F_eval], verbose=0).argmax(axis=1)
    else:
        keras_pred = model.predict(X_eval, verbose=0).argmax(axis=1)
    keras_acc = (keras_pred == y_eval).mean()
    print(f"\nKeras model accuracy (on {len(y_eval)} eval samples): {keras_acc:.4f}")

    # ---- convert to TFLite float32 (sanity) ----
    print("\n[1/2] Converting → TFLite float32 ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = os.path.join(run_dir, "model_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)
    fp32_size = len(tflite_fp32)
    print(f"  written: {fp32_path}  size: {fp32_size/1024:.1f} KB")

    # ---- convert to TFLite int8 (the one that actually goes on the chip) ----
    print("\n[2/2] Converting → TFLite int8 (with calibration) ...")

    def representative_dataset():
        for i in range(len(X_calib)):
            sample = {"signal": X_calib[i:i+1]}
            if needs_features:
                sample["features"] = F_calib[i:i+1]
            yield sample

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_int8 = converter.convert()
    except Exception as e:
        print(f"  FAILED to convert to int8: {e}")
        return
    int8_path = os.path.join(run_dir, "model_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
    int8_size = len(tflite_int8)
    print(f"  written: {int8_path}  size: {int8_size/1024:.1f} KB")

    # ---- evaluate both on the same eval subset ----
    print("\nEvaluating TFLite models (this may take a minute) ...")
    interp_fp32 = tf.lite.Interpreter(model_content=tflite_fp32)
    interp_fp32.allocate_tensors()
    fp32_acc, fp32_lat = evaluate(interp_fp32, X_eval, F_eval, y_eval)
    print(f"  TFLite float32: acc={fp32_acc:.4f}  latency={fp32_lat*1000:.2f} ms/sample (PC)")

    interp_int8 = tf.lite.Interpreter(model_content=tflite_int8)
    interp_int8.allocate_tensors()
    int8_acc, int8_lat = evaluate(interp_int8, X_eval, F_eval, y_eval)
    print(f"  TFLite int8:    acc={int8_acc:.4f}  latency={int8_lat*1000:.2f} ms/sample (PC)")

    # ---- summary ----
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  preset:                  {args.preset}")
    print(f"  Keras parameters:        {model.count_params():,}")
    print(f"  Keras accuracy (eval):   {keras_acc:.4f}")
    print(f"  TFLite fp32 size:        {fp32_size/1024:.1f} KB")
    print(f"  TFLite fp32 accuracy:    {fp32_acc:.4f}")
    print(f"  TFLite int8 size:        {int8_size/1024:.1f} KB  "
          f"({int8_size/fp32_size*100:.0f}% of fp32)")
    print(f"  TFLite int8 accuracy:    {int8_acc:.4f}  "
          f"(delta vs Keras: {(int8_acc-keras_acc)*100:+.2f} pp)")
    print(f"  Approx ESP32-S3 fit:     "
          f"{'OK' if int8_size < 500*1024 else 'TIGHT — review SRAM/flash'}")
    print("=" * 70)

    # ---- save a small summary alongside the run ----
    out = {
        "preset": args.preset,
        "keras_params": int(model.count_params()),
        "keras_accuracy": float(keras_acc),
        "tflite_fp32_size_bytes": int(fp32_size),
        "tflite_fp32_accuracy": float(fp32_acc),
        "tflite_int8_size_bytes": int(int8_size),
        "tflite_int8_accuracy": float(int8_acc),
        "tflite_int8_latency_ms_pc": float(int8_lat * 1000),
        "n_eval_samples": int(len(y_eval)),
    }
    with open(os.path.join(run_dir, "quantization.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary written: {os.path.join(run_dir, 'quantization.json')}")


if __name__ == "__main__":
    main()
