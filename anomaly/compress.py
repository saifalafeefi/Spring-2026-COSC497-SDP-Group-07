"""compress the autoencoder to TFLite and measure the accuracy cost (O7 prep).

converts the saved Keras autoencoder to TFLite (float32 and int8), then compares
the anomaly-detection metrics + model size + CPU latency. the int8 model is the
one that would ship to the Pi/ESP32; this reports what that compression costs.

    python3 -m anomaly.compress

writes ae_float32.tflite + ae_int8.tflite into anomaly/saved/.
"""
from __future__ import annotations

import os
import time
import numpy as np
import tensorflow as tf

from .run import load_all, NORMAL, POSITIVE
from .wesad import SUBJECTS
from .metrics import pr_auc, recall_at_specificity
from .infer import SAVE_DIR


def _z(X):
    return ((X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)).astype(np.float32)


def _convert(keras_model, rep=None, int8=False):
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def _tflite_mse(blob, Xz):
    """reconstruction MSE per window through a TFLite model (handles int8 io)."""
    interp = tf.lite.Interpreter(model_content=blob)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.resize_tensor_input(inp["index"], [Xz.shape[0], Xz.shape[1], 1])
    interp.allocate_tensors()
    x = Xz[..., None]
    if inp["dtype"] == np.int8:
        s, z = inp["quantization"]; x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    r = interp.get_tensor(out["index"]).astype(np.float32)
    if out["dtype"] == np.int8:
        s, z = out["quantization"]; r = (r - z) * s
    xf = Xz[..., None].astype(np.float32)             # error vs the same z-scored input
    return np.mean((r - xf) ** 2, axis=(1, 2))


def _latency_ms(blob, Xz, n=200):
    interp = tf.lite.Interpreter(model_content=blob)
    inp = interp.get_input_details()[0]
    interp.resize_tensor_input(inp["index"], [1, Xz.shape[1], 1])
    interp.allocate_tensors()
    x = Xz[:1, :, None]
    if inp["dtype"] == np.int8:
        s, z = inp["quantization"]; x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()                                   # warm
    t0 = time.perf_counter()
    for _ in range(n):
        interp.invoke()
    return (time.perf_counter() - t0) / n * 1000


def main():
    model = tf.keras.models.load_model(os.path.join(SAVE_DIR, "ae.keras"), compile=False)

    data = load_all(SUBJECTS, 60, 5)
    calm = np.concatenate([data[s][0][np.isin(data[s][1], list(NORMAL))] for s in data])
    stress = np.concatenate([data[s][0][np.isin(data[s][1], list(POSITIVE))] for s in data])
    Xz = _z(np.vstack([calm, stress]))
    y = np.r_[np.zeros(len(calm)), np.ones(len(stress))].astype(int)
    print(f"eval on {len(calm)} calm + {len(stress)} stress windows\n")

    rep_pool = _z(calm)
    def rep():
        for x in rep_pool[:200]:
            yield [x[None, :, None]]

    print("converting…")
    fp32 = _convert(model)
    try:
        i8 = _convert(model, rep=rep, int8=True)
        i8_ok = True
    except Exception as e:
        print(f"  full int8 failed ({type(e).__name__}); falling back to dynamic-range int8")
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        i8 = c.convert(); i8_ok = False
    open(os.path.join(SAVE_DIR, "ae_float32.tflite"), "wb").write(fp32)
    open(os.path.join(SAVE_DIR, "ae_int8.tflite"), "wb").write(i8)

    # keras (reference) scores
    rk = model.predict(Xz[..., None], batch_size=256, verbose=0)
    sk = np.mean((rk - Xz[..., None]) ** 2, axis=(1, 2))

    def report(name, scores, size, lat):
        print(f"  {name:22s} PR-AUC {pr_auc(y,scores):.3f}  "
              f"recall@90 {recall_at_specificity(y,scores):.3f}  "
              f"{size/1024:6.0f} KB  {lat}")

    keras_kb = os.path.getsize(os.path.join(SAVE_DIR, "ae.keras"))
    si8 = _tflite_mse(i8, Xz)                          # int8 scores (reused below)
    print("\nmodel                    metrics                          size    CPU latency/window")
    report("keras float32 (ref)", sk, keras_kb, "—")
    report("tflite float32", _tflite_mse(fp32, Xz), len(fp32), f"{_latency_ms(fp32,Xz):.2f} ms")
    tag = "tflite int8" if i8_ok else "tflite int8 (dyn-range)"
    report(tag, si8, len(i8), f"{_latency_ms(i8,Xz):.2f} ms")
    print(f"\nint8 vs keras: {keras_kb/len(i8):.1f}x smaller; the metric gap above is the compression cost.")

    # re-calibrate scorer.npz on the INT8 score distribution, so the live dashboard
    # (which now runs ae_int8.tflite) flags on the same scale as the deployed device.
    calm_i8 = si8[:len(calm)]
    np.savez(os.path.join(SAVE_DIR, "scorer.npz"),
             threshold=float(np.quantile(calm_i8, 0.90)),   # 90% specificity on calm
             win_len=int(Xz.shape[1]),
             ref_lo=float(np.median(calm_i8)),               # display: 0% level
             ref_hi=float(np.quantile(calm_i8, 0.99)))       # display: ~100% level
    print(f"saved int8-calibrated scorer → {SAVE_DIR}/scorer.npz  "
          f"(the dashboard now runs ae_int8.tflite — same model as the device)")


if __name__ == "__main__":
    main()
