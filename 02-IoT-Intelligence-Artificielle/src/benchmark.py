"""Measure the embedded-relevant cost of each exported model: file size and
average inference time on this machine.

The timing here is only indicative: a laptop/desktop CPU is much faster than an
ESP32, so these numbers cannot be used directly to predict on-device latency.
They are only useful to compare the two tflite versions against each other.

Example:
    python benchmark.py
"""

import json
import time
from pathlib import Path

import numpy as np
import platform
import tensorflow as tf

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

N_RUNS = 1000
N_WARMUP = 50


def benchmark_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]

    rng = np.random.default_rng(42)
    if input_details["dtype"] == np.int8:
        sample = rng.integers(-128, 127, size=input_details["shape"], dtype=np.int8)
    else:
        sample = rng.standard_normal(input_details["shape"]).astype("float32")

    for _ in range(N_WARMUP):
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()

    durations_ms = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        durations_ms.append((time.perf_counter() - start) * 1000)

    durations_ms = np.array(durations_ms)
    size_kb = Path(model_path).stat().st_size / 1024
    return {
        "size_kb": round(size_kb, 2),
        "mean_ms": round(float(durations_ms.mean()), 4),
        "p95_ms": round(float(np.percentile(durations_ms, 95)), 4),
    }


def main():
    results = {
        "cpu": platform.processor() or platform.machine(),
        "n_runs": N_RUNS,
        "models": {
            "model_float32.tflite": benchmark_tflite(MODELS_DIR / "model_float32.tflite"),
            "model_int8.tflite": benchmark_tflite(MODELS_DIR / "model_int8.tflite"),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"CPU: {results['cpu']} ({N_RUNS} runs per model, {N_WARMUP} warm-up runs discarded)\n")
    for name, stats in results["models"].items():
        print(f"{name:20s} size={stats['size_kb']:7.2f} KB  mean={stats['mean_ms']:.3f} ms  p95={stats['p95_ms']:.3f} ms")


if __name__ == "__main__":
    main()
